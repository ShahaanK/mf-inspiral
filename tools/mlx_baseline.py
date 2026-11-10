#!/usr/bin/env python
"""GPU baseline: the standard FFT matched filter written in MLX (no matchedfilter).

Modes (docs/design-fast.md §6):
  full    stock's layout: for each (template, 512 s segment), the product of
          the overwhitened segment and the conjugate template at 2^20 points, an
          inverse FFT, |z|^2, and the loudest sample per 2048-sample bin of the
          analyze region (max and argmax), compared with each template's threshold.
  blocks  mf_fast's layout: the segment's series cut into 65536-sample blocks
          (same block layout and template batches as mf_fast, from a saved cache),
          a forward FFT per block, the product with each scaled template, an
          inverse FFT, |z|^2, and per-bin max/argmax over each block's valid window.

Templates are prepared on the CPU and are not timed (in full mode they are the
same tapered templates as reference A, generated per batch), and neither is
their upload. gpu_s, the number compared with mf_fast's search_filter, is the
GPU work of each batch, synchronised with mx.eval: product, inverse FFT, |z|^2,
per-bin max and argmax, the complex value at the argmax, the threshold, and the
copy of those per-bin results to the host. That is what mf_fast's filter step
does, except that mf_fast also forward-transforms its blocks (stock's layout
transforms each segment once, before any template).

Full mode then clusters on the host exactly as reference A and mf_fast do
(PyCBC's symmetric rule per template and segment, findchirp clustering across
segments) and times that apart as post_s (compare mf_fast's search_stitch +
search_cluster). --hdf-output writes the triggers for check A. Blocks mode is a
timing baseline only: it gathers the same per-bin results but doesn't cluster.

Each mode checks its first batch against float64 numpy and prints the largest
relative difference of the bin maxima above threshold.

Usage: mlx_baseline.py --mode full|blocks --output RESULT.json [--tmpl-batch 16]
         [--cache CACHE.npz (blocks)] [--max-batches N]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy

sys.path.insert(0, '/Users/szkhan/projects/mf-inspiral')
import mlx.core as mx  # noqa: E402
from pycbc import waveform  # noqa: E402
from pycbc.events.eventmgr import findchirp_cluster_over_window  # noqa: E402
from pycbc.types import complex64, zeros  # noqa: E402

from mf_fast.frontend import FrontEnd  # noqa: E402
from mf_fast.search import Search, block_layout  # noqa: E402
from mf_fast.output import write  # noqa: E402
from mf_fast.templates import TemplateCache, tapered_template, sigmasq  # noqa: E402
from mf_inspiral.control import symmetric_cluster  # noqa: E402

S = '/Users/szkhan/projects/example-workflow/search'
BIN = 2048
THR = 3.8


def frontend():
    return FrontEnd(S + '/H-H1_LOSC_CLN_4_V1-1186740069-3584.gwf', 'H1:LOSC-STRAIN',
                    S + '/bank.hdf', 1186740108, 1186741908, 1186740140, 1186741892)


def binmax(p, nb):
    """(..., nb*BIN) power -> (max, argmax) over each bin."""
    q = p.reshape(p.shape[:-1] + (nb, BIN))
    return mx.max(q, axis=-1), mx.argmax(q, axis=-1)


def peaks(z, p, a0, nb):
    """Per-bin max power, argmax, and the complex value there, over [a0, a0+nb*BIN)."""
    m, i = binmax(p[:, a0:a0 + nb * BIN], nb)
    ii = i[..., None].astype(mx.int32)
    zs = z[:, a0:a0 + nb * BIN]
    shape = zs.shape[:-1] + (nb, BIN)
    re = mx.take_along_axis(mx.real(zs).reshape(shape), ii, axis=-1)[..., 0]
    im = mx.take_along_axis(mx.imag(zs).reshape(shape), ii, axis=-1)[..., 0]
    return m, i, re, im


class Meta(object):
    """The cache attributes output.write reads."""

    def __init__(self, rows, hashes, duration):
        self.rows, self.hashes, self.duration = rows, hashes, duration


def run_full(fe, B, max_batches, hdf_output=None):
    bank = waveform.FilterBank(S + '/bank.hdf', fe.flen, fe.delta_f, low_frequency_cutoff=None,
                               dtype=complex64, out=zeros(fe.tlen, dtype=complex64))
    N = fe.tlen
    D = mx.array(numpy.stack(fe.whitened))                   # (nseg, N) complex64
    mx.eval(D)
    an = [seg.analyze for seg in fe.segments]
    gpu, post, pairs, valid, check = 0.0, 0.0, 0, 0, None
    T = len(bank)
    found = [[] for _ in range(T)]
    duration = numpy.zeros(T, numpy.float32)
    for bi, t0 in enumerate(range(0, T, B)):
        if max_batches and bi >= max_batches:
            break
        rows = list(range(t0, min(t0 + B, T)))
        Hs = numpy.zeros((len(rows), N), numpy.complex64)
        sig = numpy.zeros((len(rows), len(fe.segments)))
        for k, ti in enumerate(rows):
            H, _ = tapered_template(bank, ti, fe.flow, 4.0, 4.0)
            Hf = numpy.zeros(N, numpy.complex128)
            Hf[:fe.flen] = H
            Hs[k] = Hf
            sig[k] = [sigmasq(Hf, p, fe.kmin, fe.kmax, fe.delta_f) for p in fe.psds]
            row = bank.table[ti]
            duration[ti] = waveform.get_waveform_filter_length_in_time(
                approximant=bank.approximant(ti), mass1=row['mass1'], mass2=row['mass2'],
                spin1z=row['spin1z'], spin2z=row['spin2z'], f_lower=fe.flow)
        Hc = mx.conj(mx.array(Hs))
        mx.eval(Hc)
        t = time.perf_counter()
        host = []
        for s, a in enumerate(an):
            norm = 4 * fe.delta_f / numpy.sqrt(sig[:, s])
            thr2 = mx.array(((THR / norm) ** 2).astype(numpy.float32))[:, None]
            z = mx.fft.ifft(D[s][None, :] * Hc, norm='forward')      # unnormalised inverse
            p = mx.real(z) ** 2 + mx.imag(z) ** 2
            nb = (a.stop - a.start) // BIN
            m, i, re, im = peaks(z, p, a.start, nb)
            hit = m > thr2
            mx.eval(m, i, re, im, hit)          # per segment: bounds GPU memory to one batch
            host.append((numpy.array(m), numpy.array(i), numpy.array(re), numpy.array(im),
                         numpy.array(hit), norm, nb))
        gpu += time.perf_counter() - t
        # host post-processing, as reference A and mf_fast: PyCBC's symmetric rule per
        # (template, segment), then findchirp clustering across segments (timed apart)
        t = time.perf_counter()
        for s, (m, i, re, im, hit, norm, nb) in enumerate(host):
            a, seg = an[s], fe.segments[s]
            for k, ti in enumerate(rows):
                idx = numpy.where(hit[k], a.start + numpy.arange(nb) * BIN + i[k], -1)
                val = numpy.where(hit[k], re[k] + 1j * im[k], 0).astype(numpy.complex64)
                keep = symmetric_cluster(idx, val, nb)
                if keep.any():
                    found[ti].append((seg.seg_slice.start + idx[keep],
                                      (val[keep] * norm[k]).astype(numpy.complex64),
                                      numpy.full(keep.sum(), sig[k, s], numpy.float32)))
        post += time.perf_counter() - t
        pairs += len(rows) * len(an)
        valid += len(rows) * sum(a.stop - a.start for a in an)
        if check is None:
            # float64 numpy: template rows[0], segment 0
            a = an[0]
            nb = (a.stop - a.start) // BIN
            z = numpy.fft.ifft(fe.whitened[0].astype(complex) * numpy.conj(Hs[0].astype(complex))) * N
            ref = (numpy.abs(z[a.start:a.start + nb * BIN]) ** 2).reshape(nb, BIN).max(1)
            got = host[0][0][0]
            norm0 = 4 * fe.delta_f / math.sqrt(sig[0, 0])
            mk = ref > (THR / norm0) ** 2
            check = float(numpy.max(numpy.abs(got[mk] - ref[mk]) / ref[mk])) if mk.any() else 0.0
    t = time.perf_counter()
    events = []
    for ti in range(T):
        if found[ti]:
            tiv = numpy.concatenate([f[0] for f in found[ti]])
            sv = numpy.concatenate([f[1] for f in found[ti]])
            sg = numpy.concatenate([f[2] for f in found[ti]])
            keep = findchirp_cluster_over_window(tiv, sv, BIN)
            events.append((ti, tiv[keep], sv[keep], sg[keep]))
    post += time.perf_counter() - t
    ntrig = None
    if hdf_output:
        meta = Meta(numpy.arange(T), numpy.asarray(bank.table.template_hash, dtype=numpy.int64),
                    duration)
        ntrig = write(hdf_output, 'H1', events, meta, fe.gps_start, fe.sample_rate,
                      fe.trig_start, fe.trig_end)
    return dict(mode='full', n=N, pairs=pairs, valid_samples=valid, gpu_s=gpu,
                post_s=post, triggers=ntrig, all_templates=not max_batches or None,
                ms_per_pair=1e3 * gpu / max(pairs, 1),
                check_max_rel_binmax_power=check, tmpl_batch=B)


def run_blocks(fe, B, cache_path, max_batches):
    c = TemplateCache.load(cache_path, fe, S + '/bank.hdf')
    n = c.n
    sr = Search(fe, c, tmpl_batch=B)
    # zero-padded by n: the last block may run past the segment's end, and an MLX
    # gather out of bounds isn't checked
    series = [mx.array(numpy.concatenate([x, numpy.zeros(n, x.dtype)])) for x in fe.series]
    mx.eval(*series)
    gpu, prep, pairs, nvalid, check = 0.0, 0.0, 0, 0, None
    for bi, batch in enumerate(sr.batches()):
        if max_batches and bi >= max_batches:
            break
        L, R = int(c.L[batch].max()), int(c.R[batch].max())
        for s, seg in enumerate(fe.segments):
            t = time.perf_counter()
            starts, ws, we, valid = block_layout(seg.analyze, L, R, n, BIN)
            idx = starts[:, None] + numpy.arange(n)[None, :]
            scale = numpy.array([c.block_scale(j, s) for j in batch], numpy.float32)
            Hc = mx.conj(mx.array(c.spectra[batch] * scale[:, None]))      # (B, n)
            gidx = mx.array(idx.astype(numpy.int32))
            mask = numpy.zeros((len(starts), valid), numpy.float32)
            for b in range(len(starts)):
                mask[b, :we[b] - ws[b]] = 1
            maskg = mx.array(mask)
            mx.eval(Hc, gidx, maskg)
            prep += time.perf_counter() - t
            t = time.perf_counter()
            X = mx.fft.fft(series[s][gidx])                                  # (nblk, n)
            z = mx.fft.ifft(X[None, :, :] * Hc[:, None, :])                  # series convention
            p = mx.real(z) ** 2 + mx.imag(z) ** 2
            p = p[:, :, L:L + valid] * maskg[None]
            m, i = binmax(p, valid // BIN)
            ii = i[..., None].astype(mx.int32)
            zs = z[:, :, L:L + valid]
            shape = zs.shape[:-1] + (valid // BIN, BIN)
            re = mx.take_along_axis(mx.real(zs).reshape(shape), ii, axis=-1)[..., 0]
            im = mx.take_along_axis(mx.imag(zs).reshape(shape), ii, axis=-1)[..., 0]
            hit = m > THR ** 2
            mx.eval(m, i, re, im, hit)
            hm = numpy.array(m)
            numpy.array(i), numpy.array(re), numpy.array(im), numpy.array(hit)
            gpu += time.perf_counter() - t
            pairs += len(starts) * len(batch)
            nvalid += (seg.analyze.stop - seg.analyze.start) * len(batch)
            if check is None:
                # float64 numpy for the batch's first template, first block
                Hn = (c.spectra[batch[0]] * scale[0]).astype(complex)
                blk = numpy.array(series[s])[starts[0]:starts[0] + n].astype(complex)
                zz = numpy.fft.ifft(numpy.fft.fft(blk) * numpy.conj(Hn))
                ref = (numpy.abs(zz[L:L + valid]) ** 2).reshape(-1, BIN).max(1)
                got = hm[0, 0]
                mk = ref > THR ** 2
                check = float(numpy.max(numpy.abs(got[mk] - ref[mk]) / ref[mk])) if mk.any() else 0.0
    return dict(mode='blocks', n=n, pairs=pairs, valid_samples=nvalid, gpu_s=gpu,
                host_prep_s=prep,
                ms_per_pair=1e3 * gpu / max(pairs, 1), check_max_rel_binmax_power=check,
                tmpl_batch=B)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['full', 'blocks'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--tmpl-batch', type=int, default=16)
    p.add_argument('--cache')
    p.add_argument('--max-batches', type=int, default=0)
    p.add_argument('--hdf-output',
                   help="full mode: write the clustered triggers here (mf_fast's layout) "
                        "for check A against reference A; needs all templates")
    a = p.parse_args()
    print("MLX", mx.__version__, "default device", mx.default_device(), flush=True)
    t0 = time.time()
    fe = frontend()
    t_fe = time.time() - t0
    if a.mode == 'full':
        if a.hdf_output and a.max_batches:
            raise SystemExit("--hdf-output needs every template (no --max-batches)")
        res = run_full(fe, a.tmpl_batch, a.max_batches, a.hdf_output)
    else:
        if not a.cache:
            raise SystemExit("--cache (a saved n=65536 mf_fast cache) is required for blocks")
        res = run_blocks(fe, a.tmpl_batch, a.cache, a.max_batches)
    res.update(front_end_s=t_fe, total_s=time.time() - t0, mlx=mx.__version__,
               device=str(mx.default_device()),
               env={k: v for k, v in os.environ.items() if k.startswith(('MF_', 'OMP_'))})
    print(json.dumps(res, indent=1))
    with open(a.output, 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
