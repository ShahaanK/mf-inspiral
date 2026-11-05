#!/usr/bin/env python
"""Reference A: the tapered templates filtered at full segment length (2^20).

Same front end, templates, sigma^2 and output layout as mf_fast, but filtered
the stock way: one matchedfilter run() per (template, segment) at the segment's
full length, as mf_inspiral's validated MFControl does, with PyCBC's symmetric
rule and cross-segment findchirp clustering. Short blocks must reproduce this
within 1e-5 (check A, docs/design-fast.md §2).

Usage: ref_full_length.py OUT.hdf [--timing-json T.json]   (C3 job)
"""
import argparse
import json
import math
import sys
import time

import numpy

sys.path.insert(0, '/Users/szkhan/projects/mf-inspiral')
import matchedfilter as mf  # noqa: E402
from pycbc import waveform  # noqa: E402
from pycbc.events.eventmgr import findchirp_cluster_over_window  # noqa: E402
from pycbc.types import complex64, zeros  # noqa: E402

from mf_fast.frontend import FrontEnd  # noqa: E402
from mf_fast.output import write  # noqa: E402
from mf_fast.templates import tapered_template, sigmasq  # noqa: E402
from mf_inspiral.control import symmetric_cluster  # noqa: E402

S = '/Users/szkhan/projects/example-workflow/search'


class Meta(object):
    """The cache attributes output.write reads."""

    def __init__(self, rows, hashes, duration):
        self.rows, self.hashes, self.duration = rows, hashes, duration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output')
    p.add_argument('--timing-json')
    p.add_argument('--taper-low', type=float, default=4.0)
    p.add_argument('--taper-high', type=float, default=4.0)
    a = p.parse_args()
    t0 = time.time()
    fe = FrontEnd(S + '/H-H1_LOSC_CLN_4_V1-1186740069-3584.gwf', 'H1:LOSC-STRAIN',
                  S + '/bank.hdf', 1186740108, 1186741908, 1186740140, 1186741892)
    bank = waveform.FilterBank(S + '/bank.hdf', fe.flen, fe.delta_f,
                               low_frequency_cutoff=None, dtype=complex64,
                               out=zeros(fe.tlen, dtype=complex64))
    N, bs = fe.tlen, 2048
    filt = mf.MatchedFilter(N, ndata=len(fe.segments), ntemplates=1, device='cpu')
    filt.set_data(numpy.stack(fe.whitened))
    T = len(bank)
    hashes = numpy.asarray(bank.table.template_hash, dtype=numpy.int64)
    duration = numpy.zeros(T, numpy.float32)
    events, tgen, tfilt = [], 0.0, 0.0
    for ti in range(T):
        t = time.time()
        H, _ = tapered_template(bank, ti, fe.flow, a.taper_low, a.taper_high)
        Hf = numpy.zeros(N, numpy.complex128)
        Hf[:fe.flen] = H
        row = bank.table[ti]
        duration[ti] = waveform.get_waveform_filter_length_in_time(
            approximant=bank.approximant(ti), mass1=row['mass1'], mass2=row['mass2'],
            spin1z=row['spin1z'], spin2z=row['spin2z'], f_lower=fe.flow)
        filt.set_templates(Hf.astype(numpy.complex64)[None, :])
        tgen += time.time() - t
        t = time.time()
        found = []
        for s, seg in enumerate(fe.segments):
            sig = sigmasq(Hf, seg.psd, fe.kmin, fe.kmax, fe.delta_f)
            norm = 4.0 * fe.delta_f / math.sqrt(sig)
            an = seg.analyze
            idx, val = filt.run(binsize=bs, threshold=3.8 / norm, window=(an.start, an.stop),
                                data=(s, 1), raw=True)
            index, value = idx[0, 0], val[0, 0]
            keep = symmetric_cluster(index, value, len(index))
            if keep.any():
                found.append((seg.seg_slice.start + index[keep],
                              (value[keep] * norm).astype(numpy.complex64),
                              numpy.full(keep.sum(), sig, numpy.float32)))
        if found:
            tiv = numpy.concatenate([f[0] for f in found])
            sv = numpy.concatenate([f[1] for f in found])
            sg = numpy.concatenate([f[2] for f in found])
            k = findchirp_cluster_over_window(tiv, sv, bs)
            events.append((ti, tiv[k], sv[k], sg[k]))
        tfilt += time.time() - t
        if (ti + 1) % 50 == 0:
            print("%d/%d  %.0f s" % (ti + 1, T, time.time() - t0), flush=True)
    meta = Meta(numpy.arange(T), hashes, duration)
    ntrig = write(a.output, 'H1', events, meta, fe.gps_start, fe.sample_rate,
                  fe.trig_start, fe.trig_end)
    info = dict(triggers=ntrig, templates_s=tgen, filter_s=tfilt, total_s=time.time() - t0)
    print(json.dumps(info))
    if a.timing_json:
        json.dump(info, open(a.timing_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
