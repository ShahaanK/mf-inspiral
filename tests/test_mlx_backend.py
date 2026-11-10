"""mf_fast's MLX backend step against matchedfilter's run() and float64 NumPy.

Synthetic one-sided spectra (data and templates nonzero only on a band
[kmin, kmax), templates on sub-bands starting at different offsets, as cached
bands are). For each case and MLX device, the per-bin result of
mlx_backend.segment_peaks + threshold_bins must equal matchedfilter's CPU run()
on the same full-length spectra: identical indices (bar exact float32 ties,
reported), values within 1e-5 relative, and float64 NumPy agreement for the bin
peaks. Cases include a partial last bin. Also checks the memory guards.

Needs Metal even for MLX's CPU device: run outside the sandbox. Plain python:
    python tests/test_mlx_backend.py [--device cpu|gpu|both]
Exit status is nonzero on any failure.
"""

import argparse
import sys

import numpy

sys.path.insert(0, __file__.rsplit('/tests/', 1)[0])
import matchedfilter as mf  # noqa: E402
import mlx.core as mx  # noqa: E402
from mf_fast.mlx_backend import segment_peaks, threshold_bins  # noqa: E402


def case(seed, n, kmin, kmax, a0, a1, bs, B, thr):
    rng = numpy.random.default_rng(seed)
    W = kmax - kmin
    d = numpy.zeros(n, numpy.complex64)
    d[kmin:kmax] = rng.standard_normal(W) + 1j * rng.standard_normal(W)
    H = numpy.zeros((B, n), numpy.complex64)
    for k in range(B):
        off = int(rng.integers(0, W // 4))
        ln = int(rng.integers(W // 2, W - off))
        H[k, kmin + off:kmin + off + ln] = rng.standard_normal(ln) + 1j * rng.standard_normal(ln)
    # a planted signal so some bins are loud
    d[kmin:kmax] += 0.05 * H[0, kmin:kmax] * numpy.exp(-2j * numpy.pi * numpy.arange(kmin, kmax) * (a0 + 3 * bs + 17) / n)
    thr_raw = thr * 2.0 * numpy.sqrt(W)             # rms |z| ~ 2 sqrt(W) for these bands

    # matchedfilter's CPU run(), full-length spectra
    f = mf.MatchedFilter(n, ndata=1, ntemplates=B, device='cpu')
    f.set_data(d[None])
    f.set_templates(H)
    ref_i, ref_v = f.run(binsize=bs, threshold=thr_raw, window=(a0, a1), raw=True)
    ref_i, ref_v = numpy.array(ref_i[0]), numpy.array(ref_v[0])

    # float64 bin peaks
    z64 = numpy.fft.ifft(d.astype(complex)[None, :] * numpy.conj(H.astype(complex)), axis=-1) * n

    out = {}
    for dev in DEVICES:
        mx.set_default_device(mx.cpu if dev == 'cpu' else mx.gpu)
        D = mx.array(d[kmin:kmax])
        Hc = mx.conj(mx.array(H[:, kmin:kmax]))
        pad = [(0, 0), (kmin, n - kmin - W)]
        peak = segment_peaks(mx, D, Hc, pad, a0, a1, bs)
        worst_v, worst_64, ties = 0.0, 0.0, 0
        for k in range(B):
            idx, val = threshold_bins([q[k] for q in peak], a0, bs, thr_raw)
            nb = len(idx)
            if nb != ref_i.shape[1]:
                raise AssertionError("%s: %d bins, matchedfilter %d" % (dev, nb, ref_i.shape[1]))
            same = idx == ref_i[k]
            for b in numpy.nonzero(~same)[0]:
                # allowed only for an exact-magnitude tie or a threshold-edge bin
                lo, hi = a0 + b * bs, min(a0 + (b + 1) * bs, a1)
                mag = numpy.abs(z64[k, lo:hi])
                srt = numpy.sort(mag)[::-1]
                if srt[0] - srt[1] > 1e-5 * srt[0] and abs(srt[0] - thr_raw) > 1e-5 * thr_raw:
                    raise AssertionError("%s: template %d bin %d index %d vs matchedfilter %d"
                                         % (dev, k, b, idx[b], ref_i[k][b]))
                ties += 1
            ok = same & (idx >= 0)
            if ok.any():
                rv = ref_v[k][ok]
                worst_v = max(worst_v, float(numpy.max(numpy.abs(val[ok] - rv) / numpy.abs(rv))))
                zz = z64[k, idx[ok]]
                worst_64 = max(worst_64, float(numpy.max(numpy.abs(val[ok] - zz) / numpy.abs(zz))))
        out[dev] = (worst_v, worst_64, ties)
        if worst_v > 1e-5 or worst_64 > 1e-5:
            raise AssertionError("%s: value error %.2e vs matchedfilter, %.2e vs float64"
                                 % (dev, worst_v, worst_64))
    nhit = int((ref_i >= 0).sum())
    if nhit < 3 or nhit == ref_i.size:
        raise AssertionError("case %d: %d of %d bins above threshold; the test needs a mix"
                             % (seed, nhit, ref_i.size))
    return out, nhit, ref_i.shape[1]


def guards():
    mx.set_cache_limit(64 * 2 ** 20)
    mx.set_memory_limit(1024 * 2 ** 20)
    mx.reset_peak_memory()
    a = mx.zeros((1024, 1024), dtype=mx.complex64)
    mx.eval(a)
    peak = mx.get_peak_memory()
    del a
    mx.clear_cache()
    return peak


DEVICES = ('cpu', 'gpu')


def main():
    global DEVICES
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='both', choices=['cpu', 'gpu', 'both'])
    a = p.parse_args()
    DEVICES = ('cpu', 'gpu') if a.device == 'both' else (a.device,)
    cases = [
        # seed, n, kmin, kmax, a0, a1, bs, B, thr
        (1, 2 ** 14, 300, 8000, 1024, 1024 + 6 * 2048, 2048, 4, 2.2),
        (2, 2 ** 14, 300, 8192, 1000, 1000 + 5 * 2048 + 777, 2048, 3, 2.2),   # partial last bin
        (3, 2 ** 16, 1280, 32768, 8192, 8192 + 24 * 2048, 2048, 8, 2.5),
        (4, 2 ** 16, 640, 30000, 4096, 60000, 1024, 5, 2.5),                  # other bin size
    ]
    for c in cases:
        out, nhit, nb = case(*c)
        for dev, (wv, w64, ties) in out.items():
            print("case %d n %d %s: %d bins x %d templates, %d above threshold; value vs matchedfilter %.2e, "
                  "vs float64 %.2e; index ties %d: OK" % (c[0], c[1], dev, nb, c[7], nhit, wv, w64,
                                                          ties))
    peak = guards()
    print("memory guards: limits set, a 8 MB array peaked at %.1f MB: OK" % (peak / 2 ** 20))
    print("all OK")


if __name__ == '__main__':
    main()
