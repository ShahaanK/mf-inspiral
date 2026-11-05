"""mf_fast's block layout and stitching against a direct correlation.

Synthetic complex series and templates with finite support -L..+R. For every
analyze-region bin of 2048 samples, the stitched run_blocks result must be the
loudest sample of the direct correlation in that bin (and above threshold),
with the same lag and the same value to float32 rounding. Run with plain python.
"""

import sys

import numpy

sys.path.insert(0, __file__.rsplit('/tests/', 1)[0])
import matchedfilter as mf  # noqa: E402
from mf_fast.search import block_layout, segment_bins, run_blocks_ragged  # noqa: E402

BIN = 2048


def direct(series, h, L, R, n):
    """z[j] = sum_{m=-L..R} series[j+m] conj(h[m]), the run_blocks convention."""
    z = numpy.zeros(len(series), numpy.complex128)
    for m in range(-L, R + 1):
        hm = h[m % n]
        if hm == 0:
            continue
        lo, hi = max(0, -m), len(series) - max(0, m)
        z[lo:hi] += series[lo + m:hi + m] * numpy.conj(hm)
    return z


def check(seed, n, L, R, analyze, thr):
    rng = numpy.random.default_rng(seed)
    M = analyze.stop + R + 4096
    series = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(numpy.complex64)
    h = numpy.zeros(n, numpy.complex128)
    h[:R + 1] = rng.standard_normal(R + 1) + 1j * rng.standard_normal(R + 1)
    h[n - L:] = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    h /= numpy.sqrt(numpy.sum(numpy.abs(h) ** 2))     # |z| ~ unit variance
    spec = numpy.fft.fft(h).astype(numpy.complex64)
    starts, ws, we, valid = block_layout(analyze, L, R, n, BIN)
    # the windows tile the analyze region exactly, in whole bins from its start
    cov = numpy.concatenate([numpy.arange(s + a, s + b) for s, a, b in zip(starts, ws, we)])
    assert numpy.array_equal(cov, numpy.arange(analyze.start, analyze.stop)), "coverage"
    assert all((s + a - analyze.start) % BIN == 0 for s, a in zip(starts, ws)), "alignment"
    f = mf.MatchedFilter(n, ndata=len(starts), ntemplates=1)
    f.set_templates(spec[None, :])
    idx, val = run_blocks_ragged(f, series, starts, ws, we, BIN, thr)
    nbins = -(-(analyze.stop - analyze.start) // BIN)
    si, sv = segment_bins(idx[:, 0], val[:, 0], starts, ws, we, analyze.start, BIN, nbins)
    z = direct(series.astype(numpy.complex128), h, L, R, n)
    worst = 0.0
    for b in range(nbins):
        lo = analyze.start + b * BIN
        hi = min(lo + BIN, analyze.stop)
        k = lo + int(numpy.argmax(numpy.abs(z[lo:hi])))
        if abs(z[k]) > thr:
            assert si[b] == k, "bin %d: lag %d, expected %d" % (b, si[b], k)
            worst = max(worst, abs(sv[b] - z[k]) / abs(z[k]))
        else:
            assert si[b] == -1, "bin %d should be dismissed" % b
    return worst, len(starts), valid


def main():
    cases = [
        (1, 4096, 700, 300, slice(1000, 1000 + 9 * BIN)),        # whole bins
        (2, 4096, 700, 300, slice(1000, 1000 + 9 * BIN + 777)),  # partial last bin
        (3, 8192, 3000, 1000, slice(3001, 3001 + 13 * BIN + 5)),
        (4, 4096, 1, 1, slice(2, 2 + 6 * BIN)),                  # minimal support
        (5, 16384, 2000, 1000, slice(2500, 2500 + 10 * BIN + 99)),  # short last block (fewer bins)
    ]
    for seed, n, L, R, an in cases:
        w, nb, valid = check(seed, n, L, R, an, thr=2.5)
        print("seed %d n %d L %d R %d: %d blocks of %d valid, max rel diff %.2e: OK"
              % (seed, n, L, R, nb, valid, w))
        assert w < 1e-5
    print("all OK")


if __name__ == '__main__':
    main()
