#!/usr/bin/env python
"""GPU vs CPU gain of matchedfilter on synthetic data (docs/suspected-bugs.md #2).

For each n, one random complex block and one short template are filtered with
run_blocks on the CPU and on the GPU, and with run() on both. Prints, per n and
path, the median and spread of |z_gpu|/|z_cpu| - 1 over the bin peaks, and the
float64 numpy reference ratio for each device. A constant offset that grows with
log2(n) points at the FFT (twiddles); a constant independent of n points at a
normalisation step.

Usage: gpu_gain_check.py   (needs the GPU; run outside Claude)
"""
import numpy
import matchedfilter as mf


def case(n, rng):
    blk = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(numpy.complex64)
    h = numpy.zeros(n, numpy.complex64)
    h[:64] = (rng.standard_normal(64) + 1j * rng.standard_normal(64))
    spec = numpy.fft.fft(h).astype(numpy.complex64)
    w = (128, n - 128)
    bs = 2048 if n >= 8192 else n // 4
    w = (w[0], w[0] + ((w[1] - w[0]) // bs) * bs)
    # float64 series-convention reference: z[j] = sum_m blk[j+m] conj(h[m])
    ref = numpy.fft.ifft(numpy.fft.fft(blk.astype(complex)) * numpy.conj(spec.astype(complex)))
    out = {}
    for dev in ('cpu', 'gpu'):
        f = mf.MatchedFilter(n, ndata=1, ntemplates=1, device=dev)
        f.set_templates(spec[None])
        i, v = f.run_blocks(blk, numpy.array([0]), numpy.array([w[0]]), numpy.array([w[1]]),
                            binsize=bs, raw=True)
        i, v = numpy.asarray(i)[0, 0], numpy.asarray(v)[0, 0]
        out['blocks', dev] = (i, v)
        g = mf.MatchedFilter(n, ndata=1, ntemplates=1, device=dev)
        g.set_templates(spec[None])
        g.set_data(numpy.fft.fft(blk)[None].astype(numpy.complex64))
        i2, v2 = g.run(binsize=bs, window=w, raw=True)
        out['run', dev] = (numpy.asarray(i2)[0, 0], numpy.asarray(v2)[0, 0])
    for path, scale in (('blocks', 1.0), ('run', 1.0 / n)):
        ic, vc = out[path, 'cpu']
        ig, vg = out[path, 'gpu']
        same = ic == ig
        r = numpy.abs(vg[same]) / numpy.abs(vc[same]) - 1
        rc = numpy.abs(vc * scale) / numpy.abs(ref[ic]) - 1
        rg = numpy.abs(vg * scale) / numpy.abs(ref[ig]) - 1
        print("n %6d %-6s lags equal %d/%d  |gpu|/|cpu|-1 median %+.3e (min %+.3e max %+.3e)"
              "  cpu/ref-1 median %+.2e  gpu/ref-1 median %+.2e"
              % (n, path, same.sum(), len(same), numpy.median(r), r.min(), r.max(),
                 numpy.median(rc), numpy.median(rg)), flush=True)


def main():
    print("matchedfilter", mf.__version__, mf.devices())
    rng = numpy.random.default_rng(7)
    for n in (4096, 8192, 16384, 32768, 65536):
        case(n, rng)


if __name__ == '__main__':
    main()
