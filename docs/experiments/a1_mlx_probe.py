#!/usr/bin/env python
"""Route A design probe: MLX peak memory and accuracy of the planned per-batch step.

Synthetic data at the C3 shape (4 segments x 2^20, analyze window 897024 samples
= 438 bins of 2048). For each device (cpu, gpu) and template batch B, runs the
backend's per-segment step: product with the conjugate templates, unnormalised
inverse FFT, |z|^2 over the analyze window, per-bin max and argmax, the complex
value at the argmax, threshold, copy to host. Reports MLX's peak memory for the
step (mx.get_peak_memory after reset, with the buffer cache cleared first) and
the largest relative error of the bin-peak magnitudes against float64 NumPy.
Timing is not reported here (timings are run outside Claude).

Usage: a1_mlx_probe.py OUT.json   (needs Metal: run unsandboxed)
"""
import json
import sys

import numpy
import mlx.core as mx

N, BIN, A0, NB, NSEG = 2 ** 20, 2048, 65536, 438, 4


def step(D, Hc, s):
    z = mx.fft.ifft(D[s][None, :] * Hc, norm='forward')
    zs = z[:, A0:A0 + NB * BIN].reshape(Hc.shape[0], NB, BIN)
    re, im = mx.real(zs), mx.imag(zs)
    p = re * re + im * im
    m = mx.max(p, axis=-1)
    i = mx.argmax(p, axis=-1)
    ii = i[..., None].astype(mx.int32)
    vr = mx.take_along_axis(re, ii, axis=-1)[..., 0]
    vi = mx.take_along_axis(im, ii, axis=-1)[..., 0]
    hit = m > 0
    mx.eval(m, i, vr, vi, hit)
    return numpy.array(m), numpy.array(i), numpy.array(vr), numpy.array(vi)


def main():
    rng = numpy.random.default_rng(3)
    d = (rng.standard_normal((NSEG, N)) + 1j * rng.standard_normal((NSEG, N))).astype(numpy.complex64)
    out = []
    for dev in ('cpu', 'gpu'):
        mx.set_default_device(mx.cpu if dev == 'cpu' else mx.gpu)
        D = mx.array(d)
        mx.eval(D)
        for B in (8, 16, 32):
            h = numpy.zeros((B, N), numpy.complex64)
            h[:, :4096] = (rng.standard_normal((B, 4096)) + 1j * rng.standard_normal((B, 4096)))
            H = numpy.fft.fft(h, axis=-1).astype(numpy.complex64)
            Hc = mx.conj(mx.array(H))
            mx.eval(Hc)
            mx.clear_cache()
            mx.reset_peak_memory()
            base = mx.get_active_memory()
            m, i, vr, vi = step(D, Hc, 0)
            peak = mx.get_peak_memory()
            z = numpy.fft.ifft(d[0].astype(complex)[None, :] * numpy.conj(H[:2].astype(complex)), axis=-1) * N
            ref = numpy.abs(z[:, A0:A0 + NB * BIN].reshape(2, NB, BIN)).max(-1)
            got = numpy.abs(vr[:2] + 1j * vi[:2])
            err = float(numpy.max(numpy.abs(got / ref - 1)))
            r = dict(device=dev, B=B, inputs_mb=base / 2 ** 20,
                     step_peak_mb=peak / 2 ** 20, step_extra_mb=(peak - base) / 2 ** 20,
                     max_rel_err_peak_mag_vs_f64=err)
            print(r, flush=True)
            out.append(r)
            del Hc
    json.dump(out, open(sys.argv[1], 'w'), indent=1)


if __name__ == '__main__':
    main()
