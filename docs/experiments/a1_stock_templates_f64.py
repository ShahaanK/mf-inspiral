#!/usr/bin/env python
"""Route A, design question 1: can a plain full-length FFT with STOCK templates
reproduce stock pycbc_inspiral C3 within 1e-5?

Runs mf_inspiral (stock front end, stock untapered templates, stock sigma^2,
stock EventManager and output) with matchedfilter.MatchedFilter replaced by a
float64 NumPy correlator: for each (template, segment), ifft(data * conj(h)) * n
in complex128 over the full 2^20-point segment, then the per-bin maximum over the
analyze window, kept where its magnitude exceeds the threshold (matchedfilter's
run() rule), value cast to complex64. The inputs (overwhitened segments,
templates) are the float32 arrays stock uses; only the correlation is in double.

The MLX backend replaces exactly this correlator with float32 MLX on the GPU, so
this isolates the template question from MLX's precision.

Usage: a1_stock_templates_f64.py <mf_inspiral arguments>   (the C3 job)
"""
import os
import sys

import numpy

REPO = '/Users/szkhan/projects/mf-inspiral'
sys.path.insert(0, REPO)
import matchedfilter  # noqa: E402


class NumpyF64Filter(object):
    """The subset of matchedfilter.MatchedFilter that MFControl uses."""

    def __init__(self, n, ndata=1, ntemplates=1, device='cpu'):
        self.n = n

    def set_data(self, data):
        self.data = numpy.asarray(data).astype(numpy.complex128)

    def set_templates(self, templates):
        self.tmpl = numpy.conj(numpy.asarray(templates).astype(numpy.complex128))

    def run(self, binsize, threshold, window, data=(0, 1), raw=True):
        s = data[0]
        z = numpy.fft.ifft(self.data[s] * self.tmpl[0]) * self.n
        a, b = window
        nb = (b - a) // binsize
        w = z[a:a + nb * binsize].reshape(nb, binsize)
        j = numpy.argmax(numpy.abs(w), axis=1)
        v = w[numpy.arange(nb), j]
        hit = numpy.abs(v) > threshold
        idx = numpy.where(hit, a + numpy.arange(nb) * binsize + j, -1).astype(numpy.int64)
        val = numpy.where(hit, v, 0).astype(numpy.complex64)
        return idx[None, None, :], val[None, None, :]


matchedfilter.MatchedFilter = NumpyF64Filter

from mf_inspiral import control  # noqa: E402
control.mf.MatchedFilter = NumpyF64Filter
from mf_inspiral.main import main  # noqa: E402

if __name__ == '__main__':
    os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.environ.get('TMPDIR', '/tmp'), 'mpl'))
    main()
