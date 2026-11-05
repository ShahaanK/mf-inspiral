#!/usr/bin/env python
"""Cut-off error of every tapered template, measured in float64 on the job's data.

For each template, segment and padding (2 s and 4 s): the largest |dSNR|/SNR over
samples with SNR > 3.8 in the segment's analyze region, between the full-length
filter with the tapered template and the same filter with the template cut to
-(chirp + pad) .. +pad. This is exactly the error short blocks add
(docs/short-block-check.md §1), measured without float32 rounding.

Also records the cache's data-independent estimate, sqrt(sigma_tail^2/sigma^2),
so the automatic padding switch in mf_fast/templates.py can be calibrated.

Usage: cutoff_errors.py OUT.json   (C3 job, all templates)
"""
import json
import math
import sys
import time

import numpy

sys.path.insert(0, '/Users/szkhan/projects/mf-inspiral')
from mf_fast.frontend import FrontEnd
from mf_fast.templates import tapered_template, sigmasq
from pycbc import waveform
from pycbc.types import complex64, zeros

S = '/Users/szkhan/projects/example-workflow/search'
PADS = (2.0, 4.0)


def main():
    t0 = time.time()
    fe = FrontEnd(S + '/H-H1_LOSC_CLN_4_V1-1186740069-3584.gwf', 'H1:LOSC-STRAIN',
                  S + '/bank.hdf', 1186740108, 1186741908, 1186740140, 1186741892)
    bank = waveform.FilterBank(S + '/bank.hdf', fe.flen, fe.delta_f,
                               low_frequency_cutoff=None, dtype=complex64,
                               out=zeros(fe.tlen, dtype=complex64))
    N, fs = fe.tlen, fe.sample_rate
    lag = numpy.arange(N)
    lag[lag >= N // 2] -= N
    D = [w.astype(numpy.complex128) for w in fe.whitened]
    print("front end %.0f s" % (time.time() - t0), flush=True)
    rows = []
    for ti in range(len(bank)):
        H, chirp = tapered_template(bank, ti, fe.flow, 4.0, 4.0)
        Hf = numpy.zeros(N, numpy.complex128)
        Hf[:fe.flen] = H
        h = numpy.fft.ifft(Hf)
        sig = [sigmasq(Hf, p, fe.kmin, fe.kmax, fe.delta_f) for p in fe.psds]
        zf = [numpy.fft.ifft(d * numpy.conj(Hf)) * N for d in D]
        rec = dict(row=ti, chirp=float(chirp))
        for pad in PADS:
            L = int(math.ceil((chirp + pad) * fs))
            R = int(math.ceil(pad * fs))
            Ht = numpy.fft.fft(numpy.where((lag >= -L) & (lag <= R), h, 0))
            est, worst = 0.0, 0.0
            for s, (d, z) in enumerate(zip(D, zf)):
                an = fe.segments[s].analyze
                norm = 4 * fe.delta_f / math.sqrt(sig[s])
                zt = numpy.fft.ifft(d * numpy.conj(Ht)) * N
                snr = (numpy.abs(z) * norm)[an]
                err = (numpy.abs(zt - z) * norm)[an]
                m = snr > 3.8
                if m.any():
                    worst = max(worst, float((err[m] / snr[m]).max()))
                est = max(est, math.sqrt(sigmasq(Hf - Ht, fe.psds[s], fe.kmin, fe.kmax,
                                                 fe.delta_f) / sig[s]))
            rec['pad%g' % pad] = dict(max_rel=worst, estimate=est,
                                      support_s=(L + R + 1) / fs)
        rows.append(rec)
        if (ti + 1) % 25 == 0:
            print("%d/%d  %.0f s" % (ti + 1, len(bank), time.time() - t0), flush=True)
    json.dump(rows, open(sys.argv[1], 'w'), indent=0)
    print("wrote %s  %.0f s" % (sys.argv[1], time.time() - t0))


if __name__ == '__main__':
    main()
