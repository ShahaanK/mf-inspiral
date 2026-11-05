"""Template cache: tapered, truncated templates as n-point spectra.

Each bank template is made the way pycbc's FilterBank makes it (same function,
grid, end frequency and distance), but generated from ``flow - taper_low`` Hz
with ``f_ref = flow``. Then it gets PyCBC's standard half-Kaiser roll-on over
[flow - taper_low, flow) (out of band: the data are zero there) and a roll-off
over the last ``taper_high`` Hz before its end (pycbc.waveform.utils.fd_taper,
beta 8, the convention of get_td_waveform_from_fd). docs/design-fast.md §1.

The time-domain response is kept on lags -(chirp + pad) .. +pad, placed in an
n-point buffer and transformed, so each template is one complex64 row that
matchedfilter's overlap-save blocks filter exactly like a full-length filter
with the same truncated template (docs/short-block-check.md §1).
"""

import math

import numpy

import pycbc.waveform
from pycbc import DYN_RANGE_FAC
from pycbc.types import FrequencySeries, complex64, zeros
from pycbc.waveform.utils import fd_taper

# Cut-off error budget: half of the 1e-5 tolerance, leaving the rest for float32.
CUTOFF_BUDGET = 5e-6


def tapered_template(bank, index, flow, taper_low, taper_high):
    """Full-length tapered spectrum of bank template ``index``.

    Returns (complex128 numpy array of length flen, chirp length in seconds
    from the lowered start frequency).
    """
    df = bank.delta_f
    f_end = bank.end_frequency(index)
    if f_end is None or f_end >= bank.filter_length * df:
        f_end = (bank.filter_length - 1) * df
    f_start = flow - taper_low
    out = zeros(bank.filter_length, dtype=complex64)
    h = pycbc.waveform.get_waveform_filter(
        out, bank.table[index], approximant=bank.approximant(index),
        f_lower=f_start, f_final=f_end, delta_f=df, delta_t=bank.delta_t,
        distance=1.0 / DYN_RANGE_FAC, f_ref=flow)
    chirp = h.chirp_length
    # fd_taper's window is float64, so it needs a double-precision series.
    H = FrequencySeries(h.numpy().astype(numpy.complex128), delta_f=df)
    if taper_low:
        H = fd_taper(H, f_start, flow, beta=8, side='left')
    if taper_high:
        nz = numpy.nonzero(numpy.abs(H.numpy()) > 0)[0]
        f_hi = (nz[-1] + 1) * df
        H = fd_taper(H, f_hi - taper_high, f_hi, beta=8, side='right')
    return H.numpy(), chirp


def sigmasq(H, psd, kmin, kmax, delta_f):
    """4 df sum |H|^2 / S over [kmin, kmax), in float64."""
    p = psd.numpy()[kmin:kmax].astype(numpy.float64)
    return 4.0 * delta_f * float(numpy.sum(numpy.abs(H[kmin:kmax]) ** 2 / p))


class TemplateCache(object):
    """Every bank template as an n-point spectrum, with its support and norms.

    Parameters
    ----------
    bank : pycbc.waveform.FilterBank
        The bank, built on the job's frequency grid (flen, delta_f).
    frontend : FrontEnd
        Supplies the grid, kmin/kmax and the segment PSDs for sigma^2.
    n : int
        Block length (power of two).
    pad, pad_fallback : float
        Padding in seconds either side of the chirp, and the larger padding
        used for a template whose cut-off error exceeds CUTOFF_BUDGET.
    measured : dict, optional
        Bank row -> measured cut-off error at ``pad`` (tools/cutoff_errors.py
        on this job's data). When given, it decides which templates get the
        fallback. Otherwise the data-independent estimate does, scaled by
        CUTOFF_RATIO.
    """

    def __init__(self, bank, frontend, n=65536, pad=2.0, pad_fallback=4.0,
                 taper_low=4.0, taper_high=4.0, indices=None, keep_full=False,
                 measured=None, log=None):
        fe = frontend
        self.n = int(n)
        self.pad_base = pad
        self.switch_rule = 'measured' if measured is not None else 'estimate'
        self.flow = fe.flow
        self.delta_f = fe.delta_f
        self.taper_low, self.taper_high = taper_low, taper_high
        N, fs = fe.tlen, fe.sample_rate
        idx = range(len(bank)) if indices is None else indices
        self.rows = numpy.array(list(idx), dtype=numpy.int64)
        T = len(self.rows)
        self.spectra = numpy.zeros((T, self.n), dtype=numpy.complex64)
        self.L = numpy.zeros(T, dtype=numpy.int64)
        self.R = numpy.zeros(T, dtype=numpy.int64)
        self.pad = numpy.zeros(T)
        self.chirp = numpy.zeros(T)
        self.cutoff_estimate = numpy.zeros(T)
        self.sigmasq = numpy.zeros((T, len(fe.segments)))
        self.hashes = numpy.asarray(bank.table.template_hash, dtype=numpy.int64)[self.rows]
        self.duration = numpy.zeros(T, dtype=numpy.float32)
        self.full = [] if keep_full else None
        lag = numpy.arange(N)
        lag[lag >= N // 2] -= N
        for j, ti in enumerate(self.rows):
            H, chirp = tapered_template(bank, ti, fe.flow, taper_low, taper_high)
            Hfull = numpy.zeros(N, numpy.complex128)
            Hfull[:fe.flen] = H
            h = numpy.fft.ifft(Hfull)
            sig = [sigmasq(Hfull, p, fe.kmin, fe.kmax, fe.delta_f) for p in fe.psds]
            self.sigmasq[j] = sig
            # template_duration as stock writes it: the chirp length from flow.
            row = bank.table[ti]
            self.duration[j] = pycbc.waveform.get_waveform_filter_length_in_time(
                approximant=bank.approximant(ti), mass1=row['mass1'], mass2=row['mass2'],
                spin1z=row['spin1z'], spin2z=row['spin2z'], f_lower=fe.flow)
            # Cut-off error estimate: the PSD-weighted norm of the discarded
            # part, relative to the template's, sqrt(sigma_tail^2 / sigma^2), on
            # the worst segment PSD. Larger padding is used if it is over budget.
            for trial in (pad, pad_fallback):
                L = int(math.ceil((chirp + trial) * fs))
                R = int(math.ceil(trial * fs))
                keep = (lag >= -L) & (lag <= R)
                tail = Hfull - numpy.fft.fft(numpy.where(keep, h, 0))
                est = max(math.sqrt(sigmasq(tail, p, fe.kmin, fe.kmax, fe.delta_f) / s)
                          for p, s in zip(fe.psds, sig))
                used = trial
                if measured is not None:
                    over = measured[int(ti)] > CUTOFF_BUDGET
                else:
                    over = est * CUTOFF_RATIO > CUTOFF_BUDGET
                if trial == pad_fallback or not over:
                    break
            if L + R + 1 + 2048 > self.n:
                raise ValueError("template %d needs %.1f s of support; block length %d "
                                 "is too short" % (ti, (L + R + 1) / fs, self.n))
            hn = numpy.zeros(self.n, numpy.complex128)
            hn[:R + 1] = h[:R + 1]
            hn[self.n - L:] = h[N - L:]
            self.spectra[j] = numpy.fft.fft(hn).astype(numpy.complex64)
            self.L[j], self.R[j] = L, R
            self.pad[j], self.chirp[j], self.cutoff_estimate[j] = used, chirp, est
            if keep_full:
                self.full.append(Hfull.astype(numpy.complex64))
            if log and (j + 1) % 50 == 0:
                log("templates: %d/%d" % (j + 1, T))
        self.support = self.L + self.R + 1

    @property
    def fallback_rows(self):
        """Bank rows that got the larger padding."""
        return self.rows[self.pad > self.pad_base]

    def block_scale(self, j, seg):
        """Real factor that makes matchedfilter's block output complex SNR.

        The series is tlen * ifft(D), and the template spectrum is fft(h_n) with
        h_n = ifft(H) cut to its support. run_blocks returns the time-domain
        correlation sum s conj(h_n), not run()'s n-times-larger unnormalised
        inverse (docs/suspected-bugs.md #1), so it gives tlen * sum d conj(h).
        PyCBC's z is the same tlen * sum d conj(h), and SNR = z * 4 df / sqrt(sigma^2),
        so scale = 4 df / sqrt(sigma^2).
        """
        return 4.0 * self.delta_f / math.sqrt(self.sigmasq[j, seg])


# Measured cut-off error (max |dSNR|/SNR above 3.8) per unit of the estimate
# above: 0.88-1.62 (median 1.03) over all 431 templates x 4 segments of the C3
# job, at 2 s and 4 s padding (tools/cutoff_errors.py, docs/fast-cpu-results.md).
# The largest ratio is used, so the estimate rule errs towards more padding.
CUTOFF_RATIO = 1.62
