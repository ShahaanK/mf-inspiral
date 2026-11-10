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

import json
import math
import os
import time

import numpy

import pycbc.waveform
from pycbc import DYN_RANGE_FAC
from pycbc.types import FrequencySeries, complex64, zeros
from pycbc.waveform.utils import fd_taper

# Cut-off error budget: half of the 1e-5 tolerance, leaving the rest for float32.
CUTOFF_BUDGET = 5e-6


def tapered_template(bank, index, flow, taper_low, taper_high, timing=None):
    """Full-length tapered spectrum of bank template ``index``.

    Returns (complex128 numpy array of length flen, chirp length in seconds
    from the lowered start frequency). ``timing``, if given, accumulates the
    seconds spent in 'generate' and 'taper'.
    """
    t0 = time.perf_counter()
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
    t1 = time.perf_counter()
    # fd_taper's window is float64, so it needs a double-precision series.
    H = FrequencySeries(h.numpy().astype(numpy.complex128), delta_f=df)
    if taper_low:
        H = fd_taper(H, f_start, flow, beta=8, side='left')
    if taper_high:
        nz = numpy.nonzero(numpy.abs(H.numpy()) > 0)[0]
        f_hi = (nz[-1] + 1) * df
        H = fd_taper(H, f_hi - taper_high, f_hi, beta=8, side='right')
    if timing is not None:
        timing['generate'] = timing.get('generate', 0.0) + t1 - t0
        timing['taper'] = timing.get('taper', 0.0) + time.perf_counter() - t1
    return H.numpy(), chirp


def template_power(H, kmin, kmax):
    """|H|^2 on the data band [kmin, kmax), rounded to float32.

    sigma^2 is always computed from this float32 power (sigmasq_from_power), so
    a cache loaded from disk gives exactly the sigma^2 of a fresh build.
    """
    return (numpy.abs(H[kmin:kmax]) ** 2).astype(numpy.float32)


def sigmasq_from_power(power, psd, kmin, delta_f):
    """4 df sum power / S, power starting at bin kmin, summed in float64."""
    p = psd.numpy()[kmin:kmin + len(power)].astype(numpy.float64)
    return 4.0 * delta_f * float(numpy.sum(power.astype(numpy.float64) / p))


def _npz_member_memmap(path, name):
    """Map one member of an uncompressed .npz read-only, or return None.

    numpy.load ignores mmap_mode for .npz files and reads every member into the
    process's own memory. A stored (uncompressed) member is a plain .npy block
    inside the zip, so it can be mapped in place: clean file pages are then shared
    through the page cache by every process that maps the same cache file.
    """
    import struct
    import zipfile
    with zipfile.ZipFile(path) as z:
        try:
            info = z.getinfo(name + '.npy')
        except KeyError:
            return None
        if info.compress_type != zipfile.ZIP_STORED:
            return None
        with open(path, 'rb') as f:
            f.seek(info.header_offset)
            head = f.read(30)
            if head[:4] != b'PK\x03\x04':
                return None
            nlen, xlen = struct.unpack('<HH', head[26:30])
            start = info.header_offset + 30 + nlen + xlen
            f.seek(start)
            fmt = numpy.lib.format
            version = fmt.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = fmt.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran, dtype = fmt.read_array_header_2_0(f)
            else:
                return None
            offset = f.tell()
        if dtype.hasobject:
            return None
        return numpy.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape,
                            order='F' if fortran else 'C')


def sigmasq(H, psd, kmin, kmax, delta_f):
    """sigma^2 of spectrum H over [kmin, kmax), by the float32-power definition."""
    return sigmasq_from_power(template_power(H, kmin, kmax), psd, kmin, delta_f)


def _trim(power):
    """The power up to its last nonzero bin (templates end before Nyquist)."""
    nz = numpy.nonzero(power)[0]
    return power[:nz[-1] + 1] if len(nz) else power[:0]


class TemplateCache(object):
    """Every bank template as an n-point spectrum, with its support and norms.

    Build with ``TemplateCache.build`` or load a saved one with
    ``TemplateCache.load``; then ``set_psds`` computes sigma^2 for the job.

    The padding decision (``pad`` or ``pad_fallback`` per template) is made at
    build time: from ``measured`` (bank row -> measured cut-off error at ``pad``,
    tools/cutoff_errors.py) when given, otherwise from the data-independent
    estimate sqrt(sigma_tail^2 / sigma^2) on the build job's PSDs, scaled by
    CUTOFF_RATIO. A saved cache keeps its build's decisions.
    """

    META = ('n', 'tlen', 'delta_f', 'kmin', 'kmax', 'flow', 'sample_rate', 'taper_low',
            'taper_high', 'pad_base', 'pad_fallback', 'switch_rule', 'bank_file')
    ARRAYS = ('rows', 'hashes', 'duration', 'chirp', 'L', 'R', 'pad', 'cutoff_estimate',
              'spectra')
    MAPPED = ('spectra',)            # plus 'power', handled separately in load

    @classmethod
    def build(cls, bank, frontend, bank_file, n=65536, pad=2.0, pad_fallback=4.0,
              taper_low=4.0, taper_high=4.0, indices=None, measured=None, fit_only=False,
              log=None):
        fe = frontend
        c = cls()
        c.n, c.tlen, c.delta_f = int(n), fe.tlen, fe.delta_f
        c.kmin, c.kmax, c.flow, c.sample_rate = fe.kmin, fe.kmax, fe.flow, fe.sample_rate
        c.taper_low, c.taper_high = taper_low, taper_high
        c.pad_base, c.pad_fallback = pad, pad_fallback
        c.switch_rule = 'measured' if measured is not None else 'estimate'
        c.bank_file = bank_file
        N, fs = fe.tlen, fe.sample_rate
        c.rows = numpy.array(list(range(len(bank)) if indices is None else indices),
                             dtype=numpy.int64)
        T = len(c.rows)
        c.spectra = numpy.zeros((T, c.n), dtype=numpy.complex64)
        c.L = numpy.zeros(T, dtype=numpy.int64)
        c.R = numpy.zeros(T, dtype=numpy.int64)
        c.pad = numpy.zeros(T)
        c.chirp = numpy.zeros(T)
        c.cutoff_estimate = numpy.full(T, numpy.nan)
        c.hashes = numpy.asarray(bank.table.template_hash, dtype=numpy.int64)[c.rows]
        c.duration = numpy.zeros(T, dtype=numpy.float32)
        c.power = []
        tm = dict.fromkeys(('generate', 'taper', 'time_domain', 'sigmasq', 'duration',
                            'estimate'), 0.0)
        lag = numpy.arange(N)
        lag[lag >= N // 2] -= N
        skipped = []
        for j, ti in enumerate(c.rows):
            H, chirp = tapered_template(bank, ti, fe.flow, taper_low, taper_high, timing=tm)
            t = time.perf_counter()
            Hfull = numpy.zeros(N, numpy.complex128)
            Hfull[:fe.flen] = H
            h = numpy.fft.ifft(Hfull)
            tm['time_domain'] += time.perf_counter() - t
            t = time.perf_counter()
            power = _trim(template_power(Hfull, fe.kmin, fe.kmax))
            c.power.append(power)
            sig = [sigmasq_from_power(power, p, fe.kmin, fe.delta_f) for p in fe.psds]
            tm['sigmasq'] += time.perf_counter() - t
            t = time.perf_counter()
            # template_duration as stock writes it: the chirp length from flow.
            row = bank.table[ti]
            c.duration[j] = pycbc.waveform.get_waveform_filter_length_in_time(
                approximant=bank.approximant(ti), mass1=row['mass1'], mass2=row['mass2'],
                spin1z=row['spin1z'], spin2z=row['spin2z'], f_lower=fe.flow)
            tm['duration'] += time.perf_counter() - t
            for trial in (pad, pad_fallback):
                L = int(math.ceil((chirp + trial) * fs))
                R = int(math.ceil(trial * fs))
                used = trial
                if measured is not None:
                    over = measured[int(ti)] > CUTOFF_BUDGET
                else:
                    # the estimate costs one full-length FFT per padding tried
                    t = time.perf_counter()
                    keep = (lag >= -L) & (lag <= R)
                    tail = Hfull - numpy.fft.fft(numpy.where(keep, h, 0))
                    est = max(math.sqrt(sigmasq(tail, p, fe.kmin, fe.kmax, fe.delta_f) / s)
                              for p, s in zip(fe.psds, sig))
                    c.cutoff_estimate[j] = est
                    over = est * CUTOFF_RATIO > CUTOFF_BUDGET
                    tm['estimate'] += time.perf_counter() - t
                if trial == pad_fallback or not over:
                    break
            if L + R + 1 + 2048 > c.n:
                if fit_only:
                    skipped.append(j)
                    continue
                raise ValueError("template %d needs %.1f s of support; block length %d "
                                 "is too short (use fit_only to skip such templates)"
                                 % (ti, (L + R + 1) / fs, c.n))
            t = time.perf_counter()
            hn = numpy.zeros(c.n, numpy.complex128)
            hn[:R + 1] = h[:R + 1]
            hn[c.n - L:] = h[N - L:]
            c.spectra[j] = numpy.fft.fft(hn).astype(numpy.complex64)
            tm['time_domain'] += time.perf_counter() - t
            c.L[j], c.R[j] = L, R
            c.pad[j], c.chirp[j] = used, chirp
            if log and (j + 1) % 50 == 0:
                log("templates: %d/%d" % (j + 1, T))
        c.build_timing = tm
        c.skipped_rows = c.rows[skipped]
        if skipped:
            # drop templates that don't fit this block length (fit_only)
            keep = numpy.setdiff1d(numpy.arange(T), skipped)
            for k in cls.ARRAYS:
                setattr(c, k, getattr(c, k)[keep])
            c.power = [c.power[k] for k in keep]
        c._finish()
        c.set_psds(fe.psds)
        return c

    def _finish(self):
        self.support = self.L + self.R + 1

    def set_psds(self, psds):
        """sigma^2 of every template against every segment PSD of this job."""
        t = time.perf_counter()
        self.sigmasq = numpy.array([[sigmasq_from_power(pw, p, self.kmin, self.delta_f)
                                     for p in psds] for pw in self.power])
        self.sigmasq_time = time.perf_counter() - t

    def save(self, path):
        """Write the cache as an uncompressed .npz (fast to load)."""
        lens = numpy.array([len(p) for p in self.power], dtype=numpy.int64)
        arrays = {k: getattr(self, k) for k in self.ARRAYS}
        meta = {k: getattr(self, k) for k in self.META}
        numpy.savez(path, power=numpy.concatenate(self.power), power_len=lens,
                    meta=numpy.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path, frontend, bank_file):
        """Load a saved cache, check it matches this job's grid and bank, and
        compute sigma^2 for this job's PSDs."""
        t = time.perf_counter()
        z = numpy.load(path)
        c = cls()
        meta = json.loads(str(z['meta']))
        for k, v in meta.items():
            setattr(c, k, v)
        fe = frontend
        want = dict(tlen=fe.tlen, delta_f=fe.delta_f, kmin=fe.kmin, kmax=fe.kmax,
                    flow=fe.flow, sample_rate=fe.sample_rate)
        bad = {k: (meta[k], v) for k, v in want.items() if meta[k] != v}
        if os.path.realpath(meta['bank_file']) != os.path.realpath(bank_file):
            bad['bank_file'] = (meta['bank_file'], bank_file)
        if bad:
            raise ValueError("template cache %s doesn't match this job: %s" % (path, bad))
        # the two large members (spectra, power) are mapped read-only when stored
        # uncompressed, so concurrent jobs share one copy; the rest are read
        c.mapped = []
        for k in cls.ARRAYS:
            m = _npz_member_memmap(path, k) if k in cls.MAPPED else None
            if m is not None:
                c.mapped.append(k)
            setattr(c, k, z[k] if m is None else m)
        lens = z['power_len']
        flat = _npz_member_memmap(path, 'power')
        if flat is None:
            flat = z['power']
        else:
            c.mapped.append('power')
        ends = numpy.cumsum(lens)
        c.power = [flat[e - n:e] for e, n in zip(ends, lens)]
        c.build_timing = None
        c._finish()
        c.load_time = time.perf_counter() - t
        c.set_psds(fe.psds)
        return c

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
