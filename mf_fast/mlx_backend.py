"""Full-length MLX backend for mf_fast (docs/design-mlx-backend.md).

Each (template, 512 s segment) pair is filtered the way stock pycbc_inspiral
filters it: the product of the overwhitened segment and the conjugate template
over the data band [kmin, kmax), an unnormalised inverse FFT at the full segment
length, and the loudest sample of each cluster-window bin of the analyze region.
The FFT and the per-bin reductions run in MLX (GPU, or MLX's CPU device for
tests); thresholding, PyCBC's symmetric rule and everything after it run on the
host, through PyCBC's own EventManager, exactly as mf_inspiral does. The output
file is therefore stock's SNR-only layout: psd_var_val from PyCBC's PSD
variation, and the chi-squared columns at EventManager's chi-squared-off values.

Templates (``kind``):
  stock    PyCBC's FilterBank templates, untapered (the default). sigma^2 is
           PyCBC's sigma_cached on the same complex64 values, so it is stock's.
  tapered  mf_fast's tapered templates (templates.tapered_template) with
           mf_fast's sigma^2 and duration, as reference A. Generated per job only.

Stock templates are generated per batch (as stock does), or read from a cache
file built by an earlier job (``--save-cache`` / ``--load-cache``): one HDF5 row
per template holding its nonzero band inside [kmin, kmax). Rows are read one
batch at a time, so the process never holds the whole bank.

MLX is imported only when this backend runs.
"""

import logging
import math
import os
import resource
import time
import types

import h5py
import numpy

import pycbc.psd
from pycbc import events, waveform
from pycbc.types import FrequencySeries, complex64, zeros
from pycbc.waveform.bank import sigma_cached

from mf_inspiral.control import symmetric_cluster
from mf_inspiral.main import OUT_TYPES

from .templates import sigmasq as tapered_sigmasq, tapered_template

CACHE_FORMAT = 'mf_fast-fulllength-1'


def _mx():
    import mlx.core as mx
    return mx


def filter_bank(fe):
    """The FilterBank mf_inspiral builds for this job's options."""
    opt = fe.opt
    return waveform.FilterBank(
        opt.bank_file, fe.flen, fe.delta_f,
        low_frequency_cutoff=None if opt.enable_bank_start_frequency else fe.flow,
        dtype=complex64, phase_order=opt.order, taper=opt.taper_template,
        approximant=opt.approximant, out=zeros(fe.tlen, dtype=complex64),
        max_template_length=opt.max_template_length,
        enable_compressed_waveforms=opt.use_compressed_waveforms,
        waveform_decompression_method=opt.waveform_decompression_method
        if opt.use_compressed_waveforms else None)


def band_of(h, kmin, kmax):
    """(offset from kmin, complex64 values) of h's nonzero part inside [kmin, kmax)."""
    seg = numpy.asarray(h[kmin:kmax])
    nz = numpy.nonzero(seg)[0]
    if not len(nz):
        return 0, seg[:0].astype(numpy.complex64)
    return int(nz[0]), seg[nz[0]:nz[-1] + 1].astype(numpy.complex64)


STOCK_ATTRS = ('f_lower', 'min_f_lower', 'end_idx', 'end_frequency', 'chirp_length',
               'length_in_time')


def stock_htilde(fe, band, off, attrs, approximant):
    """A FilterBank-like template (length flen) rebuilt from its cached band, with
    PyCBC's sigma_cached attached: the same values give the same sigma^2."""
    h = FrequencySeries(numpy.zeros(fe.flen, numpy.complex64), delta_f=fe.delta_f)
    k0 = fe.kmin + off
    h.data[k0:k0 + len(band)] = band
    for k in STOCK_ATTRS:
        setattr(h, k, attrs[k])
    h.approximant = approximant
    h.sigmasq = types.MethodType(sigma_cached, h)
    return h


class TemplateSource(object):
    """Batches of template bands with their sigma^2 per segment.

    ``batch(rows)`` returns (bands, sig, durations): bands is (B, W) complex64,
    W = kmax - kmin, each band placed at its offset from kmin; sig is (B, nseg)
    sigma^2 (PyCBC's for stock); durations are the template_duration column.
    It also sets each bank row's template_duration, which EventManager writes.
    """

    def __init__(self, fe, kind='stock', load=None, save=None, taper=(4.0, 4.0), log=None):
        if kind not in ('stock', 'tapered'):
            raise ValueError("kind must be 'stock' or 'tapered'")
        if kind == 'tapered' and (load or save):
            raise ValueError("tapered templates are generated per job; no cache file")
        if load and save:
            raise ValueError("load a cache or save one, not both")
        self.fe, self.kind, self.taper = fe, kind, taper
        self.log = log or (lambda *a: None)
        self.bank = filter_bank(fe)
        self.W = fe.kmax - fe.kmin
        self.timing = dict(generate=0.0, read=0.0, sigmasq=0.0, save=0.0)
        self.hashes = numpy.asarray(self.bank.table.template_hash, dtype=numpy.int64)
        self.load_path, self.save_path = load, save
        self.h5 = None
        if load:
            self._open_load(load)
        elif save:
            self._open_save(save)

    def __len__(self):
        return len(self.bank)

    # cache file -----------------------------------------------------------------
    def _meta(self):
        fe = self.fe
        return dict(format=CACHE_FORMAT, kind=self.kind, flen=fe.flen, tlen=fe.tlen,
                    delta_f=fe.delta_f, kmin=fe.kmin, kmax=fe.kmax,
                    bank_file=os.path.realpath(fe.opt.bank_file))

    def _open_save(self, path):
        T = len(self.bank)
        f = h5py.File(path, 'w')
        for k, v in self._meta().items():
            f.attrs[k] = v
        f.create_dataset('band', (T, self.W), dtype=numpy.complex64, chunks=(1, self.W))
        f.create_dataset('offset', (T,), dtype=numpy.int64)
        f.create_dataset('length', (T,), dtype=numpy.int64)
        f.create_dataset('template_hash', data=self.hashes)
        f.create_dataset('approximant', (T,), dtype=h5py.string_dtype())
        for k in STOCK_ATTRS:
            f.create_dataset(k, (T,), dtype=numpy.float64)
        self.h5 = f

    def _open_load(self, path):
        t = time.perf_counter()
        f = h5py.File(path, 'r')
        meta = self._meta()
        for k in ('format', 'kind', 'flen', 'tlen', 'kmin', 'kmax'):
            if f.attrs[k] != meta[k]:
                raise SystemExit("cache %s: %s is %r, this job needs %r"
                                 % (path, k, f.attrs[k], meta[k]))
        if not numpy.isclose(f.attrs['delta_f'], meta['delta_f'], rtol=0, atol=1e-15):
            raise SystemExit("cache %s was built for another delta_f" % path)
        if not numpy.array_equal(f['template_hash'][:], self.hashes):
            raise SystemExit("cache %s doesn't match this bank's templates" % path)
        self.h5 = f
        self.offset = f['offset'][:]
        self.length = f['length'][:]
        self.approx = [a.decode() if isinstance(a, bytes) else a for a in f['approximant'][:]]
        self.attrs = {k: f[k][:] for k in STOCK_ATTRS}
        self.timing['read'] += time.perf_counter() - t

    def close(self):
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

    # batches --------------------------------------------------------------------
    def batch(self, rows):
        fe = self.fe
        B = len(rows)
        bands = numpy.zeros((B, self.W), numpy.complex64)
        sig = numpy.zeros((B, len(fe.segments)))
        dur = numpy.zeros(B)
        if self.load_path:
            t = time.perf_counter()
            r0, r1 = rows[0], rows[-1] + 1
            assert list(rows) == list(range(r0, r1))
            block = self.h5['band'][r0:r1]
            self.timing['read'] += time.perf_counter() - t
            for k, ti in enumerate(rows):
                off, n = int(self.offset[ti]), int(self.length[ti])
                bands[k, off:off + n] = block[k, :n]
                t = time.perf_counter()
                attrs = {a: self._attr(a, ti) for a in STOCK_ATTRS}
                h = stock_htilde(fe, block[k, :n], off, attrs, self.approx[ti])
                sig[k] = [h.sigmasq(seg.psd) for seg in fe.segments]
                self.timing['sigmasq'] += time.perf_counter() - t
                dur[k] = attrs['chirp_length']
                self.bank.table[ti].template_duration = attrs['chirp_length']
            return bands, sig, dur
        for k, ti in enumerate(rows):
            t = time.perf_counter()
            if self.kind == 'stock':
                h = self.bank[ti]                 # stock generation, into shared memory
                off, b = band_of(h.numpy(), fe.kmin, fe.kmax)
                self.timing['generate'] += time.perf_counter() - t
                t = time.perf_counter()
                sig[k] = [h.sigmasq(seg.psd) for seg in fe.segments]
                self.timing['sigmasq'] += time.perf_counter() - t
                dur[k] = h.chirp_length
                bands[k, off:off + len(b)] = b
                if self.h5 is not None:
                    t = time.perf_counter()
                    self._write_row(ti, h, off, b)
                    self.timing['save'] += time.perf_counter() - t
            else:
                H, _ = tapered_template(self.bank, ti, fe.flow, *self.taper)
                self.timing['generate'] += time.perf_counter() - t
                t = time.perf_counter()
                sig[k] = [tapered_sigmasq(H, seg.psd, fe.kmin, fe.kmax, fe.delta_f)
                          for seg in fe.segments]
                self.timing['sigmasq'] += time.perf_counter() - t
                row = self.bank.table[ti]
                dur[k] = waveform.get_waveform_filter_length_in_time(
                    approximant=self.bank.approximant(ti), mass1=row['mass1'],
                    mass2=row['mass2'], spin1z=row['spin1z'], spin2z=row['spin2z'],
                    f_lower=fe.flow)
                self.bank.table[ti].template_duration = dur[k]
                bands[k] = H[fe.kmin:fe.kmax].astype(numpy.complex64)
        return bands, sig, dur

    def _attr(self, a, ti):
        v = self.attrs[a][ti]
        if a == 'end_idx':
            return int(v)
        return None if numpy.isnan(v) else float(v)

    def _write_row(self, ti, h, off, b):
        f = self.h5
        f['band'][ti, :len(b)] = b
        f['offset'][ti] = off
        f['length'][ti] = len(b)
        f['approximant'][ti] = h.approximant
        for a in STOCK_ATTRS:
            v = getattr(h, a, None)
            f[a][ti] = numpy.nan if v is None else v


def bin_peaks(mx, z, a0, a1, bs):
    """Per-bin max |z|^2, argmax (sample within the bin) and the complex value there,
    over [a0, a1) in bins of bs starting at a0 (a shorter last bin if needed)."""
    B = z.shape[0]
    nfull = (a1 - a0) // bs
    out = []
    spans = []
    if nfull:
        spans.append((a0, a0 + nfull * bs, nfull, bs))
    if a0 + nfull * bs < a1:
        spans.append((a0 + nfull * bs, a1, 1, a1 - a0 - nfull * bs))
    for s0, s1, nb, w in spans:
        zs = z[:, s0:s1].reshape(B, nb, w)
        re, im = mx.real(zs), mx.imag(zs)
        p = re * re + im * im
        i = mx.argmax(p, axis=-1)
        m = mx.max(p, axis=-1)
        ii = i[..., None].astype(mx.int32)
        out.append((m, i, mx.take_along_axis(re, ii, axis=-1)[..., 0],
                    mx.take_along_axis(im, ii, axis=-1)[..., 0]))
    return [mx.concatenate([o[q] for o in out], axis=1) if len(out) > 1 else out[0][q]
            for q in range(4)]


def segment_peaks(mx, d, Hc, pad, a0, a1, bs):
    """One segment against a batch: d (W,) overwhitened data band, Hc (B, W) conjugate
    template bands, both on the MLX device; pad places [kmin, kmax) in the n-point
    spectrum. Returns host arrays (max |z|^2, argmax in bin, Re z, Im z) per bin, with
    z the unnormalised inverse FFT (stock's convention)."""
    z = mx.fft.ifft(mx.pad(d[None, :] * Hc, pad), norm='forward')
    m, i, re, im = bin_peaks(mx, z, a0, a1, bs)
    mx.eval(m, i, re, im)
    return numpy.array(m), numpy.array(i), numpy.array(re), numpy.array(im)


def threshold_bins(peak, a0, bs, thr_raw):
    """(index, value) per bin as matchedfilter's run() returns them: the absolute lag
    and complex value of the bin's loudest sample where |z| > thr_raw, else -1 / 0."""
    m, i, re, im = peak
    hit = m > thr_raw ** 2
    index = numpy.where(hit, a0 + numpy.arange(len(m)) * bs + i, -1)
    value = numpy.where(hit, re + 1j * im, 0).astype(numpy.complex64)
    return index, value


class MLXSearch(object):
    """Filter every template against every segment in batches on MLX, and write
    stock's trigger file through PyCBC's EventManager."""

    def __init__(self, fe, source, device='gpu', tmpl_batch=8, cache_limit_mb=512,
                 memory_limit_mb=4096, log=None):
        self.mx = mx = _mx()
        self.fe, self.src = fe, source
        self.B = int(tmpl_batch)
        self.log = log or logging.info
        mx.set_default_device(mx.gpu if device == 'gpu' else mx.cpu)
        mx.set_cache_limit(int(cache_limit_mb) * 2 ** 20)
        mx.set_memory_limit(int(memory_limit_mb) * 2 ** 20)
        mx.reset_peak_memory()
        self.device = str(mx.default_device())
        self.timing = dict(psd_var=0.0, upload=0.0, filter=0.0, cluster=0.0, write=0.0)
        self.counts = dict(pairs=0, valid_samples=0, batches=0)

    def _data(self):
        """Per segment, the overwhitened band [kmin, kmax), as mf_inspiral makes it
        (seg / seg.psd in single precision)."""
        fe = self.fe
        d = numpy.stack([(seg / seg.psd).numpy()[fe.kmin:fe.kmax] for seg in fe.segments])
        return d.astype(numpy.complex64)

    def run(self, output):
        mx, fe, opt = self.mx, self.fe, self.fe.opt
        names = sorted(OUT_TYPES)
        types_ = [OUT_TYPES[n] for n in names]
        tstart = time.time()

        t = time.perf_counter()
        psd_var = None
        if opt.psdvar_segment is not None:
            psd_var = pycbc.psd.calc_filt_psd_variation(
                fe.gwstrain, opt.psdvar_segment, opt.psdvar_short_segment,
                opt.psdvar_long_segment, opt.psdvar_psd_duration, opt.psdvar_psd_stride,
                opt.psd_estimation, opt.psdvar_low_freq, opt.psdvar_high_freq)
        self.timing['psd_var'] = time.perf_counter() - t

        event_mgr = events.EventManager(opt, names, types_, psd=fe.segments[0].psd,
                                        gating_info=fe.gwstrain.gating_info)
        bs = int(opt.cluster_window * fe.sample_rate)
        thr = opt.snr_threshold
        N, kmin, W = fe.tlen, fe.kmin, self.src.W
        pad = [(0, 0), (kmin, N - kmin - W)]
        t = time.perf_counter()
        D = mx.array(self._data())
        mx.eval(D)
        self.timing['upload'] += time.perf_counter() - t
        T = len(self.src)
        consolidate_every = opt.finalize_events_template_rate
        an = [seg.analyze for seg in fe.segments]
        tsetup = time.time() - tstart

        for b0 in range(0, T, self.B):
            rows = list(range(b0, min(b0 + self.B, T)))
            bands, sig, _ = self.src.batch(rows)
            t = time.perf_counter()
            Hc = mx.conj(mx.array(bands))
            mx.eval(Hc)
            self.timing['upload'] += time.perf_counter() - t
            peaks = []
            t = time.perf_counter()
            for s, a in enumerate(an):
                peaks.append(segment_peaks(mx, D[s], Hc, pad, a.start, a.stop, bs))
            self.timing['filter'] += time.perf_counter() - t
            self.counts['pairs'] += len(rows) * len(an)
            self.counts['valid_samples'] += len(rows) * sum(a.stop - a.start for a in an)
            self.counts['batches'] += 1

            t = time.perf_counter()
            for k, ti in enumerate(rows):
                found = []
                for s, (seg, a) in enumerate(zip(fe.segments, an)):
                    norm = (4.0 * fe.delta_f) / math.sqrt(sig[k, s])
                    index, value = threshold_bins([q[k] for q in peaks[s]], a.start, bs,
                                                  thr / norm)
                    if not (index >= 0).any():
                        continue
                    keep = symmetric_cluster(index, value, len(index))
                    if not keep.any():
                        continue
                    idx = index[keep] - a.start
                    vals = dict.fromkeys(names)
                    vals['time_index'] = idx + seg.cumulative_index
                    vals['snr'] = value[keep] * norm
                    vals['sigmasq'] = numpy.zeros(len(idx), dtype=numpy.float32) + numpy.float32(sig[k, s])
                    if opt.psdvar_short_segment is not None:
                        vals['psd_var_val'] = pycbc.psd.find_trigger_value(
                            psd_var, vals['time_index'], opt.gps_start_time, opt.sample_rate)
                    found.append(vals)
                if found:
                    event_mgr.new_template(tmplt=self.src.bank.table[ti])
                    for vals in found:
                        event_mgr.add_template_events(names, [vals[n] for n in names])
                    event_mgr.cluster_template_events("time_index", "snr", bs)
                    event_mgr.finalize_template_events()
                if consolidate_every is not None and (ti + 1) % consolidate_every == 0:
                    event_mgr.consolidate_events(opt, gwstrain=fe.gwstrain)
            self.timing['cluster'] += time.perf_counter() - t

        t = time.perf_counter()
        event_mgr.consolidate_events(opt, gwstrain=fe.gwstrain)
        event_mgr.finalize_events()
        ntrig = len(event_mgr.events)
        event_mgr.save_performance(1, len(fe.segments), T, time.time() - tstart, tsetup)
        event_mgr.write_events(output)
        self.timing['write'] = time.perf_counter() - t
        self.memory = dict(
            mlx_peak_mb=mx.get_peak_memory() / 2 ** 20,
            mlx_cache_mb_end=mx.get_cache_memory() / 2 ** 20,
            mlx_active_mb_end=mx.get_active_memory() / 2 ** 20,
            maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)
        self.src.close()
        return ntrig
