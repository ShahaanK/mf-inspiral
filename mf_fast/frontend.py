"""Data front end: PyCBC conditioning, segmenting and PSDs, as in the C3 run.

This executable doesn't expose pycbc_inspiral's options. The conditioning is a fixed
preset, turned into the option namespace PyCBC's own from_cli functions read, so
the strain, segments and PSDs are produced by exactly the code stock uses.
"""

import numpy

from pycbc import DYN_RANGE_FAC, psd, strain
from pycbc.filter.matchedfilter import get_cutoff_indices

from mf_inspiral.cli import parse_args

# The [inspiral] options of notes/reference-run.md (run 2 / C3): the conditioning,
# plus the PSD-variation and event-manager options the MLX backend's stock-layout
# output uses (the matchedfilter backend reads only the conditioning).
PRESETS = {
    'c3': dict(
        pad_data=8, strain_high_pass=15, sample_rate=2048, segment_length=512,
        segment_start_pad=32, segment_end_pad=16, taper_data=1,
        psd_estimation='median', psd_segment_length=16, psd_segment_stride=8,
        psd_inverse_length=16, psd_num_segments=63, autogating_threshold=50,
        autogating_cluster=0.5, autogating_width=0.25, autogating_taper=0.25,
        autogating_pad=16, low_frequency_cutoff=20.0, allow_zero_padding=True,
        psdvar_segment=8, psdvar_short_segment=0.25, psdvar_long_segment=512,
        psdvar_psd_duration=8, psdvar_psd_stride=4, psdvar_low_freq=20,
        psdvar_high_freq=480, filter_inj_only=True, finalize_events_template_rate=500,
        injection_window=4.5, injection_filter_rejector_chirp_time_window=5),
}


class FrontEnd(object):
    """Conditioned data for one job.

    Attributes
    ----------
    segments : list of FrequencySeries
        PyCBC Fourier segments (not overwhitened), each with ``psd``,
        ``analyze``, ``seg_slice`` and ``cumulative_index``.
    gwstrain : pycbc TimeSeries
        The conditioned strain (PSD variation and the event manager use it).
    series : list of complex64 numpy arrays
        Per segment, the unnormalised inverse FFT of the overwhitened segment
        zeroed outside [kmin, kmax): the complex series every block is cut from.
    whitened : list of complex64 numpy arrays
        The overwhitened spectra themselves, length tlen, zeroed outside
        [kmin, kmax) (used by the full-length reference).
    """

    def __init__(self, frame_file, channel, bank_file, gps_start, gps_end,
                 trig_start=None, trig_end=None, preset='c3', block_series=True):
        p = PRESETS[preset]
        self.flow = float(p['low_frequency_cutoff'])
        self.sample_rate = int(p['sample_rate'])
        self.gps_start = int(gps_start)
        self.trig_start = int(trig_start if trig_start is not None
                              else gps_start + p['segment_start_pad'])
        self.trig_end = int(trig_end if trig_end is not None
                            else gps_end - p['segment_end_pad'])
        argv = ['--processing-scheme', 'cpu:1', '--bank-file', bank_file,
                '--frame-files', frame_file, '--channel-name', channel,
                '--gps-start-time', str(gps_start), '--gps-end-time', str(gps_end),
                '--trig-start-time', str(self.trig_start),
                '--trig-end-time', str(self.trig_end), '--output', '/dev/null',
                '--snr-threshold', '3.8', '--cluster-window', '1',
                '--cluster-function', 'symmetric', '--enable-bank-start-frequency']
        for k, v in p.items():
            flag = '--' + k.replace('_', '-')
            if v is True:
                argv.append(flag)
            else:
                argv += [flag, str(v)]
        self.opt = opt = parse_args(argv)

        gws = strain.from_cli(opt, dyn_range_fac=DYN_RANGE_FAC)
        self.gwstrain = gws
        self.gating_info = gws.gating_info
        ss = strain.StrainSegments.from_cli(opt, gws)
        self.tlen, self.flen, self.delta_f = ss.time_len, ss.freq_len, ss.delta_f
        self.segments = ss.fourier_segments()
        psd.associate_psds_to_segments(opt, self.segments, gws, self.flen,
                                       self.delta_f, self.flow,
                                       dyn_range_factor=DYN_RANGE_FAC,
                                       precision='single')
        self.kmin, self.kmax = get_cutoff_indices(self.flow, None, self.delta_f,
                                                  self.tlen)
        self.whitened, self.series = [], []
        if not block_series:            # the MLX backend whitens on its own
            return
        for seg in self.segments:
            d = numpy.zeros(self.tlen, dtype=numpy.complex128)
            d[self.kmin:self.kmax] = (seg.numpy()[self.kmin:self.kmax]
                                      / seg.psd.numpy()[self.kmin:self.kmax])
            self.whitened.append(d.astype(numpy.complex64))
            # numpy's ifft carries 1/tlen; undo it so the series is PyCBC's
            # unnormalised transform (see templates.block_scale).
            self.series.append((numpy.fft.ifft(d) * self.tlen).astype(numpy.complex64))

    @property
    def psds(self):
        return [s.psd for s in self.segments]
