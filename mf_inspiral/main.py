"""Find single-detector triggers by SNR, filtering with matchedfilter.

Data conditioning, PSD estimation, template generation, template norms, PSD
variation and trigger output are PyCBC's. Only the correlation, threshold
and per-segment clustering are done by matchedfilter (see control.py).
The output file has the layout pycbc_inspiral writes with chi-squared off.
"""

import logging
import time

import numpy

import matchedfilter
import pycbc
import pycbc.inject
import pycbc.psd
from pycbc import DYN_RANGE_FAC, events, fft, psd, scheme, strain, waveform
from pycbc.types import complex64, float32, zeros
from pycbc.vetoes.sgchisq import SingleDetSGChisq

from .cli import parse_args
from .control import MFControl

# Per-trigger columns and their types, as pycbc_inspiral's EventManager
# receives them. Columns that are never filled (the vetoes) are written by
# EventManager as zeros, giving the stock chi-squared-disabled values.
OUT_TYPES = {
    'time_index': int,
    'snr': complex64,
    'chisq': float32,
    'chisq_dof': int,
    'bank_chisq': float32,
    'bank_chisq_dof': int,
    'cont_chisq': float32,
    'psd_var_val': float32,
    'sigmasq': float32,
}
OUT_TYPES.update(SingleDetSGChisq.returns)


class Progress(object):
    """Write the filtering fraction to a file every ``step`` of progress."""

    def __init__(self, step, filename):
        self.step = step
        self.filename = filename
        self.last = -1.0

    def update(self, frac):
        if self.step and frac > self.last + self.step:
            with open(self.filename, 'w') as f:
                f.write("%.4f" % frac)
            self.last = frac


def main(argv=None):
    tstart = time.time()
    opt = parse_args(argv)
    pycbc.init_logging(opt.verbose)

    fft.from_cli(opt)
    inj_filter_rejector = pycbc.inject.InjFilterRejector.from_cli(opt)
    ctx = scheme.from_cli(opt)
    gwstrain = strain.from_cli(opt, dyn_range_fac=DYN_RANGE_FAC,
                               inj_filter_rejector=inj_filter_rejector)
    strain_segments = strain.StrainSegments.from_cli(opt, gwstrain)
    names = sorted(OUT_TYPES)
    types = [OUT_TYPES[n] for n in names]
    progress = Progress(opt.update_progress, opt.update_progress_file)

    with ctx:
        if opt.fft_backends == 'fftw':
            if opt.fftw_import_system_wisdom:
                fft.fftw.import_sys_wisdom()
            if opt.fftw_input_float_wisdom_file is not None:
                fft.fftw.import_single_wisdom_from_filename(
                    opt.fftw_input_float_wisdom_file)
            if opt.fftw_input_double_wisdom_file is not None:
                fft.fftw.import_double_wisdom_from_filename(
                    opt.fftw_input_double_wisdom_file)

        flow = opt.low_frequency_cutoff
        flen = strain_segments.freq_len
        tlen = strain_segments.time_len
        delta_f = strain_segments.delta_f

        logging.info("Making frequency-domain data segments")
        segments = strain_segments.fourier_segments()
        psd.associate_psds_to_segments(opt, segments, gwstrain, flen, delta_f,
                                       flow, dyn_range_factor=DYN_RANGE_FAC,
                                       precision='single')

        if len(strain_segments.segment_slices) == 0:
            logging.info("No segments to analyse; writing an empty file")
            event_mgr = events.EventManager(opt, names, types, psd=None,
                                            gating_info=gwstrain.gating_info)
            event_mgr.finalize_template_events()
            event_mgr.write_events(opt.output)
            return

        psd_var = None
        if opt.psdvar_segment is not None:
            logging.info("Calculating PSD variation")
            psd_var = pycbc.psd.calc_filt_psd_variation(
                gwstrain, opt.psdvar_segment, opt.psdvar_short_segment,
                opt.psdvar_long_segment, opt.psdvar_psd_duration,
                opt.psdvar_psd_stride, opt.psd_estimation,
                opt.psdvar_low_freq, opt.psdvar_high_freq)

        event_mgr = events.EventManager(opt, names, types,
                                        psd=segments[0].psd,
                                        gating_info=gwstrain.gating_info)

        template_mem = zeros(tlen, dtype=complex64)
        cluster_window = int(opt.cluster_window * gwstrain.sample_rate)
        ncores = getattr(ctx, 'num_threads', 1)

        logging.info("Overwhitening frequency-domain data segments")
        for seg in segments:
            seg /= seg.psd

        logging.info("Setting up matchedfilter (%s, %s)", opt.mf_device,
                     matchedfilter.backend())
        control = MFControl(segments, template_mem, flow, opt.snr_threshold,
                            cluster_window, tlen, delta_f,
                            device=opt.mf_device)

        logging.info("Read in template bank")
        bank = waveform.FilterBank(
            opt.bank_file, flen, delta_f,
            low_frequency_cutoff=None if opt.enable_bank_start_frequency
            else flow,
            dtype=complex64, phase_order=opt.order,
            taper=opt.taper_template, approximant=opt.approximant,
            out=template_mem, max_template_length=opt.max_template_length,
            enable_compressed_waveforms=opt.use_compressed_waveforms,
            waveform_decompression_method=opt.waveform_decompression_method
            if opt.use_compressed_waveforms else None)
        logging.info("Full template bank size: %s", len(bank))
        bank.template_thinning(inj_filter_rejector)
        logging.info("Template bank size after thinning: %s", len(bank))

        tsetup = time.time() - tstart
        tfilter = time.time()
        consolidate_every = opt.finalize_events_template_rate

        for t_num in range(len(bank)):
            template = None
            found = []
            for s_num, stilde in enumerate(segments):
                if not inj_filter_rejector.template_segment_checker(
                        bank, t_num, stilde):
                    continue
                if template is None:
                    template = bank[t_num]
                    control.set_template()
                progress.update((t_num + s_num / float(len(segments)))
                                / len(bank))
                logging.info("Filtering template %d/%d segment %d/%d",
                             t_num + 1, len(bank), s_num + 1, len(segments))

                sigmasq = template.sigmasq(stilde.psd)
                norm, idx, snrv = control.matched_filter_and_cluster(
                    s_num, sigmasq)
                if not len(idx):
                    continue

                vals = dict.fromkeys(names)
                vals['time_index'] = idx + stilde.cumulative_index
                vals['snr'] = snrv * norm
                vals['sigmasq'] = numpy.zeros(len(snrv), dtype=float32) \
                    + sigmasq
                if opt.psdvar_short_segment is not None:
                    vals['psd_var_val'] = pycbc.psd.find_trigger_value(
                        psd_var, vals['time_index'], opt.gps_start_time,
                        opt.sample_rate)
                found.append(vals)

            if found:
                event_mgr.new_template(tmplt=template.params)
                for vals in found:
                    event_mgr.add_template_events(names,
                                                  [vals[n] for n in names])
                event_mgr.cluster_template_events("time_index", "snr",
                                                  cluster_window)
                event_mgr.finalize_template_events()

            if consolidate_every is not None \
                    and (t_num + 1) % consolidate_every == 0:
                event_mgr.consolidate_events(opt, gwstrain=gwstrain)

        logging.info("Filtering took %.3f s", time.time() - tfilter)

    event_mgr.consolidate_events(opt, gwstrain=gwstrain)
    event_mgr.finalize_events()
    logging.info("Outputting %s triggers", len(event_mgr.events))
    event_mgr.save_performance(ncores, len(segments), len(bank),
                               time.time() - tstart, tsetup)
    logging.info("Writing out triggers")
    event_mgr.write_events(opt.output)

    if opt.fft_backends == 'fftw':
        if opt.fftw_output_float_wisdom_file:
            fft.fftw.export_single_wisdom_to_filename(
                opt.fftw_output_float_wisdom_file)
        if opt.fftw_output_double_wisdom_file:
            fft.fftw.export_double_wisdom_to_filename(
                opt.fftw_output_double_wisdom_file)
    logging.info("Finished")
