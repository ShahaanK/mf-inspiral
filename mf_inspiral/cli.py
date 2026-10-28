"""Command-line interface compatible with pycbc_inspiral.

Option groups come from PyCBC itself. The remaining top-level options are
declared here with pycbc_inspiral's names, types and defaults, so that a
pycbc_inspiral command line parses the same way. Options whose behaviour
this tool does not implement are accepted by the parser and then refused by
check_supported(), rather than silently ignored.
"""

import argparse

import numpy

import pycbc
import pycbc.inject
import pycbc.opt
from pycbc import fft, psd, scheme, strain
from pycbc.events import ranking
from pycbc.vetoes.sgchisq import SingleDetSGChisq
from pycbc.waveform.bank import add_approximant_arg


def build_parser():
    parser = argparse.ArgumentParser(
        description="Single-detector matched-filter triggers (SNR only), "
                    "using the matchedfilter package for the filter.")
    pycbc.add_common_pycbc_options(parser)

    io = parser.add_argument_group("Input and output")
    io.add_argument("--output", type=str, help="Output trigger file (.hdf)")
    io.add_argument("--bank-file", type=str, help="Template bank file")
    io.add_argument("--user-tag", type=str, metavar="TAG",
                    help="Accepted for compatibility; unused.")
    io.add_argument("--update-progress", type=float, default=0,
                    help="Write the fraction of filtering done to the "
                         "progress file whenever it has grown by this much.")
    io.add_argument("--update-progress-file", type=str,
                    default="progress.txt", help="Progress file name")

    filt = parser.add_argument_group("Filtering")
    filt.add_argument("--snr-threshold", type=float,
                      help="Keep clustered peaks with SNR above this value")
    filt.add_argument("--low-frequency-cutoff", type=float,
                      help="Lower frequency of the correlation (Hz)")
    filt.add_argument("--enable-bank-start-frequency", action="store_true",
                      help="Start each template at the bank's f_lower")
    filt.add_argument("--max-template-length", type=float,
                      help="Raise template start frequencies so that no "
                           "template is longer than this (s)")
    add_approximant_arg(parser)
    filt.add_argument("--order", type=int, default=-1,
                      choices=numpy.arange(-1, 9, 1),
                      help="Half-PN order of the approximant (-1: default)")
    filt.add_argument("--taper-template", choices=["start", "end", "startend"],
                      help="Taper time-domain templates before the FFT")
    filt.add_argument("--cluster-function", choices=["findchirp", "symmetric"],
                      default="findchirp",
                      help="Clustering method. Only 'symmetric' is "
                           "implemented.")
    filt.add_argument("--cluster-window", type=float, default=0,
                      help="Clustering window (s). Must be positive.")
    filt.add_argument("--finalize-events-template-rate", type=int,
                      default=None, metavar="NUM TEMPLATES",
                      help="Consolidate triggers every NUM TEMPLATES")
    filt.add_argument("--use-compressed-waveforms", action="store_true",
                      default=False,
                      help="Use compressed waveforms from the bank file")
    filt.add_argument("--waveform-decompression-method", default=None,
                      help="Method used to decompress bank waveforms")
    filt.add_argument("--mf-device", default="cpu",
                      help="matchedfilter device. Only 'cpu' is "
                           "supported so far.")

    # Declared so that pycbc_inspiral command lines and PyCBC's EventManager
    # see the attributes they expect. Non-default values are refused.
    unsup = parser.add_argument_group(
        "Not implemented (accepted only at their defaults)")
    unsup.add_argument("--newsnr-threshold", type=float, metavar="THRESHOLD")
    unsup.add_argument("--bank-veto-bank-file", type=str)
    unsup.add_argument("--chisq-snr-threshold", type=float)
    unsup.add_argument("--chisq-bins", default=0)
    unsup.add_argument("--chisq-threshold", type=float, default=0)
    unsup.add_argument("--chisq-delta", type=float, default=0)
    unsup.add_argument("--autochi-number-points", type=int, default=0)
    unsup.add_argument("--autochi-stride", type=int, default=0)
    unsup.add_argument("--autochi-two-phase", action="store_true",
                       default=False)
    unsup.add_argument("--autochi-onesided", default=None,
                       choices=["left", "right"])
    unsup.add_argument("--autochi-reverse-template", action="store_true",
                       default=False)
    unsup.add_argument("--autochi-max-valued", action="store_true",
                       default=False)
    unsup.add_argument("--autochi-max-valued-dof", type=int, metavar="INT")
    unsup.add_argument("--downsample-factor", type=int, default=1)
    unsup.add_argument("--upsample-threshold", type=float)
    unsup.add_argument("--upsample-method", choices=["pruned_fft"],
                       default="pruned_fft")
    unsup.add_argument("--keep-loudest-log-chirp-window", type=float)
    unsup.add_argument("--keep-loudest-interval", type=float)
    unsup.add_argument("--keep-loudest-num", type=int)
    unsup.add_argument("--keep-loudest-stat", default="newsnr",
                       choices=ranking.sngls_ranking_function_dict.keys())
    unsup.add_argument("--enable-q-transform", action="store_true")
    unsup.add_argument("--gpu-callback-method", default="none")
    unsup.add_argument("--checkpoint-interval", type=int)
    unsup.add_argument("--require-valid-checkpoint", action="store_true",
                       default=False)
    unsup.add_argument("--checkpoint-exit-maxtime", type=int)
    unsup.add_argument("--checkpoint-exit-code", type=int, default=77)
    unsup.add_argument("--multiprocessing-nprocesses", type=int)

    psd.insert_psd_option_group(parser)
    strain.insert_strain_option_group(parser)
    strain.StrainSegments.insert_segment_option_group(parser)
    scheme.insert_processing_option_group(parser)
    fft.insert_fft_option_group(parser)
    pycbc.opt.insert_optimization_option_group(parser)
    pycbc.inject.insert_injfilterrejector_option_group(parser)
    SingleDetSGChisq.insert_option_group(parser)
    return parser


def check_supported(opt, parser):
    """Refuse options whose behaviour is not implemented."""
    bad = []
    if opt.newsnr_threshold is not None:
        bad.append("--newsnr-threshold")
    if opt.bank_veto_bank_file:
        bad.append("--bank-veto-bank-file")
    if opt.chisq_snr_threshold is not None:
        bad.append("--chisq-snr-threshold")
    if str(opt.chisq_bins) not in ("0", ""):
        bad.append("--chisq-bins")
    if opt.chisq_threshold or opt.chisq_delta:
        bad.append("--chisq-threshold/--chisq-delta")
    if opt.autochi_number_points or opt.autochi_stride \
            or opt.autochi_two_phase or opt.autochi_onesided \
            or opt.autochi_reverse_template or opt.autochi_max_valued \
            or opt.autochi_max_valued_dof is not None:
        bad.append("--autochi-*")
    if opt.sgchisq_snr_threshold is not None or opt.sgchisq_locations:
        bad.append("--sgchisq-*")
    if opt.downsample_factor != 1 or opt.upsample_threshold is not None:
        bad.append("--downsample-factor/--upsample-threshold")
    if opt.keep_loudest_interval is not None \
            or opt.keep_loudest_num is not None \
            or opt.keep_loudest_log_chirp_window is not None:
        bad.append("--keep-loudest-*")
    if opt.enable_q_transform:
        bad.append("--enable-q-transform")
    if opt.gpu_callback_method != "none":
        bad.append("--gpu-callback-method")
    if opt.checkpoint_interval or opt.checkpoint_exit_maxtime:
        bad.append("--checkpoint-*")
    if opt.multiprocessing_nprocesses:
        bad.append("--multiprocessing-nprocesses")
    if opt.cluster_function != "symmetric":
        bad.append("--cluster-function (only 'symmetric')")
    if not opt.cluster_window or opt.cluster_window <= 0:
        bad.append("--cluster-window (must be positive)")
    if opt.mf_device != "cpu":
        bad.append("--mf-device (only 'cpu')")
    if bad:
        parser.error("not implemented in this tool: " + ", ".join(bad))


def parse_args(argv=None):
    parser = build_parser()
    opt = parser.parse_args(argv)
    psd.verify_psd_options(opt, parser)
    strain.verify_strain_options(opt, parser)
    strain.StrainSegments.verify_segment_options(opt, parser)
    scheme.verify_processing_options(opt, parser)
    fft.verify_fft_options(opt, parser)
    pycbc.opt.verify_optimization_options(opt, parser)
    check_supported(opt, parser)
    return opt
