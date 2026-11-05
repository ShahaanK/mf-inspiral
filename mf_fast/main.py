"""mf_fast: SNR-only triggers from batched short-block matched filtering.

The data front end is PyCBC's with a fixed conditioning preset. Templates are
PyCBC's, tapered at both band edges and cut to their support (templates.py), and
filtered in blocks with matchedfilter (search.py). See docs/design-fast.md.
"""

import argparse
import json
import logging
import time

import matchedfilter
from pycbc import waveform
from pycbc.types import complex64, zeros

from .frontend import FrontEnd, PRESETS
from .output import write
from .search import Search
from .templates import TemplateCache


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame-file', required=True)
    p.add_argument('--channel', required=True)
    p.add_argument('--bank', required=True)
    p.add_argument('--gps-start', type=int, required=True)
    p.add_argument('--gps-end', type=int, required=True)
    p.add_argument('--trig-start', type=int)
    p.add_argument('--trig-end', type=int)
    p.add_argument('--output', required=True)
    p.add_argument('--conditioning', default='c3', choices=sorted(PRESETS))
    p.add_argument('--threshold', type=float, default=3.8)
    p.add_argument('--mode', default='peak', choices=['peak'],
                   help="Only 'peak' so far; 'hier' is Phase D.")
    p.add_argument('--device', default='cpu')
    p.add_argument('--block-length', type=int, default=65536)
    p.add_argument('--pad', type=float, default=2.0)
    p.add_argument('--pad-fallback', type=float, default=4.0)
    p.add_argument('--taper-low', type=float, default=4.0)
    p.add_argument('--taper-high', type=float, default=4.0)
    p.add_argument('--cutoff-measurements',
                   help="tools/cutoff_errors.py output for this job's data: templates "
                        "whose measured cut-off error at --pad exceeds the budget get "
                        "--pad-fallback. Without it, the calibrated estimate decides.")
    p.add_argument('--tmpl-batch', type=int, default=64)
    p.add_argument('--timing-json')
    p.add_argument('--verbose', action='store_true')
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format='%(asctime)s %(message)s')
    log = logging.info
    tim = {}
    t0 = time.time()
    log("matchedfilter %s, device %s, CPU target %s", matchedfilter.__version__,
        a.device, matchedfilter.backend())
    fe = FrontEnd(a.frame_file, a.channel, a.bank, a.gps_start, a.gps_end,
                  a.trig_start, a.trig_end, preset=a.conditioning)
    tim['front_end'] = time.time() - t0

    t = time.time()
    measured = None
    if a.cutoff_measurements:
        with open(a.cutoff_measurements) as f:
            measured = {r['row']: r['pad%g' % a.pad]['max_rel'] for r in json.load(f)}
    bank = waveform.FilterBank(a.bank, fe.flen, fe.delta_f, low_frequency_cutoff=None,
                               dtype=complex64, out=zeros(fe.tlen, dtype=complex64))
    cache = TemplateCache(bank, fe, n=a.block_length, pad=a.pad,
                          pad_fallback=a.pad_fallback, taper_low=a.taper_low,
                          taper_high=a.taper_high, measured=measured, log=log)
    tim['templates'] = time.time() - t
    fb = cache.fallback_rows
    log("templates: %d; %d on %.1f s fallback padding (%s rule): %s", len(cache.rows),
        len(fb), a.pad_fallback, cache.switch_rule, fb.tolist())

    t = time.time()
    search = Search(fe, cache, threshold=a.threshold, device=a.device,
                    tmpl_batch=a.tmpl_batch, log=log)
    events = search.run()
    tim['search'] = time.time() - t
    tim.update({'search_' + k: v for k, v in search.timing.items()})

    t = time.time()
    ntrig = write(a.output, a.channel[:2], events, cache, fe.gps_start, fe.sample_rate,
                  fe.trig_start, fe.trig_end)
    tim['write'] = time.time() - t
    tim['total'] = time.time() - t0
    log("wrote %d triggers to %s", ntrig, a.output)
    if a.timing_json:
        info = dict(timing=tim, counts=search.counts, triggers=ntrig,
                    fallback_rows=fb.tolist(), switch_rule=cache.switch_rule,
                    args=vars(a),
                    matchedfilter=matchedfilter.__version__,
                    backend=matchedfilter.backend())
        with open(a.timing_json, 'w') as f:
            json.dump(info, f, indent=1)
    return 0
