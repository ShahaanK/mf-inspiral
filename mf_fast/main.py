"""mf_fast: SNR-only triggers from batched matched filtering.

The data front end is PyCBC's with a fixed conditioning preset. Two backends:

  matchedfilter (default)  PyCBC's templates, tapered at both band edges and cut to
                           their support (templates.py), filtered in short blocks
                           with matchedfilter on --device cpu|gpu (search.py).
                           docs/design-fast.md.
  mlx                      full-length FFTs in MLX, stock templates by default,
                           stock's SNR-only output (mlx_backend.py).
                           docs/design-mlx-backend.md.
"""

import argparse
import json
import logging
import os
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
    p.add_argument('--backend', default='matchedfilter', choices=['matchedfilter', 'mlx'])
    p.add_argument('--device', default='cpu',
                   help="matchedfilter backend: cpu or gpu.")
    p.add_argument('--mlx-device', default='gpu', choices=['gpu', 'cpu'],
                   help="mlx backend: MLX device (cpu for tests).")
    p.add_argument('--templates', default='stock', choices=['stock', 'tapered'],
                   help="mlx backend: stock FilterBank templates (default) or mf_fast's "
                        "tapered ones (generated per job).")
    p.add_argument('--cache-limit-mb', type=int, default=512,
                   help="mlx backend: MLX buffer-cache limit.")
    p.add_argument('--memory-limit-mb', type=int, default=4096,
                   help="mlx backend: MLX memory limit.")
    p.add_argument('--block-length', type=int, default=65536)
    p.add_argument('--pad', type=float, default=2.0)
    p.add_argument('--pad-fallback', type=float, default=4.0)
    p.add_argument('--taper-low', type=float, default=4.0)
    p.add_argument('--taper-high', type=float, default=4.0)
    p.add_argument('--cutoff-measurements',
                   help="tools/cutoff_errors.py output for this job's data: templates "
                        "whose measured cut-off error at --pad exceeds the budget get "
                        "--pad-fallback. Without it, the calibrated estimate decides.")
    p.add_argument('--save-cache',
                   help="Write the built template cache here (.npz) for later jobs.")
    p.add_argument('--load-cache',
                   help="Load a saved template cache instead of building one. Its padding "
                        "decisions are the build job's; sigma^2 is computed for this job.")
    p.add_argument('--tmpl-batch', type=int, default=None,
                   help="Templates per batch (default 64 for matchedfilter, 16 for mlx).")
    p.add_argument('--data-batch', type=int, default=0,
                   help="Blocks per filter call (0: all of a segment's blocks).")
    p.add_argument('--fit-only', action='store_true',
                   help="Skip templates whose support doesn't fit --block-length "
                        "(for block-length sweeps; the output then covers only the rest).")
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
                  a.trig_start, a.trig_end, preset=a.conditioning,
                  block_series=a.backend != 'mlx')
    tim['front_end'] = time.time() - t0
    if a.backend == 'mlx':
        return run_mlx(a, fe, tim, t0, log)
    if a.tmpl_batch is None:
        a.tmpl_batch = 64

    t = time.time()
    if a.load_cache:
        cache = TemplateCache.load(a.load_cache, fe, a.bank)
        if cache.n != a.block_length:
            raise SystemExit("cache %s was built for n=%d, not --block-length %d"
                             % (a.load_cache, cache.n, a.block_length))
        tim['cache_load'] = cache.load_time
        log("loaded template cache %s in %.2f s (sigma^2 %.2f s)", a.load_cache,
            cache.load_time, cache.sigmasq_time)
    else:
        measured = None
        if a.cutoff_measurements:
            with open(a.cutoff_measurements) as f:
                measured = {r['row']: r['pad%g' % a.pad]['max_rel'] for r in json.load(f)}
        bank = waveform.FilterBank(a.bank, fe.flen, fe.delta_f, low_frequency_cutoff=None,
                                   dtype=complex64, out=zeros(fe.tlen, dtype=complex64))
        cache = TemplateCache.build(bank, fe, a.bank, n=a.block_length, pad=a.pad,
                                    pad_fallback=a.pad_fallback, taper_low=a.taper_low,
                                    taper_high=a.taper_high, measured=measured,
                                    fit_only=a.fit_only, log=log)
        if len(cache.skipped_rows):
            log("fit-only: skipped %d templates that don't fit n=%d", len(cache.skipped_rows),
                a.block_length)
        tim.update({'cache_build_' + k: v for k, v in cache.build_timing.items()})
        if a.save_cache:
            ts = time.time()
            cache.save(a.save_cache)
            tim['cache_save'] = time.time() - ts
            log("saved template cache to %s in %.2f s", a.save_cache, tim['cache_save'])
    tim['cache_sigmasq'] = cache.sigmasq_time
    tim['templates'] = time.time() - t - tim.get('cache_save', 0.0)
    fb = cache.fallback_rows
    log("templates: %d; %d on %.1f s fallback padding (%s rule): %s", len(cache.rows),
        len(fb), a.pad_fallback, cache.switch_rule, fb.tolist())

    t = time.time()
    search = Search(fe, cache, threshold=a.threshold, device=a.device,
                    tmpl_batch=a.tmpl_batch, data_batch=a.data_batch, log=log)
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
                    templates_used=len(cache.rows),
                    args=vars(a),
                    env={k: v for k, v in os.environ.items() if k.startswith(('MF_', 'OMP_'))},
                    matchedfilter=matchedfilter.__version__,
                    backend=matchedfilter.backend())
        with open(a.timing_json, 'w') as f:
            json.dump(info, f, indent=1)
    return 0


def run_mlx(a, fe, tim, t0, log):
    """The mlx backend: stock-layout triggers from full-length MLX FFTs."""
    from .mlx_backend import MLXSearch, TemplateSource
    if a.tmpl_batch is None:
        a.tmpl_batch = 16
    src = TemplateSource(fe, kind=a.templates, load=a.load_cache, save=a.save_cache,
                         taper=(a.taper_low, a.taper_high), log=log)
    search = MLXSearch(fe, src, device=a.mlx_device, tmpl_batch=a.tmpl_batch,
                       cache_limit_mb=a.cache_limit_mb, memory_limit_mb=a.memory_limit_mb,
                       log=log)
    log("mlx backend: %s templates, %s, batch %d, cache %s", a.templates, search.device,
        a.tmpl_batch, a.load_cache and 'loaded' or (a.save_cache and 'saving') or 'none')
    ntrig = search.run(a.output)
    tim.update({'templates_' + k: v for k, v in src.timing.items()})
    tim.update(search.timing)
    tim['total'] = time.time() - t0
    log("wrote %d triggers to %s; peak MLX memory %.0f MB, max RSS %.0f MB", ntrig,
        a.output, search.memory['mlx_peak_mb'], search.memory['maxrss_mb'])
    if a.timing_json:
        info = dict(timing=tim, counts=search.counts, triggers=ntrig, memory=search.memory,
                    backend='mlx', device=search.device, args=vars(a),
                    env={k: v for k, v in os.environ.items() if k.startswith(('MF_', 'OMP_'))})
        with open(a.timing_json, 'w') as f:
            json.dump(info, f, indent=1)
    return 0
