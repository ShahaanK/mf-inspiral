#!/usr/bin/env python
"""Project per-job cost (template cache + filter) against stock, by segments per job.

Linear model fitted from one 4-segment run of each:
  stock    = W + S * (F_stock + sigma_stock)
             W: waveform generation, once per job; F, sigma: per segment.
  mf_fast  = build:  B + S * (sigma_fast + F_fast)
             load:   Lc + S * (sigma_load + F_fast)
             B: generation + taper + time-domain conversion + duration + the
             per-template power, once per job; Lc: cache load; F_fast: the whole
             search per segment (scale, ingest, filter, stitch, cluster).
The front end (conditioning, PSDs) is excluded: it is PyCBC's in both and scales
with the data, not with this choice.

Stock numbers are passed on the command line (default: C3's cProfile figures,
c3-results.md: waveform 9.97 s, matched-filter step 18.62 s and sigma^2 0.49 s
over 4 segments).

Usage: project_cost.py BUILD_TIMING.json LOAD_TIMING.json [--segments 4 16 64]
         [--stock-waveform 9.97] [--stock-filter 18.62] [--stock-sigmasq 0.49]
         [--nseg-measured 4]
"""
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('build')
    p.add_argument('load')
    p.add_argument('--segments', type=int, nargs='+', default=[4, 16, 64])
    p.add_argument('--stock-waveform', type=float, default=9.97)
    p.add_argument('--stock-filter', type=float, default=18.62)
    p.add_argument('--stock-sigmasq', type=float, default=0.49)
    p.add_argument('--nseg-measured', type=int, default=4)
    a = p.parse_args()
    b = json.load(open(a.build))['timing']
    l = json.load(open(a.load))['timing']
    m = a.nseg_measured
    sig_b = b['cache_sigmasq'] / m
    # once per job: everything in the build except the job's own sigma^2 pass
    fixed_b = b['templates'] - b['cache_sigmasq']
    f_fast = b['search'] / m
    f_fast_load = l['search'] / m
    sig_l = l['cache_sigmasq'] / m
    W, Fs = a.stock_waveform, (a.stock_filter + a.stock_sigmasq) / m
    print("per-job model (seconds): stock = %.2f + S*%.3f | fast build = %.2f + S*%.3f | "
          "fast load = %.2f + S*%.3f" % (W, Fs, fixed_b, sig_b + f_fast, l['cache_load'],
                                         sig_l + f_fast_load))
    print("%8s %10s %12s %12s %16s %15s" % ('segments', 'stock (s)', 'build (s)', 'load (s)',
                                            'stock/build', 'stock/load'))
    for S in a.segments:
        st = W + S * Fs
        fb = fixed_b + S * (sig_b + f_fast)
        fl = l['cache_load'] + S * (sig_l + f_fast_load)
        print("%8d %10.1f %12.1f %12.1f %15.2fx %14.2fx" % (S, st, fb, fl, st / fb, st / fl))


if __name__ == '__main__':
    main()
