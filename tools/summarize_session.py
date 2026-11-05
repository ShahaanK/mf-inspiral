#!/usr/bin/env python
"""Summarise a timing session from tools/bench_session.zsh.

Usage: summarize_session.py SESSION_DIR [--ref-a REFA.hdf]

Reads, for every run in SESSION_DIR:
  * <stamp>-<name>.log: the command's output plus /usr/bin/time -l (wall, user,
    sys, peak RSS); for stock / mf_inspiral also the template-loop time, from
    "Template bank size after thinning" to "Outputting" (as in c3-results.md);
  * <name>.json: mf_fast's --timing-json or mlx_baseline's result;
  * stock.prof, mfinspiral.prof: the matched-filter step, as in C3 (stock:
    full_matched_filter_and_cluster_symm; mf_inspiral: MFControl.set_template +
    matched_filter_and_cluster), plus waveform generation and sigma^2;
  * <stamp>-top-<name>.txt: the quiet check before each run (busiest processes).

Medians over repeats. Speeds are per template x valid output sample: every
full-bank configuration filters the same 431 x 3,588,096 samples; fit-only runs
use their own count. Check A runs tools/compare_triggers.py --core on the GPU
outputs against reference A. Writes SESSION_DIR/summary.json and prints tables.
"""
import argparse
import glob
import json
import os
import pstats
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
REF_A = os.path.expanduser('~/projects/example-workflow/reference/fast/refA/'
                           'H1-REFA-1186740108-1800.hdf')
VALID_FULL = 431 * 3588096
STAMP = re.compile(r'^\d{8}-\d{6}-(.+)\.log$')


def parse_log(path):
    out = {}
    text = open(path, errors='replace').read()
    m = re.search(r'^\s*([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys', text, re.M)
    if m:
        out['real'], out['user'], out['sys'] = map(float, m.groups())
    else:                                   # /usr/bin/time -p (older logs)
        for k in ('real', 'user', 'sys'):
            m = re.search(r'^%s ([\d.]+)$' % k, text, re.M)
            if m:
                out[k] = float(m.group(1))
    m = re.search(r'^\s*(\d+)\s+maximum resident set size', text, re.M)
    if m:
        out['peak_rss_mb'] = int(m.group(1)) / 2 ** 20
    m = re.search(r'^exit (\d+)', text, re.M)
    out['exit'] = int(m.group(1)) if m else None

    def ts(pattern):
        m = re.search(r'^(\S+) INFO : ' + pattern, text, re.M)
        return datetime.fromisoformat(m.group(1)) if m else None
    a, b = ts('Template bank size after thinning'), ts('Outputting')
    if a and b:
        out['template_loop'] = (b - a).total_seconds()
    return out


def profile_split(path, kind):
    """The C3 split of a cProfile run: matched-filter step, waveforms, sigma^2."""
    st = pstats.Stats(path).stats
    cum = defaultdict(float)
    for (fn, line, func), (cc, nc, tt, ct, callers) in st.items():
        cum[func] = max(cum[func], ct)
    if kind == 'stock':
        mf = cum['full_matched_filter_and_cluster_symm']
    else:
        mf = cum['matched_filter_and_cluster'] + cum['set_template']
    return dict(matched_filter_step=mf, waveforms=cum['__getitem__'],
                sigmasq=cum['sigma_cached'])


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def group(name):
    """sweep-n65536-tb4-db4-2 -> (sweep-n65536-tb4-db4, 2); stock-1 -> (stock, 1)."""
    m = re.match(r'^(.*)-(\d+)$', name)
    return (m.group(1), int(m.group(2))) if m else (name, None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session')
    p.add_argument('--ref-a', default=REF_A)
    a = p.parse_args()
    d = a.session
    runs = defaultdict(list)
    for log in sorted(glob.glob(os.path.join(d, '*.log'))):
        m = STAMP.match(os.path.basename(log))
        if not m:
            continue
        name = m.group(1)
        rec = parse_log(log)
        rec['name'] = name
        js = os.path.join(d, name + '.json')
        if os.path.exists(js):
            rec['json'] = json.load(open(js))
        top = sorted(glob.glob(os.path.join(d, '*-top-%s.txt' % name)))
        if top:
            rec['top'] = open(top[-1]).read().strip().splitlines()
        g, _ = group(name)
        runs[g].append(rec)

    summary = {}
    print("== Runs (median over repeats)")
    print("%-34s %4s %8s %8s %9s %10s %10s" % ('config', 'reps', 'wall s', 'loop s',
                                              'filter s', 'per valid', 'peak MB'))
    for g in sorted(runs):
        rs = runs[g]
        bad = [r['name'] for r in rs if r.get('exit') not in (0, None)]
        filt, valid = [], None
        for r in rs:
            j = r.get('json', {})
            t = j.get('timing', {})
            if 'search_filter' in t:
                filt.append(t['search_filter'])
                valid = j['counts']['valid_samples']
            elif 'gpu_s' in j:
                filt.append(j['gpu_s'])
                valid = j.get('valid_samples')
        row = dict(reps=len(rs), failed=bad,
                   wall=median([r.get('real') for r in rs]),
                   template_loop=median([r.get('template_loop') for r in rs]),
                   filter=median(filt) if filt else None,
                   peak_rss_mb=median([r.get('peak_rss_mb') for r in rs]),
                   valid_samples=valid)
        if row['filter'] and valid:
            row['ns_per_valid_sample'] = 1e9 * row['filter'] / valid
        summary[g] = row
        print("%-34s %4d %8s %8s %9s %10s %10s%s" % (
            g, len(rs), '%.2f' % row['wall'] if row['wall'] else '-',
            '%.2f' % row['template_loop'] if row['template_loop'] else '-',
            '%.3f' % row['filter'] if row['filter'] else '-',
            '%.3f ns' % row['ns_per_valid_sample'] if 'ns_per_valid_sample' in row else '-',
            '%.0f' % row['peak_rss_mb'] if row['peak_rss_mb'] else '-',
            '  FAILED: %s' % bad if bad else ''))

    print("\n== Matched-filter step from cProfile (as C3)")
    for kind, f in (('stock', 'stock.prof'), ('mfinspiral', 'mfinspiral.prof')):
        path = os.path.join(d, f)
        if os.path.exists(path):
            sp = profile_split(path, 'stock' if kind == 'stock' else 'mf')
            sp['ns_per_valid_sample'] = 1e9 * sp['matched_filter_step'] / VALID_FULL
            summary[kind + '-profile-split'] = sp
            print("  %-11s matched-filter step %.2f s (%.3f ns/valid sample), waveforms %.2f s, "
                  "sigma^2 %.2f s" % (kind, sp['matched_filter_step'], sp['ns_per_valid_sample'],
                                      sp['waveforms'], sp['sigmasq']))

    base = summary.get('stock-profile-split', {}).get('ns_per_valid_sample')
    if base:
        print("\n== Speed-up over stock's matched-filter step, per valid output sample")
        for g, row in sorted(summary.items()):
            v = row.get('ns_per_valid_sample') if isinstance(row, dict) else None
            if v and not g.endswith('profile-split'):
                print("  %-34s %7.2fx" % (g, base / v))
        mfi = summary.get('mfinspiral-profile-split', {}).get('ns_per_valid_sample')
        if mfi:
            print("  %-34s %7.2fx" % ('mfinspiral (cProfile)', base / mfi))

    print("\n== Check A on GPU outputs (against %s)" % a.ref_a)
    for h in sorted(glob.glob(os.path.join(d, 'fastgpu-load-*.hdf'))) + \
            sorted(glob.glob(os.path.join(d, 'sweep-n65536-*-1.hdf'))):
        r = subprocess.run([sys.executable, os.path.join(REPO, 'tools', 'compare_triggers.py'),
                            '--core', a.ref_a, h], capture_output=True, text=True)
        lines = [l for l in r.stdout.splitlines()
                 if l.startswith(('matched', '  snr', '  coa_phase', '  unmatched', 'OVERALL'))]
        summary.setdefault('checkA', {})[os.path.basename(h)] = lines
        print("  %s" % os.path.basename(h))
        for l in lines:
            print("     " + l.strip())

    with open(os.path.join(d, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=str)
    print("\nwrote", os.path.join(d, 'summary.json'))


if __name__ == '__main__':
    main()
