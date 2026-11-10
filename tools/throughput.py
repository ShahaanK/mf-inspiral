#!/usr/bin/env python
"""Throughput per Mac: jobs per hour with several concurrent jobs (design in
~/.claude/plans/wondrous-foraging-owl.md, results to go in docs/).

Subcommands
  run        run one configuration: slots of back-to-back jobs over a measurement window
  verify     compare every output of a configuration bit for bit with a per-type reference
  summarise  jobs/h with fractional credit, wall-time spread, CPU and GPU use, memory, flags
  selftest   check the driver with dummy sleep jobs (no GPU, runs in the sandbox)

run
  throughput.py run --config S8 --out DIR --warmup 60 --window 480 --stagger 5 \
      --slot 'stock:8:pycbc_inspiral ... --output {out}.hdf'
  Each slot runs its command back to back, launching new jobs only before the window ends,
  then waits for in-flight jobs (never kills). {out} = DIR/TYPE-s<slot>-j<seq>; stdout and
  stderr go to {out}.log. Per job: start/end (monotonic and wall), exit code, and rusage from
  os.wait4 (user, sys, maxrss), appended to DIR/jobs.jsonl. A sampler thread records vm_stat
  and sysctl vm.swapusage every 10 s, and top (other users' CPU) and pmset -g therm every
  30 s, into DIR/samples.jsonl. Memory pre-flight: if free+inactive+purgeable+speculative is
  below the summed per-job footprint estimate, plus each shared
  page-cached template cache once (MLX 1.8 GB, mf_fast 1.1 GB), plus 3 GB, writes DIR/SKIPPED and exits 3.

jobs/h = 3600/W * sum_jobs overlap(job, window)/duration(job)
"""

import argparse
import getpass
import glob
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time

# Per-job private memory (peak footprint), rounded up. Max RSS is not used: it counts shared
# library pages and page-cached files once per process.
#   stock, mfinspiral, mlx: /usr/bin/time, session 20261109-223435-mlx (0.46, 0.72, 0.98 GB).
#   mffast: 0.51-0.52 GB measured (proc_pid_rusage lifetime max) with the memory-mapped cache,
#   which concurrent jobs share; it was 1.55 GB when each job read its own copy.
# Shared page-cached template files are added once per configuration that uses them.
FOOT_EST_GB = {'stock': 0.5, 'mfinspiral': 0.8, 'mffast': 0.6, 'mlx': 1.0, 'dummy': 0.05}
SHARED_GB = {'mlx': 1.8, 'mffast': 1.1}     # MLX band cache 1.77 GB; mf_fast npz 1.07 GB
MARGIN_GB = 3.0
PERF = {'search/run_time', 'search/setup_time_fraction',
        'search/templates_per_core', 'search/filter_rate_per_core'}


# ---------------------------------------------------------------- system probes
def sh(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else 'ERROR rc=%d: %s' % (r.returncode, r.stderr.strip())
    except Exception as e:                                 # noqa: BLE001 (record, never fail)
        return 'ERROR %r' % e


def vm_stat():
    txt = sh(['vm_stat'])
    page = int(re.search(r'page size of (\d+)', txt).group(1)) if 'page size' in txt else 16384
    v = {}
    for line in txt.splitlines()[1:]:
        m = re.match(r'\s*"?([^:"]+)"?:\s+(\d+)', line)
        if m:
            v[m.group(1).strip()] = int(m.group(2))
    return page, v


def available_gb():
    page, v = vm_stat()
    keys = ('Pages free', 'Pages inactive', 'Pages purgeable', 'Pages speculative')
    return sum(v.get(k, 0) for k in keys) * page / 2 ** 30


def top_others(me):
    txt = sh(['top', '-l', '2', '-s', '1', '-n', '10', '-o', 'cpu', '-stats', 'pid,user,cpu,mem,command'])
    if txt.startswith('ERROR'):
        return txt, []
    block = txt.split('PID')[-1].splitlines()[1:]            # the second sample
    procs = []
    for line in block:
        f = line.split(None, 4)
        if len(f) >= 5:
            try:
                procs.append(dict(pid=int(f[0]), user=f[1], cpu=float(f[2]), mem=f[3], command=f[4]))
            except ValueError:
                pass
    return None, [p for p in procs if p['user'] != me]


class Sampler(threading.Thread):
    def __init__(self, path, me):
        super().__init__(daemon=True)
        self.path, self.me, self.stop = path, me, threading.Event()

    def run(self):
        n = 0
        with open(self.path, 'a') as f:
            while True:
                rec = dict(t=time.time(), tm=time.monotonic())
                page, v = vm_stat()
                rec['vm'] = dict(page=page, free=v.get('Pages free'), inactive=v.get('Pages inactive'),
                                 pageouts=v.get('Pageouts'), swapouts=v.get('Swapouts'))
                rec['swapusage'] = sh(['sysctl', 'vm.swapusage']).strip()
                if n % 3 == 0:
                    err, others = top_others(self.me)
                    rec['top_error'] = err
                    rec['others'] = others
                    rec['therm'] = sh(['pmset', '-g', 'therm']).strip()
                f.write(json.dumps(rec) + '\n')
                f.flush()
                n += 1
                if self.stop.wait(10):
                    return


# ---------------------------------------------------------------- run
def slot_worker(cfg, typ, slot, cmd_t, out, t_stop, lock, jobs_path, env):
    seq = 0
    while time.monotonic() < t_stop:
        base = os.path.join(out, '%s-s%d-j%d' % (typ, slot, seq))
        cmd = shlex.split(cmd_t.replace('{out}', base))
        rec = dict(config=cfg, type=typ, slot=slot, seq=seq, out=base, cmd=cmd,
                   start=time.monotonic(), start_wall=time.time())
        with open(base + '.log', 'w') as log:
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            _, status, ru = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
        rec.update(end=time.monotonic(), end_wall=time.time(), rc=p.returncode,
                   utime=ru.ru_utime, stime=ru.ru_stime, maxrss_mb=ru.ru_maxrss / 2 ** 20)
        with lock, open(jobs_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        seq += 1


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    slots = []
    for s in a.slot:
        typ, count, cmd_t = s.split(':', 2)
        slots += [(typ, cmd_t)] * int(count)
    need = (sum(FOOT_EST_GB[t] for t, _ in slots) + MARGIN_GB
            + sum(SHARED_GB.get(t, 0) for t in set(t for t, _ in slots)))
    avail = available_gb()
    meta = dict(config=a.config, slots=[t for t, _ in slots], specs=a.slot, warmup=a.warmup,
                window=a.window, stagger=a.stagger, need_gb=need, avail_gb=avail,
                env={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                    'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS',
                                                    'MF_AUTOTUNE')})
    if avail < need and not a.no_preflight:
        msg = 'SKIPPED %s: %.1f GB available (free+inactive+purgeable+speculative) < %.1f GB needed' \
              % (a.config, avail, need)
        open(os.path.join(a.out, 'SKIPPED'), 'w').write(msg + '\n')
        json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
        print(msg)
        return 3
    t0 = time.monotonic()
    meta.update(t0=t0, t0_wall=time.time(), win_start=t0 + a.warmup, win_end=t0 + a.warmup + a.window)
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
    sampler = Sampler(os.path.join(a.out, 'samples.jsonl'), getpass.getuser())
    sampler.start()
    lock = threading.Lock()
    jobs_path = os.path.join(a.out, 'jobs.jsonl')
    threads = []
    print('%s: %d slots %s, warm-up %d s, window %d s, %.1f GB available, %.1f GB needed'
          % (a.config, len(slots), ' '.join(a.slot_summary or []), a.warmup, a.window, avail, need),
          flush=True)
    for i, (typ, cmd_t) in enumerate(slots):
        th = threading.Thread(target=slot_worker, args=(a.config, typ, i, cmd_t, a.out, meta['win_end'],
                                                        lock, jobs_path, os.environ.copy()))
        th.start()
        threads.append(th)
        if i + 1 < len(slots):
            time.sleep(a.stagger)
    for th in threads:
        th.join()
    sampler.stop.set()
    sampler.join()
    meta['t_done'] = time.monotonic()
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
    jobs = load_jobs(a.out)
    bad = [j for j in jobs if j['rc'] != 0]
    print('%s: %d jobs, %d failed, drain ended %.0f s after the window'
          % (a.config, len(jobs), len(bad), meta['t_done'] - meta['win_end']), flush=True)
    return 1 if bad else 0


# ---------------------------------------------------------------- verify
def h5_items(path):
    import h5py
    import numpy
    out = {}
    with h5py.File(path, 'r') as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out['d:' + name] = obj[()]
            for k, v in obj.attrs.items():
                out['a:%s:%s' % (name, k)] = v
        for k, v in f.attrs.items():
            out['a::' + k] = v
        f.visititems(visit)
    keep = {}
    for k, v in out.items():
        name = k.split(':', 2)[1] if k.startswith('d:') else ''
        if any(name.endswith(p) for p in PERF):
            continue
        keep[k] = numpy.asarray(v)
    return keep


def same(a, b):
    import numpy
    if a.keys() != b.keys():
        return sorted(set(a) ^ set(b))
    diff = []
    for k in a:
        x, y = a[k], b[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            diff.append(k)
        elif x.dtype.kind in 'fc':
            if not numpy.array_equal(x, y, equal_nan=True):
                diff.append(k)
        elif not numpy.array_equal(x, y):
            diff.append(k)
    return diff


def cmd_verify(a):
    refs = dict(r.split('=', 1) for r in a.ref)
    cache, nok, nbad = {}, 0, 0
    for path in sorted(glob.glob(os.path.join(a.dir, '*-s*-j*.hdf'))):
        typ = os.path.basename(path).split('-s')[0]
        if typ not in refs:
            print('no reference for %s' % path)
            nbad += 1
            continue
        if typ not in cache:
            cache[typ] = h5_items(refs[typ])
        d = same(cache[typ], h5_items(path))
        if d:
            nbad += 1
            print('DIFFERS %s: %s' % (path, ', '.join(d[:8])))
        else:
            nok += 1
    jobs = load_jobs(a.dir) if os.path.exists(os.path.join(a.dir, 'jobs.jsonl')) else []
    missing = [j['out'] for j in jobs if j['rc'] == 0 and not os.path.exists(j['out'] + '.hdf')]
    for m in missing:
        print('MISSING %s.hdf' % m)
    print('verify %s: %d identical, %d differ or unchecked, %d missing'
          % (os.path.basename(a.dir.rstrip('/')), nok, nbad, len(missing)))
    return 1 if nbad or missing else 0


# ---------------------------------------------------------------- summarise
def load_jobs(d):
    return [json.loads(x) for x in open(os.path.join(d, 'jobs.jsonl'))]


def credit(start, end, ws, we):
    dur = end - start
    if dur <= 0:
        return 0.0
    return max(0.0, min(end, we) - max(start, ws)) / dur


def gpu_seconds(job):
    try:
        tm = json.load(open(job['out'] + '.json'))['timing']
        return tm['filter'] + tm['psd_var'] + tm['upload']
    except (OSError, KeyError, ValueError):
        return None


def quarter_ratio(v):
    v = sorted(v)
    q = max(1, len(v) // 4)
    return statistics.mean(v[-q:]) / statistics.mean(v[:q])


def summarise_config(d):
    meta = json.load(open(os.path.join(d, 'meta.json')))
    if os.path.exists(os.path.join(d, 'SKIPPED')):
        return dict(config=meta['config'], skipped=open(os.path.join(d, 'SKIPPED')).read().strip())
    jobs = load_jobs(d)
    ws, we = meta['win_start'], meta['win_end']
    W = we - ws
    r = dict(config=meta['config'], window_s=W, slots=len(meta['slots']), types={}, flags=[])
    tot = 0.0
    for typ in sorted(set(j['type'] for j in jobs)):
        js = [j for j in jobs if j['type'] == typ]
        ok = [j for j in js if j['rc'] == 0]
        cr = [credit(j['start'], j['end'], ws, we) for j in ok]
        jph = 3600.0 / W * sum(cr)
        tot += jph
        wall = [j['end'] - j['start'] for j in ok]
        cpu = [j['utime'] + j['stime'] for j in ok]
        t = dict(slots=meta['slots'].count(typ), jobs=len(js), failed=len(js) - len(ok),
                 jobs_per_hour=jph, credited_jobs=sum(cr),
                 wall_median=statistics.median(wall), wall_min=min(wall), wall_max=max(wall),
                 wall_q4_over_q1=quarter_ratio(wall), cpu_s_per_job=statistics.mean(cpu),
                 cores_used=sum(c * x for c, x in zip(cr, cpu)) / W,
                 maxrss_mb_max=max(j['maxrss_mb'] for j in ok))
        g = [gpu_seconds(j) for j in ok]
        if all(x is not None for x in g) and typ == 'mlx':
            t['gpu_busy_s_per_job'] = statistics.mean(g)
            t['gpu_busy_share_upper'] = sum(c * x for c, x in zip(cr, g)) / W
        if t['failed']:
            r['flags'].append('%d failed %s jobs' % (t['failed'], typ))
        r['types'][typ] = t
    r['jobs_per_hour'] = tot
    r['summed_maxrss_gb'] = sum(max([j['maxrss_mb'] for j in jobs if j['slot'] == s] or [0])
                                for s in range(len(meta['slots']))) / 1024
    r['avail_gb_before'] = meta['avail_gb']
    samples = [json.loads(x) for x in open(os.path.join(d, 'samples.jsonl'))]
    po = [s['vm']['pageouts'] for s in samples if s['vm'].get('pageouts') is not None]
    so = [s['vm']['swapouts'] for s in samples if s['vm'].get('swapouts') is not None]
    r['pageouts_delta'] = po[-1] - po[0] if po else None
    r['swapouts_delta'] = so[-1] - so[0] if so else None
    if r['swapouts_delta']:
        r['flags'].append('swap-outs during run: %d' % r['swapouts_delta'])
    if r['pageouts_delta']:
        r['flags'].append('page-outs during run: %d' % r['pageouts_delta'])
    loud = {}
    for s in samples:
        for p in s.get('others') or []:
            if p['cpu'] > 20:
                k = (p['user'], p['command'].strip())
                loud[k] = max(loud.get(k, 0), p['cpu'])
    if loud:
        r['flags'].append('other users above 20%% CPU: %s' % '; '.join(
            '%s %s %.0f%%' % (u, c, v) for (u, c), v in sorted(loud.items())))
    therm = sorted(set(s['therm'] for s in samples if 'therm' in s))
    r['therm'] = therm
    if any('CPU_Speed_Limit' in t and 'CPU_Speed_Limit \t= 100' not in t for t in therm):
        r['flags'].append('pmset reported a CPU speed limit')
    return r


def cmd_summarise(a):
    res = []
    dirs = [d for d in glob.glob(os.path.join(a.session, 'tp-*'))
            if os.path.exists(os.path.join(d, 'meta.json'))]
    for d in sorted(dirs, key=lambda d: os.path.getctime(os.path.join(d, 'meta.json'))
                    if 't0_wall' not in json.load(open(os.path.join(d, 'meta.json')))
                    else json.load(open(os.path.join(d, 'meta.json')))['t0_wall']):
        res.append(summarise_config(d))
    base = next((r['jobs_per_hour'] for r in res if r['config'] == 'S8' and 'skipped' not in r), None)
    lines = ['| config | slots | jobs/h | vs S8 | per type: jobs/h, median wall (min–max), Q4/Q1, '
             'CPU-s/job, cores used | summed max RSS | flags |', '|' + '---|' * 7]
    for r in res:
        if 'skipped' in r:
            lines.append('| %s | | skipped | | %s | | |' % (r['config'], r['skipped']))
            continue
        r['vs_S8'] = r['jobs_per_hour'] / base if base else None
        per = []
        for typ, t in r['types'].items():
            s = '%s ×%d: %.0f, %.1f s (%.1f–%.1f), %.2f, %.1f, %.2f' % (
                typ, t['slots'], t['jobs_per_hour'], t['wall_median'], t['wall_min'], t['wall_max'],
                t['wall_q4_over_q1'], t['cpu_s_per_job'], t['cores_used'])
            if 'gpu_busy_share_upper' in t:
                s += ', GPU busy ≤ %.0f%%' % (100 * t['gpu_busy_share_upper'])
            if t['failed']:
                s += ', %d FAILED' % t['failed']
            per.append(s)
        lines.append('| %s | %d | %.0f | %s | %s | %.1f GB | %s |' % (
            r['config'], r['slots'], r['jobs_per_hour'],
            '%.2f×' % r['vs_S8'] if r['vs_S8'] else '—', '<br>'.join(per),
            r['summed_maxrss_gb'], '; '.join(r['flags']) or 'none'))
    drift = [r for r in res if r['config'] == 'S8-drift' and 'skipped' not in r]
    if base and drift:
        dr = (drift[0]['jobs_per_hour'] - base) / base
        lines.append('\nThermal drift: S8-drift %.0f jobs/h vs S8 %.0f jobs/h: %+.1f%%'
                     % (drift[0]['jobs_per_hour'], base, 100 * dr))
    md = '\n'.join(lines)
    print(md)
    open(os.path.join(a.session, 'summary.md'), 'w').write(md + '\n')
    json.dump(res, open(os.path.join(a.session, 'summary.json'), 'w'), indent=1)
    return 0


# ---------------------------------------------------------------- selftest
def cmd_selftest(a):
    # fractional credit on synthetic intervals
    assert credit(0, 10, 0, 10) == 1.0
    assert credit(-5, 5, 0, 10) == 0.5
    assert credit(5, 15, 0, 10) == 0.5
    assert credit(-5, 15, 0, 10) == 0.5
    assert credit(11, 12, 0, 10) == 0.0
    # back-to-back 4 s jobs from -2 to 14, window [0, 10]: 0.5 + 1 + 1 + 0.5 = 2.5 = 10 s / 4 s
    w = sum(credit(s, s + 4, 0, 10) for s in (-2, 2, 6, 10))
    assert abs(w - 2.5) < 1e-12, w
    print('credit function: OK')
    sess = a.out
    d = os.path.join(sess, 'tp-dummy')
    os.makedirs(sess, exist_ok=True)
    job = '%s -c "import time; time.sleep(2); open(\'{out}.done\', \'w\')"' % sys.executable
    ns = argparse.Namespace(config='S8', out=d, warmup=4, window=20, stagger=0.5,
                            slot=['dummy:2:' + job], slot_summary=None, no_preflight=False)
    rc = cmd_run(ns)
    assert rc == 0, rc
    r = summarise_config(d)
    jph = r['jobs_per_hour']
    expect = 2 * 3600 / 2.0
    # each job also pays interpreter start-up; measure it from the jobs themselves
    wall = r['types']['dummy']['wall_median']
    expect_wall = 2 * 3600 / wall
    print('dummy: %.0f jobs/h, %.0f expected from the median job wall time %.3f s '
          '(%.0f for a bare 2 s job); %d jobs' % (jph, expect_wall, wall, expect,
                                                  r['types']['dummy']['jobs']))
    assert abs(jph / expect_wall - 1) < 0.05, 'jobs/h off by more than 5%'
    assert r['types']['dummy']['failed'] == 0
    samples = [json.loads(x) for x in open(os.path.join(d, 'samples.jsonl'))]
    print('sampler: %d samples; swapusage %r; top %r; therm %r' % (
        len(samples), samples[0]['swapusage'][:60], samples[0].get('top_error') or 'ok',
        samples[0].get('therm', '')[:60]))
    print('selftest OK')
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--config', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--warmup', type=float, default=60)
    r.add_argument('--window', type=float, default=480)
    r.add_argument('--stagger', type=float, default=5)
    r.add_argument('--slot', action='append', required=True, help="TYPE:COUNT:COMMAND with {out}")
    r.add_argument('--no-preflight', action='store_true')
    v = sub.add_parser('verify')
    v.add_argument('dir')
    v.add_argument('ref', nargs='+', help='TYPE=REFERENCE.hdf')
    s = sub.add_parser('summarise')
    s.add_argument('session')
    t = sub.add_parser('selftest')
    t.add_argument('--out', required=True)
    a = p.parse_args()
    if a.cmd == 'run':
        a.slot_summary = ['%s×%s' % tuple(x.split(':', 2)[:2]) for x in a.slot]
    return dict(run=cmd_run, verify=cmd_verify, summarise=cmd_summarise, selftest=cmd_selftest)[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
