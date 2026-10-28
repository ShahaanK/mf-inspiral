#!/usr/bin/env python
"""Compare two single-detector trigger files against notes/design-c.md §7.

Usage: compare_triggers.py REFERENCE NEW [--gps-start-time T]

Triggers are matched on (template_hash, sample index). Deterministic PyCBC
columns must be bit-identical for matched triggers; snr and coa_phase are
compared against relative and absolute tolerances. Unmatched triggers are
listed with the SNR evidence needed to classify them.
"""

import argparse
import sys

import h5py
import numpy

PERF = {'search/run_time', 'search/setup_time_fraction',
        'search/templates_per_core', 'search/filter_rate_per_core'}
EXACT = ['end_time', 'sigmasq', 'psd_var_val', 'template_duration',
         'template_hash', 'chisq', 'chisq_dof', 'sg_chisq', 'bank_chisq',
         'bank_chisq_dof', 'cont_chisq', 'cont_chisq_dof']
SNR_RTOL = 1e-5
PHASE_ATOL = 1e-5
THRESHOLD = 3.8


def datasets(f):
    out = {}
    f.visititems(lambda n, o: out.__setitem__(n, o)
                 if isinstance(o, h5py.Dataset) else None)
    return out


def keys(g, sample_rate, gps_start):
    samples = numpy.round((g['end_time'][:] - gps_start) * sample_rate)
    return list(zip(g['template_hash'][:].tolist(),
                    samples.astype(numpy.int64).tolist()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference')
    p.add_argument('new')
    p.add_argument('--gps-start-time', type=float, default=1186740108)
    p.add_argument('--sample-rate', type=float, default=2048)
    a = p.parse_args()

    ok = True
    A, B = h5py.File(a.reference, 'r'), h5py.File(a.new, 'r')
    da, db = datasets(A), datasets(B)

    print("== 1. Structure")
    if set(da) != set(db):
        ok = False
        print("dataset names differ:", sorted(set(da) ^ set(db)))
    print("datasets: %d reference, %d new" % (len(da), len(db)))
    for n in sorted(set(da) & set(db)):
        x, y = da[n], db[n]
        if x.dtype != y.dtype or (x.shape != y.shape and n.count('/') > 1):
            ok = False
            print("  %s: dtype/shape %s%s vs %s%s" % (n, x.dtype, x.shape,
                                                    y.dtype, y.shape))
    for n in sorted(set(da) & set(db)):
        if n.count('/') > 1 and n.split('/', 1)[1] not in PERF:
            same = numpy.array_equal(da[n][:], db[n][:])
            ok &= same
            print("  %-28s %s" % (n, "identical" if same else "DIFFERENT"))
    attrs_same = dict(A.attrs) == dict(B.attrs) and all(
        dict(A[g].attrs) == dict(B[g].attrs) for g in A if g in B)
    ok &= attrs_same
    print("  attributes", "identical" if attrs_same else "DIFFERENT")

    ifo = list(A.keys())[0]
    ga, gb = A[ifo], B[ifo]
    ka = keys(ga, a.sample_rate, a.gps_start_time)
    kb = keys(gb, a.sample_rate, a.gps_start_time)

    print("\n== 2. Matching on (template_hash, sample)")
    print("triggers: %d reference, %d new" % (len(ka), len(kb)))
    for name, k in (('reference', ka), ('new', kb)):
        if len(set(k)) != len(k):
            ok = False
            print("  duplicate keys in", name)
    ia = {k: i for i, k in enumerate(ka)}
    ib = {k: i for i, k in enumerate(kb)}
    common = sorted(set(ia) & set(ib))
    only_a = sorted(set(ia) - set(ib))
    only_b = sorted(set(ib) - set(ia))
    print("matched %d, only in reference %d, only in new %d"
          % (len(common), len(only_a), len(only_b)))
    xa = numpy.array([ia[k] for k in common], dtype=numpy.int64)
    xb = numpy.array([ib[k] for k in common], dtype=numpy.int64)

    print("\n== 3. Exact columns on matched triggers")
    for n in EXACT:
        if n not in ga and n not in gb:
            print("  %-18s absent from both files" % n)
            continue
        if n not in ga or n not in gb:
            ok = False
            print("  %-18s present in only one file" % n)
            continue
        u, v = ga[n][:][xa], gb[n][:][xb]
        same = numpy.array_equal(u, v)
        ok &= same
        extra = "" if same else "  (%d differ, max |diff| %.3g)" % (
            (u != v).sum(), numpy.max(numpy.abs(u.astype(float) - v)))
        print("  %-18s %s%s" % (n, "identical" if same else "DIFFERENT",
                                extra))

    print("\n== 4. snr and coa_phase on matched triggers")
    sa, sb = ga['snr'][:][xa].astype(float), gb['snr'][:][xb].astype(float)
    rel = numpy.abs(sb - sa) / sa
    pa, pb = ga['coa_phase'][:][xa].astype(float), \
        gb['coa_phase'][:][xb].astype(float)
    dphi = numpy.abs(numpy.angle(numpy.exp(1j * (pb - pa))))
    for label, v, tol in (("snr |rel diff|", rel, SNR_RTOL),
                          ("coa_phase |diff| rad", dphi, PHASE_ATOL)):
        q = numpy.percentile(v, [50, 99.9]) if len(v) else [0, 0]
        print("  %-22s max %.3g  99.9%% %.3g  median %.3g  exact %d/%d  "
              "tolerance %.0e: %s"
              % (label, v.max() if len(v) else 0, q[1], q[0],
                 (v == 0).sum(), len(v), tol,
                 "PASS" if (v <= tol).all() else "FAIL"))
        ok &= bool((v <= tol).all())

    print("\n== 5. Unmatched triggers")
    for name, only, g in (('reference', only_a, ga), ('new', only_b, gb)):
        idx = {'reference': ia, 'new': ib}[name]
        for k in only:
            s = float(g['snr'][idx[k]])
            near = s < THRESHOLD * (1 + SNR_RTOL)
            print("  only in %s: template_hash %d sample %d snr %.7f%s"
                  % (name, k[0], k[1], s,
                     "  (within threshold band)" if near else ""))
    n_unmatched = len(only_a) + len(only_b)
    limit = int(1e-4 * len(ka))
    print("  unmatched %d, limit %d: %s" % (n_unmatched, limit,
          "PASS" if n_unmatched <= limit else "FAIL"))
    ok &= n_unmatched <= limit

    print("\nOVERALL:", "PASS" if ok else "FAIL (see above; unmatched "
          "triggers still need classifying by hand)")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
