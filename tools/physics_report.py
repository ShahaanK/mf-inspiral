#!/usr/bin/env python
"""Criterion B: physics impact of mf_fast (tapered templates) against stock C3.

Reported, not pass/fail (docs/design-fast.md §2). Triggers are matched on
(template_hash, sample). For matched triggers: the |dSNR|/SNR distribution,
overall and by template duration, and the sigmasq change. For each SNR
threshold (3.8, 5, 6): matched, stock-only and new-only counts, with each
unmatched trigger classified:

  shift-1   a partner in the other file, same template, within +-2 samples
            (the same peak at an adjacent lag);
  same-bin  a partner in the same 2048-sample bin;
  band      no partner, and its SNR is within the observed 99.9th-percentile
            |dSNR| of the threshold (it crossed or missed by that much);
  other     anything else.

Usage: physics_report.py STOCK.hdf NEW.hdf [--bank-mismatch b_bank_mismatch.json]
"""
import argparse
import json

import h5py
import numpy

GPS0, FS = 1186740108, 2048


def load(path):
    g = h5py.File(path, 'r')['H1']
    samp = numpy.round((g['end_time'][:] - GPS0) * FS).astype(numpy.int64)
    return dict(h=g['template_hash'][:], s=samp, snr=g['snr'][:].astype(float),
                sig=g['sigmasq'][:].astype(float), dur=g['template_duration'][:])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stock')
    p.add_argument('new')
    p.add_argument('--bank-mismatch')
    a = p.parse_args()
    A, B = load(a.stock), load(a.new)
    ka = {(h, s): i for i, (h, s) in enumerate(zip(A['h'].tolist(), A['s'].tolist()))}
    kb = {(h, s): i for i, (h, s) in enumerate(zip(B['h'].tolist(), B['s'].tolist()))}
    common = sorted(set(ka) & set(kb))
    xa = numpy.array([ka[k] for k in common])
    xb = numpy.array([kb[k] for k in common])
    rel = numpy.abs(B['snr'][xb] - A['snr'][xa]) / A['snr'][xa]
    sig = numpy.abs(B['sig'][xb] - A['sig'][xa]) / A['sig'][xa]
    q = lambda v: dict(max=float(v.max()), p999=float(numpy.percentile(v, 99.9)),
                       p99=float(numpy.percentile(v, 99)), median=float(numpy.median(v)))
    rep = dict(stock=len(ka), new=len(kb), matched=len(common),
               snr_rel_diff=q(rel), sigmasq_rel_diff=q(sig))
    # by template duration
    edges = [0, 2, 4, 6, 7]
    rep['snr_rel_diff_by_duration'] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (A['dur'][xa] >= lo) & (A['dur'][xa] < hi)
        if m.any():
            rep['snr_rel_diff_by_duration'].append(dict(duration_s=[lo, hi], n=int(m.sum()),
                                                        **q(rel[m])))
    band = rep['snr_rel_diff']['p999']

    def partner(X, h, s):
        """Classify an unmatched trigger against the other file's triggers of the template."""
        m = X['h'] == h
        if not m.any():
            return 'other'
        ds = numpy.abs(X['s'][m] - s)
        if ds.min() <= 2:
            return 'shift-1'
        same_bin = (X['s'][m] - 65536) // 2048 == (s - 65536) // 2048   # approx. bin grid
        return 'same-bin' if same_bin.any() else None

    rep['by_threshold'] = []
    for thr in (3.8, 5.0, 6.0):
        sa = {k for k in ka if A['snr'][ka[k]] > thr}
        sb = {k for k in kb if B['snr'][kb[k]] > thr}
        row = dict(threshold=thr, stock=len(sa), new=len(sb), matched=len(sa & sb))
        for name, only, X, Y, idx in (('stock_only', sa - sb, B, A, ka),
                                      ('new_only', sb - sa, A, B, kb)):
            cls = dict.fromkeys(('shift-1', 'same-bin', 'band', 'other'), 0)
            for (h, s) in only:
                c = partner(X, h, s)
                if c is None:
                    # also counts a trigger that exists in both files but is
                    # above thr in only one of them: that is a band case
                    c = 'band' if Y['snr'][idx[(h, s)]] < thr * (1 + 10 * band) else 'other'
                cls[c] += 1
            row[name] = dict(total=len(only), **cls)
        rep['by_threshold'].append(row)
    if a.bank_mismatch:
        bm = json.load(open(a.bank_mismatch))
        rep['bank_mismatch'] = {k: dict(median_fitting_factor=v['fitting_factor']['median'],
                                        p10_fitting_factor=v['fitting_factor']['p10'],
                                        median_snr_loss=1 - v['fitting_factor']['median'])
                                for k, v in bm.items()}
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
