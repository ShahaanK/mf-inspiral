"""Phase B: block length vs duration grouping, per valid output sample.

Each template t needs support s_t = chirp_t + 2x samples, where chirp_t is its
chirp length from the chosen low-edge start frequency and x is the padding
chosen from b_taper.py. A block of n samples then yields
valid_t(n) = floor((n - s_t) / 2048) * 2048 searchable lags (whole 2048-sample
bins, aligned to PyCBC's), and a 464 s analyze region needs
ceil(950272 / valid_t(n)) blocks.

Cost proxy per (template, block) pair: n log2 n (FFT work). It is a stand-in to
be replaced by measured GPU time per pair in Phase C. Cost per valid output
sample = blocks * c(n) / 950272, summed over templates.

Usage: python b_blocks.py X_SECONDS LOW_WIDTH_HZ OUT.json   (LOW_WIDTH_HZ 0 = stock start)
"""
import sys, json
import numpy as np
import h5py
import pycbc.waveform

X = float(sys.argv[1]); WLOW = float(sys.argv[2]); OUT = sys.argv[3]
FLOW = 20.0; FS = 2048; ANALYZE = 464 * FS; BIN = 2048
NS = [8192, 16384, 32768, 65536]

b = h5py.File('/Users/szkhan/projects/example-workflow/search/bank.hdf', 'r')
p = {k: b[k][:] for k in ('mass1', 'mass2', 'spin1z', 'spin2z')}
apx = [a.decode() if isinstance(a, bytes) else a for a in b['approximant'][:]]
chirp = np.array([pycbc.waveform.get_waveform_filter_length_in_time(
    approximant=apx[i], mass1=p['mass1'][i], mass2=p['mass2'][i], spin1z=p['spin1z'][i],
    spin2z=p['spin2z'][i], f_lower=FLOW - WLOW) for i in range(len(apx))])
support = np.ceil((chirp + 2 * X) * FS)


def cost(n, s):
    """Cost per valid sample of templates with support s at block length n (inf if none fits)."""
    valid = np.floor((n - s) / BIN) * BIN
    out = np.full(s.shape, np.inf)
    ok = valid >= BIN
    out[ok] = np.ceil(ANALYZE / valid[ok]) * n * np.log2(n) / ANALYZE
    return out, valid


res = dict(x=X, low_width=WLOW, chirp_minmax=[float(chirp.min()), float(chirp.max())],
           support_minmax_s=[float(support.min() / FS), float(support.max() / FS)])
single = {}
for n in NS:
    c, v = cost(n, support)
    single[n] = dict(total=float(c.sum()), fits=int(np.isfinite(c).sum()),
                     mean_valid_frac=float(np.mean(v[np.isfinite(c)] / n)) if np.isfinite(c).any() else 0)
res['single'] = single
best_single = min(v['total'] for v in single.values() if v['fits'] == len(support))

# Per-template best n (the ideal of fine grouping).
allc = np.array([cost(n, support)[0] for n in NS])
res['per_template_best'] = dict(total=float(allc.min(0).sum()),
                                n_counts={str(NS[k]): int((allc.argmin(0) == k).sum()) for k in range(len(NS))})

# k contiguous duration groups, boundaries optimised exhaustively over sorted support.
order = np.argsort(support); s_sorted = support[order]
cs = np.array([cost(n, s_sorted)[0] for n in NS])          # (len(NS), T)
pref = np.concatenate([np.zeros((len(NS), 1)), np.cumsum(np.where(np.isfinite(cs), cs, 1e30), 1)], 1)


def group_cost(i, j):
    """Best (cost, n) for templates sorted[i:j] at one n."""
    tot = pref[:, j] - pref[:, i]
    k = int(np.argmin(tot))
    return float(tot[k]), NS[k]


T = len(s_sorted)
for k in (2, 3):
    best = None
    cuts = range(1, T)
    if k == 2:
        for a in cuts:
            c1, n1 = group_cost(0, a); c2, n2 = group_cost(a, T)
            if best is None or c1 + c2 < best[0]:
                best = (c1 + c2, [(0, a, n1), (a, T, n2)])
    else:
        step = 4
        for a in range(step, T - step, step):
            for bb in range(a + step, T, step):
                c1, n1 = group_cost(0, a); c2, n2 = group_cost(a, bb); c3, n3 = group_cost(bb, T)
                if best is None or c1 + c2 + c3 < best[0]:
                    best = (c1 + c2 + c3, [(0, a, n1), (a, bb, n2), (bb, T, n3)])
    res['groups%d' % k] = dict(total=best[0], groups=[
        dict(count=j - i, support_s=[float(s_sorted[i] / FS), float(s_sorted[j - 1] / FS)], n=n)
        for i, j, n in best[1]])
res['best_single_total'] = best_single
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(res, indent=1))
