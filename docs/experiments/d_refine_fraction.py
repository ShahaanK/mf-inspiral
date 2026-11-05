"""Refine-fraction estimate for hierarchical mode at n = 65536 (32 s blocks).

Each template t has support s_t = chirp_t(from 16 Hz) + 2*2 s and valid window
v_t = floor((65536 - s_t) / 2048) * 2048 samples (the approved design). The
1752 s trigger span is cut into consecutive windows of v_t for each template.
A (template, block) pair must be refined if at least one C3 stock trigger of
that template with SNR > rho lies in its window.

C3 triggers are clustered (1 per 2048-sample bin after the symmetric rule and the
cross-segment findchirp step), so this undercounts crossings slightly. It is a
lower bound on what an ideal gate must refine. Gaussian estimate per pair:
1 - exp(-N exp(-rho^2/2)), N = v_t samples (independent-sample approximation;
neighbouring samples of a matched-filter output are correlated, so N overstates
the independent trials).

Usage: python d_refine_fraction.py OUT.json
"""
import sys, json
import numpy as np
import h5py
import pycbc.waveform

FS, BIN, N = 2048, 2048, 65536
T0, SPAN = 1186740140, 1752
C3 = '/Users/szkhan/projects/example-workflow/reference/c3/pycbc/H1-INSPIRAL-1186740108-1800.hdf'
BANK = '/Users/szkhan/projects/example-workflow/search/bank.hdf'
RHOS = [3.8, 4.0, 4.5, 5.0, 5.5, 6.0]

b = h5py.File(BANK, 'r')
apx = [a.decode() if isinstance(a, bytes) else a for a in b['approximant'][:]]
chirp = np.array([pycbc.waveform.get_waveform_filter_length_in_time(
    approximant=apx[i], mass1=b['mass1'][i], mass2=b['mass2'][i], spin1z=b['spin1z'][i],
    spin2z=b['spin2z'][i], f_lower=16.0) for i in range(len(apx))])
valid = np.floor((N - np.ceil((chirp + 4.0) * FS)) / BIN) * BIN

# template_hash per bank row, as PyCBC computes it, to map triggers to rows
from pycbc.waveform.bank import TemplateBank
tb = TemplateBank(BANK)
hashes = np.asarray(tb.table.template_hash, dtype=np.int64)
row_of = {int(h): i for i, h in enumerate(hashes)}

g = h5py.File(C3, 'r')['H1']
snr, tend, th = g['snr'][:], g['end_time'][:], g['template_hash'][:]
rows = np.array([row_of[int(h)] for h in th])
samp = np.round((tend - T0) * FS).astype(np.int64)

out = dict(valid_s=dict(min=float(valid.min() / FS), median=float(np.median(valid) / FS),
                        max=float(valid.max() / FS)), table=[])
short_g = valid <= np.percentile(valid, 10)      # longest templates -> shortest valid windows
long_g = valid >= np.percentile(valid, 90)       # shortest templates -> longest valid windows
for rho in RHOS:
    m = snr > rho
    nblocks = np.ceil(SPAN * FS / valid).astype(int)
    hit = np.zeros(len(valid))
    for t in range(len(valid)):
        sel = m & (rows == t)
        hit[t] = len(np.unique(samp[sel] // int(valid[t]))) / nblocks[t]
    gauss = 1 - np.exp(-valid * np.exp(-rho ** 2 / 2))
    w = nblocks / nblocks.sum()
    row = dict(rho=rho, measured=float(np.sum(hit * w)), gauss=float(np.sum(gauss * w)),
               measured_shortest_valid=float(np.mean(hit[short_g])),
               gauss_shortest_valid=float(np.mean(gauss[short_g])),
               measured_longest_valid=float(np.mean(hit[long_g])),
               gauss_longest_valid=float(np.mean(gauss[long_g])))
    out['table'].append(row)
    print(row, flush=True)
out['shortest_valid_group_s'] = [float(valid[short_g].min() / FS), float(valid[short_g].max() / FS)]
out['longest_valid_group_s'] = [float(valid[long_g].min() / FS), float(valid[long_g].max() / FS)]
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'table'}))
