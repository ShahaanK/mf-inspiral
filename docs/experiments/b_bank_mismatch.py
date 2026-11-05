"""Typical template-bank mismatch, the yardstick for criterion B.

1. Fitting factor: for random signals drawn uniformly from the bank's construction
   region (pycbc_brute_bank: mass1, mass2 in [10, 40], spin1z, spin2z in [0, 0.2],
   IMRPhenomD, f_lower 20), the best match over the bank, maximised over time and
   phase (pycbc.filter.match). 1 - FF is the SNR lost to bank discreteness.
2. Nearest-neighbour match: for each template, its best match against the others.

Matches use the 30 templates nearest in tau0 (chirp time from 20 Hz) as candidates,
at delta_f = 1/32 Hz (every template fits in 32 s). Two PSDs: C3 segment 0
(measured, interpolated to 1/32 Hz), and aLIGOZeroDetLowPower (the bank's design PSD).

Usage: python b_bank_mismatch.py OUT.json [NSIG]
"""
import sys, json, time
import numpy as np
import h5py
import pycbc.psd, pycbc.waveform
from pycbc import DYN_RANGE_FAC, psd as pycbc_psd, strain
from pycbc.filter import match
from pycbc.conversions import tau0_from_mass1_mass2

sys.path.insert(0, '/Users/szkhan/projects/mf-inspiral')
from mf_inspiral.cli import parse_args

OUT = sys.argv[1]
NSIG = int(sys.argv[2]) if len(sys.argv) > 2 else 200
DF, FLOW, FS = 1.0 / 32, 20.0, 2048
FLEN = int(FS / 2 / DF) + 1
K = 30

S = '/Users/szkhan/projects/example-workflow/search'
argv = f"""--processing-scheme cpu:1 --bank-file {S}/bank.hdf
--frame-files {S}/H-H1_LOSC_CLN_4_V1-1186740069-3584.gwf --channel-name H1:LOSC-STRAIN
--gps-start-time 1186740108 --gps-end-time 1186741908
--trig-start-time 1186740140 --trig-end-time 1186741892 --output /dev/null
--pad-data 8 --strain-high-pass 15 --sample-rate 2048 --segment-length 512
--segment-start-pad 32 --segment-end-pad 16 --allow-zero-padding --taper-data 1
--psd-estimation median --psd-segment-length 16 --psd-segment-stride 8
--psd-inverse-length 16 --psd-num-segments 63
--autogating-threshold 50 --autogating-cluster 0.5 --autogating-width 0.25
--autogating-taper 0.25 --autogating-pad 16 --low-frequency-cutoff 20
--enable-bank-start-frequency --snr-threshold 3.8 --cluster-window 1
--cluster-function symmetric""".split()

t0 = time.time()
opt = parse_args(argv)
gws = strain.from_cli(opt, dyn_range_fac=DYN_RANGE_FAC)
ss = strain.StrainSegments.from_cli(opt, gws)
segs = ss.fourier_segments()
pycbc_psd.associate_psds_to_segments(opt, segs, gws, ss.freq_len, ss.delta_f, FLOW,
                                     dyn_range_factor=DYN_RANGE_FAC, precision='double')
psds = {'C3_segment0': pycbc.psd.interpolate(segs[0].psd, DF),
        'aLIGOZeroDetLowPower': pycbc.psd.from_string('aLIGOZeroDetLowPower', FLEN, DF, FLOW)}
for k in psds:
    p = psds[k]
    if len(p) < FLEN:
        p.resize(FLEN)
    psds[k] = p[:FLEN]
print("PSDs ready %.0f s" % (time.time() - t0), flush=True)

b = h5py.File(S + '/bank.hdf', 'r')
P = {k: b[k][:] for k in ('mass1', 'mass2', 'spin1z', 'spin2z')}
T = len(P['mass1'])
tau0 = tau0_from_mass1_mass2(P['mass1'], P['mass2'], FLOW)


def wf(m1, m2, s1, s2):
    hp, _ = pycbc.waveform.get_fd_waveform(approximant='IMRPhenomD', mass1=m1, mass2=m2,
                                           spin1z=s1, spin2z=s2, f_lower=FLOW, delta_f=DF)
    hp.resize(FLEN)
    return hp


bank = [wf(P['mass1'][i], P['mass2'][i], P['spin1z'][i], P['spin2z'][i]) for i in range(T)]
print("bank generated %.0f s" % (time.time() - t0), flush=True)


def best_match(h, tau, exclude, p):
    cand = np.argsort(np.abs(tau0 - tau))
    cand = [c for c in cand if c != exclude][:K]
    return max(match(h, bank[c], psd=p, low_frequency_cutoff=FLOW)[0] for c in cand)


rng = np.random.default_rng(20261105)
res = {}
for name, p in psds.items():
    ff = []
    for i in range(NSIG):
        m1, m2 = rng.uniform(10, 40, 2)
        s1, s2 = rng.uniform(0, 0.2, 2)
        ff.append(best_match(wf(m1, m2, s1, s2), tau0_from_mass1_mass2(m1, m2, FLOW), -1, p))
    nn = [best_match(bank[t], tau0[t], t, p) for t in range(T)]
    ff, nn = np.array(ff), np.array(nn)
    res[name] = dict(
        fitting_factor=dict(n=NSIG, median=float(np.median(ff)), mean=float(ff.mean()),
                            p10=float(np.percentile(ff, 10)), min=float(ff.min()),
                            frac_below_0p95=float(np.mean(ff < 0.95))),
        nearest_neighbour=dict(n=T, median=float(np.median(nn)), p10=float(np.percentile(nn, 10)),
                               min=float(nn.min())))
    print(name, json.dumps(res[name]), "%.0f s" % (time.time() - t0), flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
