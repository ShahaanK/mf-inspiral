"""Phase B: edge-taper trade-off for short-block filtering.

For each taper variant (low edge x high edge), using PyCBC's standard fd_taper
(half-Kaiser, beta=8; pycbc/waveform/utils.py:397):

  1. algorithm error: largest |dSNR|/SNR over SNR > 3.8 samples, from cutting the
     variant's time-domain response to -(chirp+x)..+x, against the full 2^20
     filter with the SAME variant template (float64), on 6 templates x 2 segments;
  2. in-band identity of out-of-band low tapers vs the stock template;
  3. physics: SNR loss for a signal equal to the stock template, 1 - overlap, over
     [kmin, kmax) with each segment's PSD, for all 431 templates (written by
     the second part, from stock templates and the in-band windows).

Usage: python b_taper.py OUT.json
"""
import sys, json, time
import numpy as np

sys.path.insert(0, '/Users/szkhan/projects/mf-inspiral')
from mf_inspiral.cli import parse_args
from pycbc import DYN_RANGE_FAC, psd, strain, waveform
from pycbc.types import complex64, zeros, FrequencySeries
from pycbc.filter.matchedfilter import get_cutoff_indices
from pycbc.waveform.utils import fd_taper
import pycbc.waveform

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

FLOW = 20.0
opt = parse_args(argv)
t0 = time.time()
gws = strain.from_cli(opt, dyn_range_fac=DYN_RANGE_FAC)
ss = strain.StrainSegments.from_cli(opt, gws)
flen, tlen, df = ss.freq_len, ss.time_len, ss.delta_f
segs = ss.fourier_segments()
psd.associate_psds_to_segments(opt, segs, gws, flen, df, FLOW,
                               dyn_range_factor=DYN_RANGE_FAC, precision='single')
for s in segs:
    s /= s.psd
kmin, kmax = get_cutoff_indices(FLOW, None, df, tlen)
N = tlen
mem = zeros(tlen, dtype=complex64)
bank = waveform.FilterBank(opt.bank_file, flen, df, low_frequency_cutoff=None,
                           dtype=complex64, out=mem)
print("front end %.1f s" % (time.time() - t0), flush=True)

lag = np.arange(N); lag[lag >= N // 2] -= N
f = np.arange(flen) * df


def generate(ti, f_start):
    """Template ti as FilterBank makes it, but generated from f_start with
    f_ref = FLOW (PyCBC's get_td_waveform_from_fd convention)."""
    row = bank.table[ti]
    f_end = bank.end_frequency(ti)
    if f_end is None or f_end >= bank.filter_length * df:
        f_end = (bank.filter_length - 1) * df
    out = zeros(bank.filter_length, dtype=complex64)
    h = pycbc.waveform.get_waveform_filter(
        out, row, approximant=bank.approximant(ti), f_lower=f_start, f_final=f_end,
        delta_f=df, delta_t=bank.delta_t, distance=1.0 / DYN_RANGE_FAC, f_ref=FLOW)
    return h


def pycbc_rule_start(ti):
    """f_lower lowered by 1% steps until the filter duration grows by 1.5x
    (waveform.py:786-792)."""
    row = bank.table[ti]
    p = dict(approximant=bank.approximant(ti), mass1=row['mass1'], mass2=row['mass2'],
             spin1z=row['spin1z'], spin2z=row['spin2z'], f_lower=FLOW)
    dur = full = pycbc.waveform.get_waveform_filter_length_in_time(**p)
    while full < dur * 1.5:
        full = pycbc.waveform.get_waveform_filter_length_in_time(**p)
        p['f_lower'] *= 0.99
    return p['f_lower']


def high_edge(H, wh):
    """Right fd_taper over [f_hi - wh, f_hi], f_hi = just past the last nonzero bin."""
    if not wh:
        return H
    nz = np.nonzero(np.abs(H.numpy()) > 0)[0]
    f_hi = (nz[-1] + 1) * df
    return fd_taper(H, f_hi - wh, f_hi, beta=8, side='right')


LOWS = [('none', None), ('oob1', 1.0), ('oob2', 2.0), ('oob4', 4.0), ('oob8', 8.0),
        ('pycbc', 'rule'), ('inband2', -2.0)]
HIGHS = [0, 4, 16, 64]
XS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
TEMPLATES = [101, 155, 229, 246, 405, 427]
SEGS = [0, 3]

Ds = {}
for si in SEGS:
    a = np.zeros(N, np.complex128); a[kmin:kmax] = segs[si].numpy()[kmin:kmax]
    Ds[si] = a

rows = []
ident = []
for ti in TEMPLATES:
    stock = bank[ti]
    Hs = stock.numpy().astype(np.complex128)
    sig2s = {si: float(stock.sigmasq(segs[si].psd)) for si in SEGS}
    for lname, w in LOWS:
        if w is None:
            base, f_start = FrequencySeries(stock.numpy().astype(np.complex128), delta_f=df), FLOW
            chirp = stock.chirp_length
        elif w == 'rule' or w > 0:
            f_start = pycbc_rule_start(ti) if w == 'rule' else FLOW - w
            g = generate(ti, f_start)
            chirp = g.chirp_length
            base = fd_taper(FrequencySeries(g.numpy().astype(np.complex128), delta_f=df), f_start, FLOW,
                            beta=8, side='left')
            Hb = base.numpy()
            ident.append(dict(t=ti, low=lname, f_start=f_start,
                              inband_maxdiff=float(np.max(np.abs(Hb[kmin:] - Hs[kmin:]))
                                                   / np.max(np.abs(Hs)))))
        else:   # in-band ramp FLOW .. FLOW+|w|
            f_start = FLOW
            base = fd_taper(FrequencySeries(stock.numpy().astype(np.complex128), delta_f=df), FLOW,
                            FLOW - w, beta=8, side='left')
            chirp = stock.chirp_length
        for wh in HIGHS:
            Hv = high_edge(base, wh).numpy().astype(np.complex128)
            H = np.zeros(N, np.complex128); H[:flen] = Hv
            h = np.fft.ifft(H)
            for si in SEGS:
                a = Ds[si]
                norm = 4 * df / np.sqrt(sig2s[si])
                zf = np.fft.ifft(a * np.conj(H)) * N
                an = segs[si].analyze
                snr = (np.abs(zf) * norm)[an]
                m = snr > 3.8
                for x in XS:
                    L = int(np.ceil((chirp + x) * 2048)); R = int(np.ceil(x * 2048))
                    ht = np.where((lag >= -L) & (lag <= R), h, 0)
                    zt = np.fft.ifft(a * np.conj(np.fft.fft(ht))) * N
                    err = (np.abs(zt - zf) * norm)[an]
                    rows.append(dict(t=ti, seg=si, low=lname, high=wh, x=x, chirp=float(chirp),
                                     f_start=float(f_start), support_s=(L + R + 1) / 2048,
                                     max_rel=float((err[m] / snr[m]).max()),
                                     max_abs=float(err.max())))
        print("t%d %s done (%.0f s)" % (ti, lname, time.time() - t0), flush=True)

# Physics part: SNR loss vs stock for in-band windows, all templates, all PSDs.
# Out-of-band low tapers leave bins >= kmin unchanged (checked above), so only the
# high-edge window and the in-band low ramp enter.
loss = []
for ti in range(len(bank)):
    stock = bank[ti]
    Hs = stock.numpy().astype(np.complex128)
    for wl in (0.0, 2.0):
        lowwin = np.ones(flen)
        if wl:
            lowwin = fd_taper(FrequencySeries(np.ones(flen, np.complex128), delta_f=df),
                              FLOW, FLOW + wl, beta=8, side='left').numpy().real
        for wh in HIGHS:
            win = lowwin * high_edge(FrequencySeries(np.ones(flen, np.complex128) * (np.abs(Hs) > 0),
                                                     delta_f=df), wh).numpy().real
            for si, sg in enumerate(segs):
                w = np.zeros(flen); w[kmin:kmax] = 1.0 / sg.psd.numpy()[kmin:kmax]
                p = np.abs(Hs) ** 2 * w
                ov = np.sum(p * win) / np.sqrt(np.sum(p) * np.sum(p * win ** 2))
                loss.append(dict(t=ti, seg=si, inband_low=wl, high=wh, loss=float(1 - ov)))
    if ti % 50 == 0:
        print("loss t%d (%.0f s)" % (ti, time.time() - t0), flush=True)

json.dump(dict(rows=rows, identity=ident, loss=loss), open(sys.argv[1], 'w'))
print("wrote", sys.argv[1], "%.0f s" % (time.time() - t0))
