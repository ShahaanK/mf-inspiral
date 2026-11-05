"""Phase B check: can overlap-save short blocks reproduce the 2^20 stock correlation?

Front end is PyCBC with the C3 settings (no PSD variation: it does not touch z).
Truth: z_full = unnormalised IFFT_N(D conj H), float64, D = overwhitened segment
zeroed outside [kmin, kmax), H = FilterBank template (as stock).
Short blocks: d = ifft(D) (complex), h = ifft(H); keep h on lags [-L, R]
(chosen by an energy tolerance eps in the PSD-weighted norm), place it in an
n-point buffer, correlate each block circularly, keep lags [L, n-R).
Errors are reported in SNR units (z * 4 df / sqrt(sigmasq)).
"""
import sys, json, time
import numpy as np

sys.path.insert(0, '/Users/szkhan/projects/mf-inspiral')
from mf_inspiral.cli import parse_args
import pycbc
from pycbc import DYN_RANGE_FAC, psd, strain, waveform
from pycbc.types import complex64, zeros
from pycbc.filter.matchedfilter import get_cutoff_indices

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

opt = parse_args(argv)
t0 = time.time()
gws = strain.from_cli(opt, dyn_range_fac=DYN_RANGE_FAC)
ss = strain.StrainSegments.from_cli(opt, gws)
flen, tlen, df = ss.freq_len, ss.time_len, ss.delta_f
segs = ss.fourier_segments()
psd.associate_psds_to_segments(opt, segs, gws, flen, df, 20.0,
                               dyn_range_factor=DYN_RANGE_FAC, precision='single')
for s in segs:
    s /= s.psd
kmin, kmax = get_cutoff_indices(20.0, None, df, tlen)
N = tlen
mem = zeros(tlen, dtype=complex64)
bank = waveform.FilterBank(opt.bank_file, flen, df, low_frequency_cutoff=None,
                           dtype=complex64, out=mem)
print("front end %.1f s, N=%d, %d segments, %d templates" % (time.time() - t0, N, len(segs), len(bank)),
      flush=True)

durs = np.array([bank.table[i].template_duration if hasattr(bank.table[i], 'template_duration') else 0
                 for i in range(len(bank))])
# Pick templates spread over duration: generate all durations cheaply is not possible
# without generating; use mtotal order as a proxy and pick 12 spread indices.
mt = bank.table['mass1'] + bank.table['mass2']
order = np.argsort(mt)
pick = sorted(set(order[np.linspace(0, len(order) - 1, 12).astype(int)].tolist()))

N = tlen
a = np.zeros(N, np.complex128); a[kmin:kmax] = segs[0].numpy()[kmin:kmax]
d = np.fft.ifft(a); an = segs[0].analyze
f = np.arange(N) * df
lag = np.arange(N); lag[lag >= N // 2] -= N

def err_for(Hx, dur, x, sig2):
    """SNR error from dropping lags outside -(dur+x)..+x, for spectrum Hx (vs Hx itself)."""
    h = np.fft.ifft(Hx)
    L = int(np.ceil((dur + x) * 2048)); R = int(np.ceil(x * 2048))
    ht = np.where((lag >= -L) & (lag <= R), h, 0)
    norm = 4 * df / np.sqrt(sig2)
    zf = np.fft.ifft(a * np.conj(Hx)) * N
    zt = np.fft.ifft(a * np.conj(np.fft.fft(ht))) * N
    err = (np.abs(zt - zf) * norm)[an]; snr = (np.abs(zf) * norm)[an]
    e = np.abs(h) ** 2
    return err.max(), (err[snr > 3.8] / snr[snr > 3.8]).max(), 1 - e[(lag >= -L) & (lag <= R)].sum() / e.sum()

def ramp(f, f0, f1):
    """0 below f0, 1 above f1, cosine in between."""
    r = np.clip((f - f0) / (f1 - f0), 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * r)

ti = 427
tmpl = bank[ti]; dur = tmpl.chirp_length
H = np.zeros(N, np.complex128); H[:flen] = tmpl.numpy()
sig2 = float(tmpl.sigmasq(segs[0].psd))

# (a) overlap-save with a truncated template == full-length filter with the same truncated template
x = 4.0; n = 65536
h = np.fft.ifft(H)
L = int(np.ceil((dur + x) * 2048)); R = int(np.ceil(x * 2048))
ht = np.where((lag >= -L) & (lag <= R), h, 0)
zt_full = np.fft.ifft(a * np.conj(np.fft.fft(ht))) * N
hn = np.zeros(n, np.complex128); hn[:R + 1] = ht[:R + 1]; hn[n - L:] = ht[N - L:]
Hn = np.conj(np.fft.fft(hn)); valid = ((n - (L + R + 1)) // 2048) * 2048
worst = 0.0; start = an.start
while start < an.stop:
    stop = min(start + valid, an.stop)
    idx = (start - L + np.arange(n)) % N
    c = np.fft.ifft(np.fft.fft(d[idx]) * Hn) * N
    zs = c[L:L + stop - start]
    worst = max(worst, float(np.max(np.abs(zs - zt_full[start:stop]) / np.max(np.abs(zt_full[an])))))
    start = stop
print("(a) overlap-save n=%d (valid %.1f s) vs full-length, same truncated template: max rel diff %.1e"
      % (n, valid / 2048, worst))

# (b) which spectral edge makes the tail? template 427 spans 20 Hz .. Nyquist bin
print("(b) template %d, dur %.2f s; support -(dur+x)..+x" % (ti, dur))
variants = [("stock template", np.ones(N)),
            ("low edge 20->22 Hz smoothed", ramp(f, 20, 22)),
            ("high edge smoothed (1000->1023 Hz)", 1 - ramp(f, 1000, 1023)),
            ("both edges smoothed", ramp(f, 20, 22) * (1 - ramp(f, 1000, 1023)))]
for name, w in variants:
    Hx = H * w
    s2 = sig2 * 1.0  # norm uses stock sigma; only relative behaviour matters here
    out = ["x=%gs: rel %.1e (tail E %.0e)" % ((x,) + tuple(err_for(Hx, dur, x, s2)[1:])) for x in (0.25, 1, 4, 16)]
    print("   %-36s %s" % (name, "; ".join(out)))
