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
for ti in (101, 229, 427):
    tmpl = bank[ti]; dur = tmpl.chirp_length
    H = np.zeros(N, np.complex128); H[:flen] = tmpl.numpy()
    nz = np.nonzero(np.abs(H) > 0)[0]
    print("template %d dur %.2f s: H nonzero bins %d..%d (%.2f..%.2f Hz)" % (ti, dur, nz[0], nz[-1], f[nz[0]], f[nz[-1]]))
    h = np.fft.ifft(H); e = np.abs(h) ** 2; tot = e.sum()
    lag = np.arange(N); lag[lag >= N // 2] -= N
    pk = lag[np.argmax(e)]
    print("  |h| peak at lag %.3f s; energy within [-dur-x, +x]:" % (pk / 2048), end='')
    for x in (0.25, 1, 4, 16, 64):
        m = (lag >= -(dur + x) * 2048) & (lag <= x * 2048)
        print(" x=%gs: 1-%.1e" % (x, 1 - e[m].sum() / tot), end='')
    print()
    # S-weighted: error variance in SNR units of dropping lags outside the window, measured on data
    sig2 = float(tmpl.sigmasq(segs[0].psd)); norm = 4 * df / np.sqrt(sig2)
    zf = np.fft.ifft(a * np.conj(H)) * N
    for x in (0.25, 1, 4, 16):
        L = int(np.ceil((dur + x) * 2048)); R = int(np.ceil(x * 2048))
        keep = (lag >= -L) & (lag <= R)
        ht = np.where(keep, h, 0)
        zt = np.fft.ifft(a * np.conj(np.fft.fft(ht))) * N    # exact linear-in-tail effect, full length
        err = (np.abs(zt - zf) * norm)[an]
        snr = (np.abs(zf) * norm)[an]
        mm = snr > 3.8
        print("    support -(dur+%gs)..+%gs (%.1f s): max |dSNR| %.2e, rms %.2e, max rel (snr>3.8) %.2e"
              % (x, x, (L + R) / 2048, err.max(), np.sqrt((err ** 2).mean()), (err[mm] / snr[mm]).max()))
