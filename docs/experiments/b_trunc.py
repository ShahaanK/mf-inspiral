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

EPS = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
NS = [16384, 32768, 65536]
SEGS = [0, 3]
D = []
for si in SEGS:
    a = np.zeros(N, np.complex128)
    a[kmin:kmax] = segs[si].numpy()[kmin:kmax]
    D.append(a)
dts = [np.fft.ifft(a) for a in D]       # complex overwhitened series, numpy-normalised

rows = []
for ti in pick:
    tmpl = bank[ti]
    H = np.zeros(N, np.complex128)
    H[:flen] = tmpl.numpy()
    h = np.fft.ifft(H)
    for si, a, d in zip(SEGS, D, dts):
        sig2 = float(tmpl.sigmasq(segs[si].psd))
        norm = 4.0 * df / np.sqrt(sig2)
        zf = np.fft.ifft(a * np.conj(H)) * N          # unnormalised IFFT
        an = segs[si].analyze
        # PSD-weighted energy of h per lag: weight lag energy by the response of
        # the overwhitened data is not separable; use plain |h|^2 for support choice
        # and measure the actual error directly.
        e = np.abs(h) ** 2
        tot = e.sum()
        # lags as signed: m in [-N/2, N/2)
        lag = np.arange(N); lag[lag >= N // 2] -= N
        o = np.argsort(np.abs(lag))   # not used for choice; support chosen asymmetrically below
        csum_neg = np.cumsum(e[::-1])          # energy at lags -1, -2, ... (from the end)
        csum_pos = np.cumsum(e)                 # energy at lags 0, 1, ...
        for eps in EPS:
            # smallest L, R with tail energy (beyond -L and beyond R) each <= eps*tot/2
            negtail = tot - csum_neg - 0  # energy at lags < -(i+1)... approximate below
            # energy strictly beyond lag -L: sum_{m=L+1}^{N/2} e[N-m]
            neg = e[N // 2:][::-1]               # lags -1 .. -N/2 in order
            pos = e[:N // 2]                     # lags 0 .. N/2-1
            neg_beyond = neg.sum() - np.cumsum(neg)   # beyond lag -(i+1)
            pos_beyond = pos.sum() - np.cumsum(pos)   # beyond lag i
            L = int(np.argmax(neg_beyond <= eps * tot / 2)) + 1
            R = int(np.argmax(pos_beyond <= eps * tot / 2))
            for n in NS:
                span = L + R + 1
                valid = ((n - span) // 2048) * 2048
                if valid < 2048:
                    rows.append(dict(t=int(ti), seg=si, eps=eps, n=n, L=L, R=R, valid=0))
                    continue
                hn = np.zeros(n, np.complex128)
                hn[:R + 1] = h[:R + 1]
                hn[n - L:] = h[N - L:]
                Hn = np.conj(np.fft.fft(hn))
                maxerr = 0.0; maxsnr_err_rel = 0.0
                start = an.start
                while start < an.stop:
                    stop = min(start + valid, an.stop)
                    bs = start - L                          # block start in segment samples
                    idx = (bs + np.arange(n)) % N
                    c = np.fft.ifft(np.fft.fft(d[idx]) * Hn) * N   # matches z_full scaling
                    zs = c[L:L + (stop - start)]
                    zt = zf[start:stop]
                    err = np.abs(zs - zt) * norm
                    maxerr = max(maxerr, float(err.max()))
                    snr = np.abs(zt) * norm
                    m = snr > 3.8
                    if m.any():
                        maxsnr_err_rel = max(maxsnr_err_rel, float((err[m] / snr[m]).max()))
                    start = stop
                rows.append(dict(t=int(ti), seg=si, eps=eps, n=n, L=L, R=R, valid=valid,
                                 max_abs_snr_err=maxerr, max_rel_err_above_thr=maxsnr_err_rel,
                                 dur=float(tmpl.chirp_length)))
    print("template %d (dur %.2f s, mtot %.1f) done" % (ti, tmpl.chirp_length, mt[ti]), flush=True)

json.dump(rows, open(sys.argv[1], 'w'), indent=0)
print("wrote", sys.argv[1])
