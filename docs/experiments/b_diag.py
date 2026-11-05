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
