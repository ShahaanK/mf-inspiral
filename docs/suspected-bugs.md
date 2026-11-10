# Suspected bugs and undocumented behaviour in matchedfilter

Package: matchedfilter 0.1.0a9 in env `gw`. I haven't patched it or contacted anyone. Each entry
has a minimal repro, which runs on CPU.

## 1. `run_blocks` / `run_series` output is 1/n of `run()`'s, and this isn't documented (2026-11-05)

**Severity:** documentation or consistency. The results are self-consistent and correct up to a
constant; mf_fast accounts for it.

**What:**
- `MatchedFilter.run()` returns the **unnormalised** inverse transform of `data · conj(template)`,
  as documented: "a perfect match returns n * energy" (`_filter.py:307-310`, `matchedfilter.h:11`).
- For the same block and template spectrum, `run_blocks()` (and `run_series()`, which shares its
  implementation) returns exactly **1/n** of that. That equals the time-domain correlation
  Σₘ s[j+m]·conj(h[m]), with h = ifft(spectrum).
- Lags agree; only the scale differs. No docstring in the installed package mentions the
  difference (searched `_filter.py`, `time_domain.py`, `_series.py`, `tutorial.py`).

**Minimal repro** (CPU, about 1 s):

```python
import numpy, matchedfilter as mf
rng = numpy.random.default_rng(1)
n = 4096
blk = (rng.standard_normal(n) + 1j*rng.standard_normal(n)).astype(numpy.complex64)
h = numpy.zeros(n, numpy.complex64); h[:5] = 1; h[-5:] = 1j
spec = numpy.fft.fft(h).astype(numpy.complex64)
w = (10, n - 10)
f = mf.MatchedFilter(n, 1, 1); f.set_templates(spec[None])
f.set_data(numpy.fft.fft(blk)[None].astype(numpy.complex64))
a = f.run(binsize=n, window=w, raw=True)
g = mf.MatchedFilter(n, 1, 1); g.set_templates(spec[None])
b = g.run_blocks(blk, numpy.array([0]), numpy.array([w[0]]), numpy.array([w[1]]),
                 binsize=n, raw=True)
print(a[0][0,0,0], b[0][0,0,0], abs(a[1][0,0,0]) / abs(b[1][0,0,0]))   # same lag, ratio == n
```

**Measured** ratio of `run()` to `run_blocks()`: 1024 at n = 1024, 4096 at n = 4096, 65536 at
n = 65536.

**Impact:**
- The threshold passed to `run_series` / `run_blocks` is compared with the 1/n-scaled values.
  A caller who reuses a `run()` threshold gets results wrong by a factor of n, silently.
- mf_fast scales templates for the series convention: `templates.TemplateCache.block_scale`, tested
  in `tests/test_fast_blocks.py`.

**GPU:** the GPU series path uses the same 1/n convention: the GPU SNRs land within
2.3e-5 of the CPU ones with the same `block_scale` (see #2 for the remaining 1.2e-5).

## 2. The GPU transform loses gain that grows with n: peaks 1.2e-5 low at n = 65536 (2026-11-09, confirmed on synthetic data)

**Severity:** accuracy. Small, but above the 1e-5 tolerance, and it fails check A for the GPU path.

**What:**
- Same job, same saved template cache (`--load-cache`), only `--device` differs:
  - CPU: SNR within 4.4e-6 of reference A (median 0).
  - GPU: every SNR is low by 1.23e-5 (median −1.23e-5, spread σ 2.8e-6, range −2.31e-5 to
    −2.1e-6 over 221,281 matched triggers).
- The offset is the same in all four segments, and for every template: per-template medians run
  from −1.29e-5 to −1.17e-5. It doesn't correlate with SNR (r = 0.002) or with σ² (r = 0.01).
- Identical on all three repeats and in every n = 65536 sweep configuration (template batch
  4–178, data batch 4 or all).
- mf_fast passes nothing device-specific except `device=` to `MatchedFilter`
  (`mf_fast/search.py:132`). So the offset comes from inside matchedfilter's GPU path.
- Removing the constant still leaves a GPU spread (σ 2.8e-6) about 4× the CPU's (7.6e-7), with
  extremes near ±1.1e-5. The scatter is a second, separate effect.

**Minimal repro:** `tools/gpu_gain_check.py`, which needs the GPU. It filters synthetic data at
n = 4096–65536 with `run_blocks` and `run()` on CPU and GPU, and compares both with float64 numpy.
- An offset that grows with log₂ n would point at the FFT twiddles.
- One that's the same at every n would point at a normalisation step.

**Result** (`~/benchmark-runs/20261109-194150-followup/gpu-gain-check.log`): median of
|GPU|/float64 − 1 over the bin peaks. The CPU stays within 1.5e-7 at every n.

| n | `run()` (inverse only) | `run_blocks` (forward + inverse) |
|---|---|---|
| 4096 | −1.1e-6 | −2.2e-6 |
| 8192 | −1.2e-6 | −2.5e-6 |
| 16384 | −1.1e-6 | −2.3e-6 |
| 32768 | −3.3e-6 | −6.4e-6 |
| 65536 | −6.2e-6 | **−1.23e-5** |

- **Reproduces the job exactly:** `run_blocks` at 65536 gives −1.23e-5, the same offset mf_fast
  shows on real data. So the offset is matchedfilter's, not mf_fast's.
- **Per-transform:** `run_blocks` does two GPU transforms (the block forward FFT and the
  inverse), and its deficit is twice `run()`'s at every n.
- **Grows with n, but not as log₂ n:** it's flat at about 1.1e-6 per transform up to 16384,
  then roughly doubles with each doubling of n. The jump starts at 32768, where the GPU pair
  limit 2²⁸/n starts to apply and the kernels change (`matchedfilter-internals.md` §2). Which
  step loses the gain (twiddles, a multi-pass stage, or a scale) is unknown.
- **Not visible in `python -m matchedfilter.benchmark`:** its accuracy column checks the CPU
  result only. Its GPU column is timing only (`benchmark.py:248-285`).

**Impact:** the GPU path fails check A (SNR tolerance; unmatched 21–27 vs a limit of 22, all
near-threshold or one-sample peak moves). mf_fast's GPU results are reported with this caveat.
