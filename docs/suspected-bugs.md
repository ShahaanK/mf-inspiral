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

**Not checked:** whether the GPU series path uses the same convention. That's part of the
Phase C GPU check.
