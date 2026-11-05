# matchedfilter internals (Phase A of the fast-path work)

2026-11-05. Read-only. Package: matchedfilter 0.1.0a9 in env `gw`, at
`~/miniforge3/envs/gw/lib/python3.11/site-packages/matchedfilter/`. All paths
below are relative to that directory. The GPU is not visible inside the Claude
sandbox, so **nothing here was run on the GPU**. Everything is from reading the
Python, the C header and the shipped Metal source.

- **(read)** means it's stated in code or docs.
- **(inferred)** means I'm reasoning from the code.

Context: after the Oct 28 advisor meeting, the goal is maximum matched-filter
throughput on the M2 GPU, by filtering batches of templates × short segments.
See `design-fast.md` (Phase B, same folder) for the design this feeds. Older
drop-in notes (Phases A–C3 of mf_inspiral) stay in
`~/projects/example-workflow/notes/`.

## 1. Where the speedup comes from

### 1a. One fused kernel per pair, peaks only (read)

The flat GPU filter is `fusedTierB` in `metal/tierb_<n>.metal`. The entry point
is at `tierb_4096.metal:1773`, and the body is `filterPair_0`.

- **One threadgroup per (data, template) pair.**
  `pair = threadgroup_position_in_grid.x`, `d = pair / ntmpl`,
  `t = pair % ntmpl`.
- **The correlation product is fused into the load.** Each thread loads its
  data and template samples from device memory and multiplies them at once:
  `r[i] = cmulConj(data[d*n + idx], tmpl[t*n + idx])`.
  - Nothing intermediate is written.
  - The template conjugation is applied in the multiply, with the template
    stored as ingested.
- **The inverse FFT runs in registers plus threadgroup "staging" memory**
  (`transform_0`). At n=4096 that's 16 complex values per thread and 256
  threads, with 32 KB of staging (`array<uint, 8192>`). At n=65536 it's 64 per
  thread and 64 KB of staging, or 32 KB in the `_lds32` build.
- **No output reorder.** After the transform, each thread asks
  `slotToIndex(tid*16 + i)` which natural lag each register slot holds. The
  bit-reversal or transpose pass of a general FFT is never done, because a
  per-bin maximum doesn't care about order.
- **The peak scan is fused too.**
  - |z|² = re² + im² is compared as float bits (`as_type<uint>`), which works
    because magnitudes are non-negative.
  - Each value goes into its bin with
    `atomic_fetch_max_explicit(&stg[bin], mag)` in threadgroup memory,
    seeded with the threshold's bits.
  - So only values that **exceed** the threshold (strict `>`) can win.
- **Only the peaks are written back to device memory**: `peakIdx` and
  `peakVal` per bin. The n-point correlation series never leaves the chip.

Full-output mode (`CorrelationFilter`) keeps the fused product and transform but
writes every lag (README "Performance"). Peak-only saves those n·8 bytes of
stores per pair.

### 1b. Ingest is paid once per spectrum, not per pair (read)

`matchedfilter.h:43-48`: "Ingest here only rearranges: templates are conjugated,
and both sides are stored in the layout stage A walks … paid once per segment,
so it stays amortised however many pairs run." Inputs must already be
frequency-domain spectra (natural order, unnormalised forward FFT).

### 1c. Batching (read and inferred)

- **(read)** One `run()` dispatches all nd × nt pairs as one grid
  (`_filter.py:617-648` → `_gpuexec.py:793 peaks`). Pairs are ordered
  template-fastest (`d = pair / ntmpl`).
- **(read)** Above the per-dispatch pair limit, it splits into several
  dispatches (`_filter.py:604-639`). For n ≥ 32768 the limit is 2²⁸/n pairs,
  i.e. 4096 pairs at n=65536 and 8192 at 32768, to keep under driver timeouts.
- **(read)** Apple silicon uses `MTLResourceStorageModeShared`
  (`_mtlcompute.py:46-50`), so no host↔device copy is needed. Spectra are
  uploaded (made resident) only when they change (`upload_data=self._ddirty`,
  `_filter.py:641-648`).
- **(inferred) The kernel has no explicit tiling of templates × data.** Every
  pair reads its data row and template from device memory. Reuse can only come
  from the GPU's cache hierarchy:
  - consecutive threadgroups share the data row, which is likely to stay
    cached;
  - a template is reused across data rows only if the template block of a
    dispatch fits in cache (one 65536-point spectrum is 512 KB).
  - I don't know the M2's GPU last-level cache size from this code, so the
    fit has to be measured.
- **(inferred) This is why batching matters on the M2.** The package records
  the M2 GPU as 3.23 TFLOP/s fp32 and 93.4 GB/s (`_peaks.py`, "Apple M2 GPU"),
  a ridge point of about 35 flop/byte.
  - One pair does about 5·n·log₂n flop of FFT and reads 16·n bytes (data plus
    template) if nothing is cached: about 5·log₂n/16 flop/byte, or 5 at
    n=65536.
  - That's memory-bound by about 7× unless cache reuse cuts the bytes per pair.
  - So batching does two things: it amortises per-call overhead, and it lets
    cached rows be reused. The advisor's "more work per memory copy" is this.

### 1d. Apple-specific constraints (read)

- Threadgroup memory is capped at 32 KB on Apple. The preferred builds at 8192
  and 16384 (and 32768, 65536) ask for more, so the `_lds32` portable variants
  are chosen automatically and run slower (`_mtlcompute.py:740-800`, `_stem`).
- Pipeline thread counts on the M2 (`_mtlcompute.py:819-823`): tierb_4096 is
  left alone at 256 threads; tierb_16384_lds32 is forced to 1024 threads
  ("spilling if it must").
- SIMD group width is 32.
- For fused multi-plan batches (`TimeDomainFilterBank`), two submissions per
  command buffer measured best on the M2 (`_mtlcompute.py:415-423`).

### 1e. Band-limited templates (read)

`set_templates` accepts an (ntemplates, n/2) array (`_filter.py:583-592`),
which sets `_bandlimited`. On CPU it builds a half-size plan. I haven't checked
what this does to the output lag grid; not used.

## 2. API for batches, and size limits

### Classes and calls (read)

- `MatchedFilter(n, ndata=1, ntemplates=1, device=None, *, valid=None)`
  (`_filter.py:327`). `device` is `"cpu"`, `"gpu"`, `"gpu:N"` or `"auto"`.
  The default is CPU unless `MF_DEVICE` is set.
- `set_data(spectra, index=None)` and `set_templates(spectra, index=None)`:
  - spectra are (ndata or ntemplates, n) complex64, or one row with `index`;
  - **both copy their input**;
  - templates are conjugated at ingest.
- `run(binsize=None, threshold=0.0, window=None, data=None, templates=None,
  counts=False, raw=False)` (`_filter.py:685`):
  - returns (nd, nt, nbins) records `{index int64, value complex64}`, with
    `index = -1` and `value = 0` for bins not above threshold;
  - `data=(start,count)` and `templates=(start,count)` select a sub-block of
    the all-pairs grid; "the answer is identical to the matching slice of a
    full run";
  - **every combination** of the selected data rows and templates is computed;
  - `raw=True` returns two plain arrays and skips assembling the structured
    array, which matters for small batches;
  - **results reuse internal buffers**, so `.copy()` anything kept past the
    next call.
- **The threshold is one scalar per call** (`_check_threshold`,
  `_filter.py:85`). There's no per-template or per-pair threshold or scale.
  Pairs that need different normalisations need either pre-scaled spectra or a
  common (lowest) threshold followed by an exact cut on the host.
- `run_series(series, starts=None, win_start=None, win_end=None, binsize=None,
  threshold=0.0, templates=None, raw=False, …)` (`_filter.py:1082`;
  C: `ap_mf_run_series`, `matchedfilter.h:131-152`):
  - overlap-save over a complex64 time series with a block layout supplied by
    the caller;
  - the forward FFT of each block is done by the library, on the device for
    GPU (`seriesForward` kernels);
  - each block has its own window [win_start, win_end) and optional
    `bin_phase`, so bins can be put on a global grid;
  - blocks sharing a window are filtered together, up to the plan's `ndata`;
  - output is [block][template][bin], with every block having the same bin
    count.
- `CorrelationFilter(...)`: the same inputs, with full n-lag output and
  `run(data, templates, out, scales, wait)`. **Note `scales`**: a full-output
  run accepts a per-template scale; the peak-only one doesn't.
- `HierarchicalFilter(n, ndata, ntemplates, snr=5.5, fd=1e-2, chain=None,
  device=None, *, valid=None, max_tiers=3, search_window=None)`
  (`_hierarchical.py:155`):
  - `set_reference(power)` takes the expected output power per bin,
    |h|²/S (`time_domain.reference_profile`);
  - `set_coarse_threshold(v)` takes explicit tier thresholds and needs a
    pinned `chain`;
  - `set_first_stage(snr)`;
  - diagnostics: `refine_rate`, `tier_stats`, `stats`, `config`.
- Also `empty_shared(shape, dtype, readback=False)` (shared-memory arrays),
  `wait()`, `set_memory_limits(cache_bytes=, series_bytes=)`,
  `performance_stats`, and `TimeDomainFilterBank` (the library's own
  template-grouping, overlap-save driver for FIR taps).

### Size limits (read)

| mode | CPU | GPU (Metal) |
|---|---|---|
| peak-only `MatchedFilter` | powers of two, 64 to 2²⁰ (`_cpu_max_n`) | 64 to 65536 (`_GPU_SIZES`, `_filter.py:35`) |
| full output `CorrelationFilter` | 1024 to 2²² | 1024 to 2²², two-stage `tc_*` kernels from 2¹⁷ |
| hierarchical | to 2²⁰; up to 8 tiers | ≤ 65536 (GPU flat sizes); **2 tiers max** (`_hierarchical.py:153`) |
| bins per call (GPU) | — | `MAX_BINS = 2048` (`_gpuhost.py:15`); split internally |
| pairs per dispatch (GPU) | — | `max_dispatch_x`; for n ≥ 32768, ≤ 2²⁸/n (`_filter.py:604-611`) |
| threadgroup memory | — | 32 KB on Apple; `_lds32` variants at 8192 to 65536 |

`ndata` and `ntemplates` are arbitrary and need not be powers of two
(`_filter.py:312-313`). The CPU path is single-threaded.

## 3. How hierarchical mode decides to stop, and whether it can miss triggers

### Mechanism (read)

`matchedfilter.h:173-198`, the `_hierarchical.py:117-149` docstring, and the
`gatechain.py` docstring:

- A **gate chain** of coarse bands b₁ < b₂ < … (powers of two, 64 to n/2), then
  a full refine.
  - Tier 0 runs on **every** (block, template) pair.
  - Tier i runs only on tier i−1's survivors.
  - Pairs that pass every tier get the **full n-point correlation and the
    every-bin peak search**.
- A tier at band b computes the **b-point inverse transform of the first b
  spectrum bins** of the product: the low-frequency part of the template-data
  product, on a coarse lag grid spaced n/b samples apart. It compares that
  coarse maximum with the tier's threshold. Below the threshold, the pair is
  dismissed.
- This differs from the advisor's recollection in one way. It isn't a running
  sum accumulated from low frequency up. Each tier is a separate,
  low-resolution transform of a low band, and a later tier redoes a wider band
  from scratch. The idea, that most of the SNR is at low frequency so you can
  dismiss early, is the same.
- The band (what fraction of the spectrum the first tier uses) is **chosen by
  a cost model**, priced on this machine and then measured in a short trial
  (`_choose_chain`, `_hierarchical.py:604`). It's not fixed at "half the band".
  On GPU the first tier can run in **fp16** (`_gpuhier.py`: C16 constants);
  the README says "GPU hierarchical screening can use reduced-precision
  kernels".

### Strict bound or heuristic? **A statistical gate. It can miss above-threshold triggers.** (read)

- The tier thresholds come from `gatemodel.gate_for` / `gatechain`. For an
  injected signal of strength `snr`, the joint distribution of each tier's
  coarse maximum and the full maximum is drawn by Monte Carlo from the
  reference profile (|h|²/S). The threshold is set at the `fd` quantile of the
  coarse maxima of the draws the full filter **keeps** (`gatemodel.py:115-155`).
- So, by construction, about a fraction `fd` of pairs that the full filter
  would report above threshold are dismissed, for signals at strength `snr`.
  Defaults are `snr=5.5`, `fd=1e-2`: by design it loses about 1% of the
  triggers of an SNR-5.5 signal.
- The docs say so: "The gate can omit peaks; `fd` is its modelled
  false-dismissal target at strength `snr`" (`_hierarchical.py:138-140`); "Use
  :class:`MatchedFilter` for an ungated full-bin search."
- The model is validated against the filter to within ratios of 0.88 to 1.10
  across 15 configurations (`gatemodel.py:37-41`). So `fd` itself is only
  accurate to about ±10–20%, and the modelled budget is quoted ±20%.
- Two more ways it can lose triggers:
  - **direct refine** (CPU, only with `set_taps`) searches only near the last
    coarse candidate and can return a smaller local maximum. It's off unless
    taps are set, or disabled with `MF_DIRECT_REFINE=0`;
  - **the reference profile must describe every template in the plan**
    (`set_reference` docstring). One profile for a bank whose |h|²/S shapes
    differ (mtotal 20 to 77 here) mis-calibrates the gate for the templates it
    doesn't describe.
- The only strict bound in the package is the CPU **Q15 screen with its
  margin** (`matchedfilter.h:111-121`: "pass … unless the pair's maximum …
  certainly stays below threshold"). As a hierarchical first tier it runs
  **without** the margin (`matchedfilter.h:219-224`), under the same `fd`
  budget.
- **Implication:**
  - losses depend on the SNR distribution of real triggers;
  - triggers just above threshold, whose high-frequency part pushed them over,
    are the ones most at risk;
  - noise triggers near the 3.8 threshold aren't what `snr=5.5` calibrates for.

  This is why Phase D measures loss directly against peak-only.

### Why the 100× needs a high threshold (measured, C3 triggers, CPU only)

The speed-up rides on the refine fraction, the pairs that pass every tier
(`ap_hmf_stats`: "The ratio is the measured refine rate, which is what the
speedup rides on"). A lower bound on it is the fraction of (template, block)
pairs that contain **at least one** stock trigger above threshold. Every such
pair has to be refined, and the gate also passes some noise-only pairs.

From `~/projects/example-workflow/reference/c3/pycbc/H1-INSPIRAL-1186740108-1800.hdf` (221,314 triggers,
431 templates, 1752 s, max SNR 9.50):

| block | thr 3.8 | 4.0 | 5.0 | 5.5 | 6.0 |
|---|---|---|---|---|---|
| 4 s | 88.4% | 71.5% | 2.34% | 0.30% | 0.096% |
| 8 s | 99.2% | 92.9% | 4.65% | 0.60% | 0.19% |
| 24 s | 100% | 99.97% | 13.4% | 1.79% | 0.58% |

- **At the C3 threshold (3.8), and at the advisor's "typically 4", almost
  every pair must be refined.** Hierarchical can't beat peak-only there and
  will be slower, because the coarse tiers are pure overhead.
- At 5.5, 98–99.7% of pairs could be dismissed, which is where a large
  speed-up is possible.
- The full table at 3.8 to 6.0, compared with the Gaussian estimate, goes in
  `hierarchical-results.md` (Phase D step 1).

## 4. What the code implies about the best segment length and batch size on Apple

- **No Apple-specific rule ships.** `execution-policy.json` has one rule, for
  Haswell (`hierarchical_series`, n=4096, band=1024). There's no M2 cost file
  in the wheel. Choices are made by on-machine timing (autotune) unless
  `MF_AUTOTUNE=0`.
- Hints (read):
  - The README benchmark uses **4096 points, 512 × 512 pairs**, and the M2
    is one of the fleet machines in its comparison (the doc it cites isn't in
    the wheel).
  - `TimeDomainFilterBank` defaults to `max_block_length=8192` and candidate
    FFT lengths of 2048 to 65536 at a 2048 Hz data rate (`time_domain.py:953,
    1034-1039`). It groups templates by length and picks the block length by
    valid fraction, or by cost once a reference is set.
  - The 32 KB threadgroup limit pushes n ≥ 8192 onto the slower `_lds32`
    kernels (§1d). n ≤ 4096 runs the preferred kernels.
  - Hierarchical series grouping: `series_group` 32 for n ≤ 2048, else 16
    (`_hierarchical.py:208`). The C engine halves the group "until its held
    spectra fit 4 MB" (`matchedfilter.h:216-218`), a cache-budget heuristic of
    4 MB per group, on CPU.
- **Our workload constrains it more than the library does.** The C3 bank's
  templates last 0.60 to 6.65 s (median 3.6 s; from `template_duration` in
  the C3 file).
  - The template's time support eats into each block's valid region.
  - At 4096 (2 s) almost no template fits.
  - At 16384 (8 s) the longest leave about 1 s valid.
  - 32768 (16 s) and 65536 (32 s) give about 9 s and about 25 s.
- **Conclusion:** the code doesn't name an optimum for the M2. The trade-off is
  this:
  - shorter n means smaller spectra, more likely to stay cached across a
    batch, and faster kernels at ≤ 4096;
  - shorter n also means a worse valid fraction for long templates.

  Grouping templates by duration, each group with its own n, is the obvious
  design, and it's what `TimeDomainFilterBank` does. The sweep in Phase C
  (n ∈ {16384, 32768, 65536} and batch sizes) has to decide.

## Open or unverified

- M2 GPU last-level cache size and whether a batch's spectra fit: not in this
  code.
- Behaviour of `_bandlimited` templates on the lag grid: not checked.
- How well one reference profile calibrates the gate across this bank's mass
  range: Phase D measures it.
- Nothing in this note was executed on the GPU.
