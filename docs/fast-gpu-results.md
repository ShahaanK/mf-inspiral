# Phase C (GPU): mf_fast peak-only, timing session 2026-11-09

Session: `~/benchmark-runs/20261109-182256-fastpath/`, run by you on empire, inside screen,
outside Claude (`tools/bench_session.zsh`).
- Summary: `tools/summarize_session.py` writes `summary.json` in that directory.
- Environment: Apple M2, macOS 26.5.2, Python 3.11.16, pycbc 2.10.0, matchedfilter 0.1.0a9
  (NEON; GPU Apple M2 Metal), MLX 0.32.2. Commit 62f6cd8 (now fa49bea after the rewrite, same
  code).
- Every run used the four single-thread variables and `MF_AUTOTUNE=0`. Three repeats each.

**Quiet checks.** Before most runs, the busiest process was `top` itself (about 9%). The
exceptions:
- `mds_stores` (Spotlight) at 30–115% before 5 runs;
- `launchd` at about 30% before 2 runs;
- `Code Helper` (cdcapano) at 23% before 1 run.

Repeats agree to within 2% in every configuration (e.g. GPU 11.28/11.10/11.10 s; sweep
n65536-tb64-db4 9.22/9.02/9.01 s), so those spikes didn't move the results visibly.

## 1. Correctness: check A on the GPU output: **FAIL** (SNR offset, suspected bug #2)

Against reference A (full length, same tapered templates), with `tools/compare_triggers.py --core`.
Only runs that cover all 431 templates are compared (the n = 65536 runs). The n = 32768 runs are
fit-only, 386 templates.

| run | matched | SNR rel diff median / max | coa_phase max | unmatched (limit 22) | |
|---|---|---|---|---|---|
| CPU, same cache (earlier) | 221,298 | 4.8e-7 / 4.4e-6 | 4.8e-6 | 7 | PASS |
| `fastgpu-load` ×3 (identical) | 221,281 | 1.23e-5 / 2.31e-5 | 7.0e-6 | 26 | FAIL |
| sweep n65536, tb 4 / 16 / 64 / 178 | 221,281–221,283 | 1.23–1.24e-5 / 2.28–2.33e-5 | ≤ 7.6e-6 | 21–27 | FAIL |

Every GPU SNR is low by a constant 1.23e-5 (σ 2.8e-6). The details are in
`docs/suspected-bugs.md` #2.
- The offset doesn't depend on segment, template, SNR or batching.
- The extra unmatched triggers are threshold-band triggers pushed under 3.8, and peaks that moved
  by one sample.
- Correcting the constant would still leave a GPU scatter about 4× the CPU's, with extremes near
  ±1.1e-5, right at the tolerance. So a rescale wouldn't make the GPU path pass cleanly.

I haven't applied a correction.

`tools/gpu_gain_check.py` (synthetic data, CPU vs GPU vs float64, n = 4096–65536) is written but
hasn't been run yet.

## 2. Timing

The stock matched-filter step comes from cProfile (`full_matched_filter_and_cluster_symm`, as in
C3): **18.45 s = 11.93 ns per template × valid sample** (431 × 3,588,096 = 1.546e9). mf_fast and
MLX are timed on the filter step only. Template generation is excluded for all of them (stock's
waveforms are 10.0 s; mf_fast's cache is built once or loaded).

| path | filter s (median of 3) | ns / valid sample | × stock MF step |
|---|---|---|---|
| stock pycbc_inspiral, CPU 1 thread | 18.45 | 11.93 | 1.00 |
| mf_inspiral, CPU 1 thread | 7.03 | 4.54 | 2.63 |
| **mf_fast CPU** (n 65536, tb 64, all blocks) | 6.22 | 4.02 | **2.97** |
| mf_fast GPU, default (tb 64, all blocks) | 11.10 | 7.18 | 1.66 |
| **mf_fast GPU, best of sweep** (n 65536, tb 64, db 4) | 9.02 | 5.83 | **2.05** |
| **MLX, stock layout** (2²⁰-point FFTs, 16 templates/batch) | 3.13 | 2.03 | **5.89** (no phase gather; 5.22× with it, §4.4) |
| MLX, 65536 blocks | 6.59 (§4.4) | 4.26 | 2.80 |

Whole-job wall time (one 1800 s job, 4 segments):

| path | wall s |
|---|---|
| stock | 39.4 / 38.4 (runs 2–3; run 1 was 51.2 s, because Matplotlib built its font cache) |
| mf_inspiral | 27.4 |
| mf_fast CPU, cache built | 37.6 (cache 21.5 s; saved in 0.44 s) |
| mf_fast CPU, cache loaded | 16.5 |
| mf_fast GPU, cache loaded | 21.6 |
| MLX full | 28.5 (includes CPU template generation, untimed in the filter column) |

The front end (PyCBC conditioning and PSDs) is about 7.0 s in every mf_fast run. Cache load is
0.19 s plus σ² 0.60 s.

### Sweep (GPU, filter s, median of 3)

| n | db | tb 4 | tb 16 | tb 64 | tb 178 |
|---|---|---|---|---|---|
| 65536 | 4 | 12.66 | 9.84 | **9.02** | 9.44 |
| 65536 | all | 13.46 | 10.51 | 11.09 | 11.41 |
| 32768* | 4 | 33.11 | 24.24 | 26.87 | 46.29 |
| 32768* | all | 22.14 | **20.81** | 24.70 | 40.62 |

\*The n = 32768 rows aren't like-for-like with 65536:
- **Templates:** they filter 386 templates, not 431. The 45 skipped are the longest, which don't
  fit 16 s blocks. ns/valid sample is per template filtered, so the count itself is normalised
  out. But the templates kept include all the ones that barely fit.
- **Padding:** they use the estimate rule, which puts 174 templates on 4 s padding (the measured
  rule at 65536 puts 9 there). More padding means a shorter valid window and more blocks per
  segment.
- **Net effect:** the 32768 numbers are pessimistic. Their per-pair kernel cost is still
  meaningful, though. Per processed sample (pairs × n), 32768 costs 2.13 ns against 3.45 ns at
  65536, so the 32768 kernel is about 1.6× more efficient per sample. It loses because the
  windows are short. The template support L before lag 0 (taper from 16 Hz) runs 3.4–14.1 s,
  median 8.65 s. At 16 s blocks most templates are left with 1–4 s of valid window.
- **Mixed block lengths:** I projected the cost of choosing, per template, whichever of
  {32768, 65536} is cheaper. I used the measured per-pair costs and the 65536 cache's measured
  padding, so the 32768 padding penalty above doesn't apply. The result is 8.45 s against 8.65 s
  for all at 65536: 130 templates would move to 32768, for a gain of about 2%. Duration grouping
  across block lengths isn't worth it on this bank.

Template batch matters little above 16. Data batch 4 beats "all blocks" at n = 65536 by 7–19%,
and loses at 32768.

## 3. Why the GPU path is slow, and the 8× target

- **The GPU kernel runs far below the M2's roofline.** mf_fast's best GPU configuration:
  - 39,876 pairs at n = 65536 in 9.02 s, 0.226 ms per pair;
  - 5·n·log₂n = 5.2 Mflop per inverse FFT, so **23 GFLOP/s, 0.7% of the 3.23 TFLOP/s peak**;
  - the minimum memory traffic (template plus data, 1 MB per pair) is **4.6 GB/s, 5% of 93 GB/s**.

  It isn't compute-bound or bandwidth-bound. It's even slower per pair than one NEON CPU core
  (0.156 ms/pair).
- **MLX's plain FFT does better per sample.** MLX's unfused 2²⁰-point FFT, with separate multiply,
  |z|² and bin-max passes, does 58 GFLOP/s, about 1.73 ns per processed sample, against
  matchedfilter's 3.45 ns at 65536.
- **A possible reason (inference, not measured):** at n ≥ 8192 matchedfilter uses its portable
  `_lds32` kernels (`docs/matchedfilter-internals.md` §1). Apple's 32 KB of threadgroup memory
  can't hold a 65536-point transform, and one threadgroup per pair limits occupancy. The 32768
  kernel's better per-sample cost fits that pattern. The package benchmark across n (below) would
  show where the GPU kernel's efficiency peaks.
- **This bank needs long blocks.** The support is up to 14 s plus 2–4 s padding. Blocks of 8192 or
  16384 points (4 or 8 s) are where the GPU kernel would likely run well, and they don't fit most
  templates. Using them would mean splitting templates in time: a design change, not tuning.
- **Against the 8× target:**
  - best mf_fast GPU is **2.05×** stock's matched-filter step;
  - mf_fast CPU is **2.97×**;
  - the plain MLX FFT baseline is **5.89×**.

  None reaches 8×. On this workload, the plain MLX FFT at full length is 2.9× faster than the
  fused kernel.
- **A caveat on the MLX baseline:** it doesn't do PyCBC-style clustering. It does the same per-bin
  max and threshold as mf_fast, but its candidates aren't converted to triggers. Its check against
  float64 on the first batch was 3.2e-6.

## 4. Follow-up session 2026-11-09 (`~/benchmark-runs/20261109-194150-followup/`)

Run by you with `followup.zsh`, at working tree d7fefbb plus the uncommitted `mf_fast/main.py`
and `tools/mlx_baseline.py` changes. Every run's log and `runs.txt` record its autotune setting.

### 4.1 The 26 unmatched GPU triggers (check A, autotune-on run; same as the session)

Classified at bin level. For each one I recomputed the float64 full-length correlation for that
template and segment and looked at the neighbouring bins.

| class | count | what happens |
|---|---|---|
| threshold | 13 | reference SNR 3.8000031–3.8000472. The GPU's −1.23e-5 (×3.8 = −4.7e-5) puts them under 3.8. `compare_triggers` labels only 9 as "within threshold band" because its band is narrower than this offset. |
| one-sample swaps | 2 pairs | the bin's best and second-best samples differ by 1.3e-7 and 3.7e-7 relative. MLX flips one of them as well. |
| adjacent-bin swaps | 3 pairs | the next bin's peak is within 5.8e-7 to 1.3e-5 relative, so the GPU keeps that bin's peak instead (848–1686 samples later). The bin after it is quieter, so it survives. |
| lost with no partner | 3 | SNR 3.93/3.93/4.07. The next bin's peak is within 2.5e-8 / 5.0e-6 / 1.5e-6 relative. On the GPU it edges ahead, so this bin fails the symmetric rule. That neighbour then fails too, because the bin after it is louder (SNR 4.13/4.35/4.20). PyCBC's symmetric rule, not findchirp clustering. |

Total 13 + 2×2 + 3×2 + 3 = 26. Apart from the threshold class, these are near-ties whose
order the GPU's scatter decides.

### 4.2 Autotune on vs off (n 65536, 64 templates, 4 blocks; same session)

| | filter s | total s | autotune decisions logged | cost file written |
|---|---|---|---|---|
| `MF_AUTOTUNE=0` | 9.007 | 17.17 | — | — |
| on (`MF_COST_FILE` in `$OUT`, `MF_AUTOTUNE_LOG=1`) | 9.013 | 17.15 | none | none |

Autotune made no choices on this path. Its trials cover CPU block sizes in
`TimeDomainFilterBank`, the Q15 screen, and hierarchical gate chains. GPU peak-only
`run_blocks` at a fixed n and batch has nothing to tune, so `MF_AUTOTUNE=0` cost nothing. Which
Metal kernel ran isn't logged by the package, so I can't name it. Check A is unchanged (identical
output).

### 4.3 matchedfilter's own benchmark (`python -m matchedfilter.benchmark`)

| n | pairs | CPU 1 thread µs/pair | GPU µs/pair | GPU ns / sample | GPU vs CPU |
|---|---|---|---|---|---|
| 4096 | 32768 | 7.19 | 0.80 | 0.20 | 9.0× |
| 8192 | 32768 | 14.78 | 1.88 | 0.23 | 7.9× |
| 16384 | 32768 | 34.57 | 12.71 | 0.78 | 2.7× |
| 32768 | 32768 | 71.68 | 70.27 | 2.14 | 1.0× |
| 65536 | 8192 | 146.91 | 259.50 | 3.96 | 0.6× |

- **The library itself is slow at large n on the M2.** Its own GPU cost per sample rises
  20× from 4096 to 65536. The GPU beats one CPU core only up to 16384.
- **mf_fast gets what the library delivers:** 70 µs/pair at 32768 (library 70.3) and 226 µs/pair
  at 65536 (library 259.5; mf_fast's 4-block batches do slightly better than the benchmark's).
  Our use isn't the bottleneck.
- **Why (inference, not measured):** the cliff starts at 16384 and steepens at 32768, the same
  lengths where Apple's 32 KB threadgroup memory forces the portable `_lds32` kernels, and the
  pair limit 2²⁸/n applies.
- **What this means for 8×:** the library's sweet spot (4096–8192, 0.2 ns per sample) is about
  17× cheaper per sample than 65536. But blocks that short don't fit this bank's templates, whose
  support before lag 0 is 3.4–14.1 s. Reaching it would mean splitting templates in time.

### 4.4 MLX baseline, now with phase and clustering

| | gpu_s (median of 3) | post_s | ns / valid sample | × stock MF step | check A (431 templates) |
|---|---|---|---|---|---|
| full 2²⁰ (stock layout) | 3.534 | 0.040 | 2.29 (2.31 with post) | 5.22 (5.16 with post) | **PASS** |
| 65536 blocks (1 run) | 6.591 | — (no clustering) | 4.26 | 2.80 | not run (timing only) |
| mf_fast GPU, same blocks | 9.007 | 0.13 (stitch + cluster) | 5.82 | 2.05 | FAIL (bug #2) |

- **What gpu_s now covers:** product, inverse FFT, |z|², per-bin max and argmax, the complex
  value at the argmax (gives phase), threshold, and copy to the host. Clustering (the symmetric
  rule and findchirp, exactly as reference A) is `post_s`.
- **Not included in either path:** making and uploading templates.
- **Cost of the additions:** the gather adds 0.40 s over the session's 3.13 s, so 5.89× is now
  5.22× like for like.
- **Check A, MLX full, all 431 templates:** 221,301 of 221,302 matched; SNR max 2.5e-6,
  median 7.9e-7; phase max 7.2e-7; 2 unmatched (one pair, the same one-sample near-tie as on the
  GPU path); **PASS**. Its first-batch float64 check is 3.2e-6 in power.
- **The blocks run:** MLX's plain FFT on the same 32 s blocks (38,724 pairs at 16 templates per
  batch) beats matchedfilter's fused kernel by 1.37×.

### 4.5 Spotlight and other load

`mds_stores` before each follow-up run: 3.3–16.5% (12.6% at the end). `launchd` was 31.5%
before the MLX blocks run. These are single samples taken before each run, not during it.

No timing tracks them:
- autotune off 9.007 s (Spotlight 3.3%) vs on 9.013 s (16.5%);
- MLX full 3.586 / 3.534 / 3.531 s at 3.6 / 8.4 / 16.1%, so the run with the least Spotlight was
  the slowest;
- in the session, the 115% sample before sweep n65536-tb64-db4-1 coincides with that run being
  2% slower than its repeats (9.22 vs 9.02/9.01 s). That's the only possible effect, and it's
  within what I'd call noise.

Separately, `mlx-full-1`'s wall time (41.3 s vs 29.0/27.6 s) includes Matplotlib building its font
cache: the follow-up used a fresh `MPLCONFIGDIR`, as with the session's stock-1. Its gpu_s isn't
affected.
