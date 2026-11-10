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
| **MLX, stock layout** (2²⁰-point FFTs, 16 templates/batch) | 3.13 | 2.03 | **5.89** |
| MLX, 65536 blocks | — | — | failed (my bug, fixed; to rerun) |

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

## 4. Still to run (GPU, outside Claude)

1. `tools/gpu_gain_check.py`: isolates bug #2.
2. `python -m matchedfilter.benchmark --n 4096 8192 16384 32768 65536 --no-hier`: the package's
   own CPU vs GPU cost per pair, across n.
3. MLX 65536-block baseline: rerun after the fix (`valid` was reused as an accumulator in
   `run_blocks`).
4. One mf_fast GPU run with autotuning on (`MF_AUTOTUNE` unset), to check that `MF_AUTOTUNE=0`
   didn't pin a slow kernel choice.
