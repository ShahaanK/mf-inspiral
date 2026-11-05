# Phase C (CPU): mf_fast peak-only results

2026-11-05. matchedfilter 0.1.0a9 (CPU target NEON), PyCBC 2.10.0, env `gw`. C3 job: H1,
GPS 1186740108–1186741908, 4 segments, 431 templates. Design: `design-fast.md`.

**Status: correctness done (check A PASS; B reported). Timing runs not done yet.** The sandbox
blocks `top`, so the required quiet-machine check can't be run from inside Claude. Waiting on
your instruction.

## What was built (uncommitted until the Phase C checkpoint)

- `mf_fast/`:
  - `frontend.py`: PyCBC conditioning from the fixed C3 preset;
  - `templates.py`: tapered, truncated template cache, σ², automatic padding switch;
  - `search.py`: block layout, `run_blocks`, stitching, PyCBC symmetric and findchirp
    clustering;
  - `output.py`, `main.py`.
- `bin/mf_fast`.
- `tests/test_fast_blocks.py`: block layout and stitching against a direct float64 correlation,
  5 cases, including a partial last bin and a short last block. **All pass**, max rel diff
  2.7e-7.
- `tools/`:
  - `ref_full_length.py`: reference A;
  - `cutoff_errors.py`: float64 cut-off error, every template;
  - `physics_report.py`: criterion B;
  - `compare_triggers.py --core`.
- `docs/suspected-bugs.md` #1: `run_blocks` / `run_series` return 1/n of `run()`'s scale, and
  this isn't documented. mf_fast accounts for it.

Outputs are under `~/projects/example-workflow/reference/fast/` (`refA/`, `cpu/`), each with
`run.log` and `timing.json`.

Command (CPU correctness run):

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 MF_AUTOTUNE=0 \
python bin/mf_fast --frame-file .../H-H1_LOSC_CLN_4_V1-1186740069-3584.gwf --channel H1:LOSC-STRAIN \
  --bank .../bank.hdf --gps-start 1186740108 --gps-end 1186741908 --trig-start 1186740140 \
  --trig-end 1186741892 --output .../reference/fast/cpu/H1-MFFAST-1186740108-1800.hdf --device cpu \
  --cutoff-measurements docs/experiments/c_cutoff_errors.json --timing-json .../cpu/timing.json --verbose
```

## 1. Padding margin on the worst cases (your item 1)

`tools/cutoff_errors.py` measures **every** template on all 4 segments, in float64:
- the largest |ΔSNR|/SNR above 3.8 between the full-length filter with the tapered template and
  the same filter with the template cut to its support;
- output in `experiments/c_cutoff_errors.json`.

| padding | max | 99th percentile | median | templates over 5e-6 |
|---|---|---|---|---|
| 2 s | 6.29e-6 | 5.58e-6 | 3.11e-6 | **9** |
| 4 s | 2.79e-6 | 2.58e-6 | 1.81e-6 | 0 |

- **The worst cases are the shortest (high-mass) templates, not the longest.** The correlation
  of log-error with chirp length is −0.80.
- The 9 over budget are rows 1, 13, 16, 20, 41, 61, 75, 101 and 170, with chirp 1.13–2.79 s from
  16 Hz.
- The 5 longest templates (chirp about 12 s) are at 2.3–2.8e-6.
- So the earlier hand-picked six were not the worst cases.
- **Automatic switch:**
  - with `--cutoff-measurements`, templates measured over budget get 4 s. In this run that's
    exactly those 9 (logged in `run.log` and `timing.json`);
  - without it, the cache's data-independent estimate √(σ²_tail/σ²) decides, scaled by
    `CUTOFF_RATIO` = 1.62. That's the largest measured/estimate ratio over all templates; the
    range is 0.88–1.62 and the median 1.03.
  - The estimate rule is conservative: it would switch 177 templates (all short, with spare room
    in a 32 s block). A ratio of 1.25 would already catch all 9.
- Cost of the switch: short templates' support grows from about 5–7 s to 9–11 s, but they still
  fit 22+ s of valid output per 32 s block.

## 2. Check A: short blocks vs full length, same templates (pass/fail at 1e-5): **PASS**

`tools/compare_triggers.py --core refA/H1-REFA-…hdf cpu/H1-MFFAST-…hdf`:

| criterion | result |
|---|---|
| structure | 8 core datasets on both sides; `search/*` identical |
| matching | 221,302 reference, 221,301 new; **221,298 matched**, 4 reference-only, 3 new-only |
| exact columns | `end_time`, `sigmasq`, `template_duration`, `template_hash` identical |
| SNR | max 4.38e-6, 99.9th percentile 2.83e-6, median 4.8e-7: **PASS** (1e-5) |
| coa_phase | max 4.77e-6 rad, 99.9th percentile 2.86e-6: **PASS** |
| unmatched | 7 against a limit of 22, **all explained** (below) |

The 7 unmatched triggers, each checked against the float64 full-length correlation:
- **2 in-bin near-ties (5c):** the same peak one sample apart.
  - template 2677775207278627854, samples 2862590/2862589, SNR 4.5900741/4.5900736;
  - template 7691187381213205096, samples 330566/330565, SNR 3.8522592/3.8522589.
- **Neighbour-bin near-ties (5b), template −3339670447308286505 (row 81):**
  - segment 3, bins 363/364, float64 maxima 4.0037459/4.0037517 (gap 1.4e-6): the reference
    keeps 364 (sample 3503971), mf_fast keeps 363 (sample 3500232);
  - segment 2, bins 276/277, 4.0069613/4.0069745 (gap 3.3e-6): the reference keeps 277
    (sample 2426943). In mf_fast's rounding, 276 is louder, so 277 is dropped; 276 is then
    dropped beside the louder bin 275 (4.2026).
  - Both gaps are well under criterion 5b's 2e-5.

The maximum SNR difference (4.4e-6) is the float64 cut-off error (≤ 5e-6 after the switch) plus
float32 rounding (about 5e-7), as budgeted.

## 3. Criterion B: physics impact against stock C3 (reported; needs advisor sign-off)

`tools/physics_report.py` against `reference/c3/pycbc/H1-INSPIRAL-1186740108-1800.hdf`; output
in `experiments/c_physics_cpu.json`.

**SNR difference on matched triggers** (221,034 matched):

| templates | n | median | 99th percentile | 99.9th percentile | max |
|---|---|---|---|---|---|
| all | 221,034 | 3.1e-6 | 6.4e-4 | 1.1e-3 | **1.9e-3** |
| duration < 2 s | 35,672 | 6.0e-7 | 2.6e-6 | 3.4e-6 | 4.8e-6 |
| 2–4 s | 87,754 | 1.0e-6 | 1.7e-5 | 2.9e-5 | 4.9e-5 |
| 4–6 s | 74,086 | 3.7e-5 | 4.0e-4 | 5.9e-4 | 9.5e-4 |
| 6–7 s | 23,522 | 2.4e-4 | 1.1e-3 | 1.5e-3 | 1.9e-3 |

- The difference grows strongly with template duration.
- **Interpretation (inferred, not separately verified):** this is the stock filter's integration
  of the 1/t tails from the sharp 20 Hz edge (`short-block-check.md` §3). The tapered template
  no longer has them.
  - Long, low-mass templates have most of their power near 20 Hz, so their edge step is
    largest.
  - Short templates match stock at the float32 level.
- σ² changed by at most 8.1e-6 (the 4 Hz high-edge taper).

**Trigger counts** (matched / stock-only / new-only):

| SNR above | stock | mf_fast | matched | stock-only | new-only |
|---|---|---|---|---|---|
| 3.8 | 221,314 | 221,301 | 221,034 | 280 | 267 |
| 5.0 | 4,429 | 4,425 | 4,420 | 9 (all the same peak ±1–2 samples) | 5 (same) |
| 6.0 | 181 | 181 | **181** | 0 | 0 |

The 280 stock-only and 267 new-only at 3.8, by distance to the nearest same-template trigger in
the other file:

| class | stock-only | new-only |
|---|---|---|
| same peak, ≤ 2 samples away | 141 | 141 |
| partner within 2048 samples (same or adjacent bin) | 63 | 63 |
| partner in the neighbouring bin (2048–4096) | 42 | 44 |
| no partner within 2 bins, SNR < 3.81 (threshold band) | 23 | 11 |
| no partner within 2 bins, SNR ≥ 3.81 | 11 | 8 |

- The last row: all are long templates (4.8–6.6 s), SNR ≤ 4.19, with partners 2–3 bins away.
- That's consistent with a cascade of symmetric-rule decisions across neighbouring bins, under
  the up-to-1e-3 SNR change. I haven't traced them individually.

**Against template-bank mismatch** (`experiments/b_bank_mismatch.py`, 200 random signals from
the bank's construction region, plus each template's nearest neighbour):

| PSD | median fitting factor | 10th percentile | worst | median SNR loss | median nearest-neighbour match |
|---|---|---|---|---|---|
| C3 segment 0 (measured) | 0.9950 | 0.9902 | 0.9827 | **0.50%** | 0.9885 |
| aLIGOZeroDetLowPower (design) | 0.9879 | 0.9744 | 0.9619 | 1.21% | 0.9448 |

- **Signal SNR loss from the taper:** ≤ 2.8e-6 (`design-fast.md` §1), about 2000× below the
  typical bank loss.
- **Trigger-level SNR changes:**
  - median 3e-6, about 1600× below the 0.5% bank loss;
  - worst 1.9e-3, about 2.6× below it.
- Above SNR 6 the trigger sets are identical. Above 5, all differences are 1–2-sample shifts of
  the same peak.

## 4. Timing: not done (blocked)

The `top -l 1 -n 10` quiet-machine check failed inside the sandbox:
`(eval):1: operation not permitted: top`. Per instructions, I've stopped rather than check load
another way.

For information only (no quiet check, so **not** a valid timing), from the correctness run's
`timing.json`:
- the search took 6.86 s, of which `run_blocks` was 6.42 s, for 41,224 (template, block) pairs;
- that covers 1.546e9 template × valid output samples (431 × 3,588,096: exactly stock's analysed
  samples, so per-valid-sample ratios equal time ratios);
- the template cache took 34.3 s: generation from 16 Hz plus the cut-off estimate's two extra
  2²⁰ FFTs per template;
- the front end took 7.1 s.

The C3 matched-filter step for comparison: stock 18.62 s, mf_inspiral 7.00 s (cProfile,
`c3-results.md`).

## 5. Other items from this round

- **Refine fraction at 32 s blocks** (your item 4): `hierarchical-results.md` §1.
- **18 Hz and 19 Hz low-edge rows** (your item 2): `design-fast.md` §1, "Alternative start
  frequencies".
- **Bank mismatch** (your item 3): above, and `experiments/b_bank_mismatch.json`.
