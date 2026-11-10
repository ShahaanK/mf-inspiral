# Throughput per Mac: jobs per hour with the whole M2 busy

Session `~/benchmark-runs/20261109-232648-throughput`, run overnight 9–10 Nov 2026 on empire, at
commit 5ddb6e7.
- **Machine:** Apple M2 with 4 performance and 4 efficiency cores (`hw.perflevel0/1`) and 24 GB.
- **Script:** `~/benchmark-runs/throughput-session.zsh`; the driver is `tools/throughput.py`.
- **Job:** the C3 job (431 templates, 4 segments of 512 s), with the single-thread variables set
  and `MF_AUTOTUNE=0`.
- **Measurement:** each configuration runs its slots back to back, with a 1-min warm-up and then
  an 8-min window (5 min for S8-drift).
- **Throughput** is computed with fractional credit:
  `jobs/h = 3600/W · Σ overlap(job, window)/duration(job)`.

The MLX jobs use the GPU at batch 8 with the template cache loaded. "mf_fast CPU" is the
matchedfilter CPU path: short blocks and tapered templates, so it is in the check-A class and not
stock-identical. Since 5ddb6e7 its template cache is memory-mapped and shared between concurrent
jobs: the measured footprint is 0.52 GB per job, down from 1.55 GB.

## 1. Results

| config | slots | jobs/h | vs S8 | median job wall (min–max) | CPU-s/job | cores used | output |
|---|---|---|---|---|---|---|---|
| S8 | 8 stock | 385 | 1.00× | 75.2 s (53.5–76.2) | 71.6 | 7.78 | exact |
| I8 | 8 mf_inspiral | 608 | 1.58× | 47.5 s (32.8–48.1) | 45.2 | 7.73 | exact (≤ 1e-5) |
| F8 | 8 mf_fast CPU | 976 | 2.54× | 29.5 s (19.2–30.7) | 27.9 | 7.67 | check-A class |
| G1 | 1 MLX | 257 | 0.67× | 14.0 s (14.0–14.4) | 11.1 | 0.79 | exact (≤ 1e-5) |
| G2 | 2 MLX | 457 | 1.19× | 15.5 s (14.4–17.1) | 12.0 | 1.53 | exact |
| G4 | 4 MLX | 692 | 1.80× | 21.2 s (14.6–23.2) | 14.1 | 2.73 | exact |
| **G8** | 8 MLX | **981** | **2.55×** | 29.2 s (14.6–34.0) | 20.2 | 5.60 | **exact** |
| G1+F7 | 1 MLX + 7 mf_fast CPU | 992 (164 + 828) | 2.58× | MLX 21.8 s, CPU 30.4 s | 17.1 / 29.0 | 0.80 + 6.74 | mixed |
| G2+F6 | 2 MLX + 6 mf_fast CPU | 998 (319 + 680) | 2.60× | MLX 22.5 s, CPU 31.8 s | 17.6 / 30.1 | 1.60 + 5.73 | mixed |
| G4+F4 | 4 MLX + 4 mf_fast CPU | 1011 (584 + 427) | 2.63× | MLX 24.8 s, CPU 33.5 s | 19.3 / 32.4 | 3.20 + 3.88 | mixed |
| S8-drift | 8 stock, 5-min window | 387 | 1.01× | 75.0 s (59.3–75.9) | 71.2 | 7.79 | exact |

**Thermal drift:** S8-drift ran 1 h 45 min after S8 and came out **+0.7%** against it. That is
within the noise, so there was no thermal throttling.

## 2. Correctness under load

**Every job verified bit for bit: 1,158 output files, 0 differing, 0 missing, 0 failed jobs.**
`throughput.py verify` compares every dataset and attribute except PyCBC's run-time performance
fields.

| config | files identical | reference |
|---|---|---|
| S8 | 60 | stock C3 |
| I8 | 92 | mf_inspiral-1 |
| F8 | 148 | mffast-cpu-1 |
| G1 | 39 | gate MLX output |
| G2 | 70 | gate MLX output |
| G4 | 107 | gate MLX output |
| G8 | 148 | gate MLX output |
| G1+F7 | 149 | per type, as above |
| G2+F6 | 151 | per type, as above |
| G4+F4 | 154 | per type, as above |
| S8-drift | 40 | stock C3 |

The gates passed:
- **Gate 1:** the MLX self-test on the GPU.
- **Gate 2:** the gate MLX job against stock C3, all columns: 221,314 of 221,314 matched, SNR max
  2.4e-6, OVERALL PASS.

## 3. Interference flags

No configuration swapped (swap-outs were 0 in every run). The flags were small, and none of them
visibly moved the result:

| config | flag | effect on jobs/h |
|---|---|---|
| G2+F6 | 1,193 page-outs (in the window) | 1,193 × 16 KiB = **19 MB**. Sub-windows flat (1012/994/989/998). None visible. |
| G8 | 398 page-outs (266 in the window) | **6 MB**. Sub-windows flat (973/979/982/989). None visible. |
| G1 | Spotlight (`mds_stores`) at 106% CPU, one sample, 3 s into the window | G1 uses 0.79 cores, leaving 7 idle. Job wall 14.0 s, the same as alone (13.98 s); sub-windows 256/257/257/257. None. |
| G4 | cdcapano's `contactsd` (83%) and `BackgroundShortcuts` (51%); root `mobileassetd` (84%) | Single samples at 129, 476 and 445 s. G4 does drift (sub-windows 757/650/686/673), but G2 drifts the same way with no interference at all (next row), so the drift is not attributable to these processes. |
| G2, G4 | the MLX filter step slows during the run | **G2:** filter time goes from 2.96 to 4.5 s per job, while the front end stays at 7.3 s. **G4:** filter time goes from 3.0 s to 7.5–8.3 s per job. The most likely cause is the processes drifting into phase, so that their GPU steps overlap: they start 5 s apart with almost identical job lengths. This is not verified. It is a real property of few-process GPU sharing, not contamination, and it gives G2 and G4 an uncertainty of about ±5%. |
| others | Spotlight 24–51%; XProtect, trustd and kernel_task 20–56%; page-outs 13–124 | Transient. Every sub-window series is flat to within 1%. |

**Nothing needs rerunning for contamination.** G2 and G4 would need a much longer window to pin
down their phase-averaged value, but neither config decides anything.

## 4. Noise: are F8, G8 and the mixes distinguishable?

**Two uncertainty estimates:**
- **Per-slot spread:** ±0–1 jobs/h per slot in every configuration. macOS rotates processes across
  the P- and E-cores, so every slot gets the same mix. This is therefore **not** a useful noise
  measure.
- **Sub-window spread:** four 2-min sub-windows, standard error of the mean. This is the
  within-run statistical uncertainty:

  | config | SE | relative |
  |---|---|---|
  | F8 | ±2 | 0.2% |
  | G8 | ±3 | 0.3% |
  | G1+F7 | ±1 | 0.1% |
  | G2+F6 | ±5 | 0.5% |
  | G4+F4 | ±6 | 0.6% |

**Run-to-run repeatability:** S8 against S8-drift gives 0.7%. I take **about ±1% (±10 jobs/h)**
as the realistic uncertainty of each figure near 1,000 jobs/h.

**Reading the differences:**
- **F8 (976) and G8 (981) are tied** (0.5%).
- **The mixes are slightly higher:** G1+F7 +1.6%, G2+F6 +2.3%, G4+F4 +3.6% against F8. G4+F4 is
  about 3.5 times the ±1% repeatability above F8 and G8, and the ordering is monotonic in the
  number of MLX slots. That is consistent with a real gain of about 2–4%, but it is small.
- **For any practical purpose, F8, G8 and the three mixes are tied at about 1,000 jobs/h.**

## 5. Scaling against the single-job ceilings

Ceiling = 8 × 3600 / (single-job wall time), which would require E-cores to match P-cores and no
contention. "Efficiency" is measured jobs/h divided by the ceiling.

| config | measured | ceiling | efficiency | CPU-s per job: alone → loaded |
|---|---|---|---|---|
| S8 | 385 | 748 | 51% | 38.5 → 71.6 (×1.86) |
| I8 | 608 | 1,056 | 58% | 27.2 → 45.2 (×1.66) |
| F8 | 976 | 1,770 | 55% | 16.2 → 27.9 (×1.72) |
| G8 | 981 | about 1,090 (the GPU ceiling) | 90% | 11.0 → 20.2 (×1.84) |

**What limits the CPU configurations:**
- **The jobs are not waiting; each CPU-second simply does less work.** They use 7.7–7.8 cores,
  yet need 1.66–1.86 times the CPU time they need alone. The slowdown therefore comes from where
  and how the work runs, not from idle time.
- **E-cores are the larger factor.** With rotation, a fully loaded Mac delivers
  (4 + 4r)/8 single-job core-equivalents, where r is an E-core's throughput relative to a P-core.
  That gives the 50–60% seen if r is small and some shared-resource contention is added on top.
  - Front-end time in the MLX configurations shows the transition. With 2.7 busy cores (G4),
    which could all be on P-cores, the front end is 7.9 s against 7.1 s alone (+11%). With
    5.6 busy cores (G8), which must spill onto E-cores, it is 10.3 s (+45%).
- **Contention for memory bandwidth and the shared L2 is secondary but real.** Stock, whose
  2²⁰-point FFTW transforms stream about 8 MB arrays, degrades most (×1.86). mf_inspiral degrades
  least (×1.66).
- **This data cannot split the loss exactly** between E-core speed, P-core clock under all-core
  load, and contention. That would need per-core counters (`powermetrics`, which needs sudo), or
  a 4-job CPU run for comparison.

**What limits the GPU:**
- **Two processes already share the GPU imperfectly.** G1 → G2 → G4 → G8 gives
  257 → 457 → 692 → 981 jobs/h, and the MLX filter step per job grows from 2.93 s alone to
  about 8 s in G8, because the processes queue for the GPU.
- **In G8 the GPU is close to saturated.** Measured in single-job GPU-seconds (filter + PSD
  variation + upload = 3.28 s), G8's GPU work comes to 981 × 3.28 / 3600 = 0.89 of the GPU.
  Only 5.6 of the 8 cores are busy.
- **The host-side "GPU busy" share in `summary.md` exceeds 100% for G4 and G8.** It counts the
  time each process waits in the GPU queue, so it is only an upper bound, as labelled.

## 6. Stock-identical output only

**G8 against I8: 981 / 608 = 1.61× per Mac** for output that matches stock within 1e-5. Against
stock itself, G8 is 2.55× (S8, 385 jobs/h). For comparison, the single-job figures were 2.75×
(MLX against stock) and 1.41× (mf_inspiral against stock).

**G4+I4 was not measured.** An estimate from the data:
- **MLX part:** 584 jobs/h using 3.20 cores, as in G4+F4.
- **mf_inspiral part:** in I8, mf_inspiral delivers 608 / 7.73 = 78.7 jobs/h per busy core. In
  G4+I4, the four CPU slots would occupy about the 3.88 cores F4 used, giving 3.88 × 78.7 ≈ 305
  jobs/h.
- **Total: G4+I4 ≈ 890 jobs/h, about 9% below G8.** This assumes mf_inspiral meets the same
  contention as F4 did in the mix.

**Why G8 should win:** an MLX job yields 175–183 jobs/h per core it occupies (G8 981/5.60; G4+F4
584/3.20). mf_inspiral yields about 79. So every core given to MLX instead of mf_inspiral helps,
until the GPU saturates. G8 is already at about 89% GPU, with 2.4 cores to spare. A G6+I2-type
mix might add a few per cent (rough guess: 0–5%), which is within ±1% repeatability plus the
estimate's own error.

**Recommendation:** don't run G4+I4, because it is predicted to lose to G8. G6+I2 would be the
only exact mix worth an 11-minute slot, and only if a few per cent matters.

## 7. Where the bottleneck sits, and what would raise it

**The Mac is now roughly balanced at about 1,000 C3 jobs per hour.**
- **G8 is limited by the GPU:** 89% busy, filter time ×2.7 from queueing.
- **The mixes are limited by the CPU:** 7.1–7.5 cores busy, the GPU 15–53% busy.
- **F8 is limited by the CPU.**

**The largest single cost per job is fixed CPU work repeated in every job:**
- Python start-up, about 2 s.
- The front end: frame decoding (4.5 s alone), conditioning and PSD estimation.

Together these are about 9 s alone and about 12–15 s under load. In G8, each MLX job spends
10.3 s in the front end, against 8 s in the filter.

**What would raise throughput:**
1. **Amortise the front end over more templates per job.** A larger sub-bank per job, or one front
   end serving several template sub-banks on the same data. This cuts CPU per template directly,
   and it is what `docs/mlx-results.md` §4 projects: the per-template cost ratio becomes the limit.
2. **Cut GPU time per template** once the CPU is no longer the limit:
   - The full-length 2²⁰-point transform is about 6.8 ms per template at B8.
   - Shorter transforms would need either the matchedfilter GPU path at large n (slow at 65536,
     and suspected bug #2) or time-split templates, which are excluded for now.
   - σ² (1.4 ms per template) could move to the GPU.
3. **Pin down the E-core factor.** If it dominates, the choice of how many jobs per Mac (for
   example 6 instead of 8) could matter; that needs a 4-job CPU run or per-core counters.
