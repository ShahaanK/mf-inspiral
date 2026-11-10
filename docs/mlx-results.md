# Route A results: full-length MLX backend, whole-job timings (Phase A checkpoint)

Session `~/benchmark-runs/20261109-223435-mlx`, run 9 Nov 2026 on empire (Apple M2, 24 GB) from
`~/benchmark-runs/mlx-session.zsh`, at commit 36062a8.
- **Job:** the C3 span (GPS 1186740108–1186741908, 431 templates, 4 segments of 512 s).
- **Settings:** single-threaded (`OMP/OPENBLAS/VECLIB/MKL_NUM_THREADS=1`), `MF_AUTOTUNE=0`.
- **Load:** a `top` snapshot was taken before every run.
- **Summary:** produced by `summarise.py` in that directory; its output is `summary.json` there.

The design is in `docs/design-mlx-backend.md`. After this session the MLX default batch was changed
from 16 to **8** (§2).

## 1. Correctness

| check | result |
|---|---|
| Gate 1: `tests/test_mlx_backend.py --device both` | all OK. GPU values within 1.2e-6 of float64; indices identical to matchedfilter's `run()`. |
| Gate 2: MLX backend, stock templates, cache loaded, vs stock C3, **all columns** | **PASS**. 221,314 of 221,314 matched, 0 unmatched. Every exact column identical (end_time, sigmasq, psd_var_val, template_duration, template_hash, all χ² placeholders, search start/end, attributes). SNR max 2.44e-6, coa_phase max 9.5e-7 rad. |
| Check A: tapered option vs reference A | **PASS**. SNR max 2.44e-6; 2 unmatched (one near-tie that moved by one sample). |
| Timed MLX outputs (load ×3, regenerate ×3, batch 8 ×3, batch 32 ×3, cache build) | All **bit-identical** to the gate-2 output. |
| stock-2 and stock-3 | Identical to stock-1, which is identical to the C3 reference. |
| mf_inspiral-1 vs stock C3 | PASS. SNR max 4.5e-7; 2 one-sample near-ties. |
| matchedfilter GPU backend | 221,286 triggers. It still fails check A: the GPU gain loss, `docs/suspected-bugs.md` #2. Timed for reference only. |

## 2. Whole-job time

Wall time is `/usr/bin/time` real, the mean of 3 runs. The spread was 0.08 s or less in every
configuration.

| path | wall time | vs stock |
|---|---|---|
| stock pycbc_inspiral | 38.46 s | 1.00× |
| mf_inspiral | 27.27 s | 1.41× |
| mf_fast CPU (matchedfilter, short blocks, tapered templates, cache) | 16.27 s | 2.36× |
| mf_fast matchedfilter GPU (fails check A) | 19.21 s | 2.00× |
| MLX, templates regenerated every job (as stock does), B = 16 | 25.00 s | 1.54× |
| MLX, cache loaded, B = 16 | 14.72 s | 2.61× |
| **MLX, cache loaded, B = 8 (new default)** | **13.98 s** | **2.75×** |
| MLX, cache loaded, B = 32 | 15.01 s | 2.56× |
| MLX, building the cache (once) | 26.26 s | — |

**Batch sweep (cache loaded):**

| B | MLX filter step | MLX peak memory | max RSS |
|---|---|---|---|
| 8 | 2.93 s | 384 MB | 1.40 GB |
| 16 | 3.58 s | 748 MB | 1.39 GB |
| 32 | 3.70 s | 1,477 MB | 1.90 GB |

B = 8 is both the fastest and the lightest, so it is now the default. The cause of the speed
difference was not investigated.

**Filter step alone:** 18.87 s for stock against 3.58 s for MLX at B = 16 (5.3×) and 2.93 s at
B = 8 (6.4×). Phase C's standalone MLX baseline measured 5.22×.

## 3. Where the time goes (seconds, mean of 3)

**Sources:**
- **stock and mf_inspiral:** log timestamps. Their cProfile runs both failed inside PyCBC's
  `calc_filt_psd_variation` (`filt.resize`: "cannot resize an array that references or is
  referenced by another array"). The normal runs are unaffected, and the cause under the profiler
  is not verified.
- **mf_fast:** the timing JSON.
- **Start-up:** wall time minus the in-process span.

| step | stock | mf_inspiral | MLX cache, B16 | MLX cache, B8 | MLX regenerate, B16 |
|---|---|---|---|---|---|
| Python start-up and imports | 2.26 | 2.35 | 2.03 | 2.03 | 2.03 |
| front end: frame read | 4.54 | 4.54 | 4.54 | 4.53 | 4.53 |
| front end: everything else | 2.41 | 2.41 | 2.35 | 2.34 | 2.35 |
| PSD variation and data upload | (in front end) | (in front end) | 0.38 | 0.35 | 0.38 |
| templates (generate or read, plus σ²) | 9.85 | 10.66 | 0.95 (read 0.36, σ² 0.60) | 0.96 | 11.16 (generate 10.79, σ² 0.38) |
| filter, including thresholding | 18.87 | 6.85 | 3.58 | 2.93 | 3.67 |
| clustering | (inside filter) | (inside filter) | 0.13 | 0.12 | 0.14 |
| output | 0.54 | 0.45 | 0.46 | 0.45 | 0.47 |
| unattributed | 0 | 0 | 0.30 | 0.28 | 0.27 |
| **total** | **38.46** | **27.27** | **14.72** | **13.98** | **25.00** |

- **Fixed costs dominate the MLX job.** About 9.5 s of the 14.0 s (68%) is start-up, the front end
  and PSD variation. No filter backend changes those. The frame read alone takes 4.5 s.
- **Regenerating templates removes most of the gain.** Generation costs about 25 ms per template,
  more than three times the MLX filter cost per template.
- **Peak memory, from `/usr/bin/time`:**

  | path | max RSS | peak footprint |
  |---|---|---|
  | stock | 1,133 MB | 455 MB |
  | mf_inspiral | 1,444 MB | 718 MB |
  | MLX B8 | 1,403 MB | 980 MB |
  | MLX B16 | 1,390 MB | 1,441 MB |
  | MLX B32 | 1,896 MB | 2,050 MB |
  | mf_fast CPU | 2,426 MB | 1,546 MB |

**Frame read, cold or warm.** Every read was warm, including the first run of the session.
- The read took 4.53–4.56 s in all 27 runs.
- `/usr/bin/time` recorded **0 block input operations** in every run, so the frame file was
  already in the page cache.
- The 4.5 s is therefore decoding work, not disk time.
- A cold read could not be forced (that needs sudo), so its cost is unmeasured.

## 4. Projection against templates per job

Fitted from the means above, at N₀ = 431:

```
t_tmpl      = (template step + filter + clustering + output) / 431
T_fixed     = measured wall time − 431 · t_tmpl
              (start-up, front end, PSD variation, upload, unattributed)
T_job(N)    = T_fixed + N · t_tmpl
S(N)        = T_stock(N) / T_path(N)   →  t_tmpl,stock / t_tmpl,path   as N → ∞
```

| path | T_fixed | t_tmpl | N = 431 | 2,000 | 10,000 | 50,000 | limit |
|---|---|---|---|---|---|---|---|
| stock | 9.21 s | 67.9 ms | 38 s | 145 s | 688 s | 3,403 s | — |
| mf_inspiral | 9.31 s | 41.7 ms | 1.41× | 1.56× | 1.61× | 1.63× | 1.63× |
| mf_fast CPU | 8.98 s | 16.9 ms | 2.36× | 3.39× | 3.86× | 3.98× | 4.01× |
| MLX, regenerate, B16 | 9.56 s | 35.8 ms | 1.54× | 1.78× | 1.87× | 1.89× | 1.89× |
| MLX, cache, B16 | 9.60 s | 11.9 ms | 2.61× | 4.34× | 5.35× | 5.63× | 5.71× |
| **MLX, cache, B8** | 9.53 s | 10.3 ms | **2.75×** | **4.80×** | **6.10×** | **6.47×** | 6.57× |

Worked example: S(10,000) for MLX with the cache at B8 is
(9.21 + 0.0679 · 10,000) / (9.53 + 0.0103 · 10,000) = 6.10×.

**Why it levels off.** The speed-up approaches the per-template ratio. That limit is slightly above
the filter-only 6.4× at B8, because stock's per-template cost also includes waveform generation
(22.9 ms of its 67.9 ms).

**MLX per-template cost at B8 (10.3 ms):**

| part | ms per template |
|---|---|
| filter | 6.8 |
| σ² | 1.4 |
| output | 1.0 |
| cache read | 0.8 |
| clustering | 0.3 |

Even with unlimited templates this route stays below the 8× target unless σ², output and the
filter itself also get cheaper.

**Caveats:**
- **The cache stops fitting in memory.** It holds about 4.1 MB per template (1.77 GB for 431):
  8 GB at 2,000 templates, 41 GB at 10,000 and 205 GB at 50,000. Above roughly 2,000 templates it
  no longer fits in the 24 GB of RAM. The 0.83 ms/template read here came from the page cache; the
  SSD read rate is unmeasured, and so is the disk space needed.
- **It assumes constant per-template cost and constant triggers per template.** The full-length FFT
  cost does not depend on template duration. Generation, σ² and output volume could differ for
  another bank.
- **It is a one-core comparison.** Every figure here is one job on one core (or one job on the GPU)
  against one stock job on one core. Throughput per machine, with one job per core, is not measured
  here.
