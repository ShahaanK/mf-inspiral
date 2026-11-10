# Route A design: a full-length MLX backend for mf_fast

2026-11-09.

**Status:** approved 2026-11-09, with the decisions below. Implementation is Phase A2.

**Inputs:**
- `fast-gpu-results.md`: why matchedfilter's GPU kernel is slow at the long blocks this bank
  needs, and the MLX baseline's check A pass;
- the two A1 experiments below.

**Scope:**
- **Covered:** route A only. This is matchedfilter's fused kernel replaced by MLX's plain FFT, at
  full segment length, as an mf_fast backend; then the whole job benchmarked end to end.
- **Not covered:** hierarchical mode (Phase D), and splitting templates in time.
- **Unchanged:** mf_fast's CPU path and the existing matchedfilter backend.

## 1. Templates: stock templates match stock C3; they become this backend's default

### The check (double precision, NumPy)

`experiments/a1_stock_templates_f64.py` runs mf_inspiral on the C3 job with one change:
matchedfilter's correlator is replaced by float64 NumPy. Everything else is stock: front end,
**stock untapered templates**, stock σ², stock event manager and output. Per (template, segment):
1. ifft(data · conj(h)) · n in complex128 over the full 2²⁰ points;
2. the per-bin maximum over the analyze window;
3. kept where |z| exceeds the threshold (matchedfilter's `run()` rule);
4. cast to complex64.

Data and templates are the float32 arrays stock uses; only the correlation is in double. This
isolates the template question from MLX's precision.

**Result:** `compare_triggers.py`, all columns, against `reference/c3/pycbc/H1-INSPIRAL-1186740108-1800.hdf`.
Output is in `~/projects/example-workflow/reference/fast/a1/`.

| | |
|---|---|
| triggers | 221,314 reference, 221,314 new; 221,313 matched |
| exact columns | end_time, sigmasq, psd_var_val, template_duration, template_hash and all χ² columns **identical** |
| SNR | max rel diff **4.9e-7** (99.9% 3.5e-7, median 1.1e-7) |
| coa_phase | max 4.8e-7 rad |
| unmatched | 2: one pair, a one-sample near-tie (same template, 1259624 vs 1259625, SNR 4.10948) |
| | **PASS** at 1e-5 |

**Why it works where short blocks didn't:** a full 512 s segment holds the stock template's whole
1/t tail, from the sharp 20 Hz edge and the Nyquist cut-off (`short-block-check.md`). Nothing is
truncated, so a plain FFT is stock's computation.

**Decision:**
- **Default for this backend:** stock templates. Criterion B (physics against stock C3) doesn't
  apply to it, because its correctness test is the 1e-5 comparison against stock C3 itself.
- **Option:** the tapered templates stay available as `--templates tapered`. Their correctness test
  remains check A against reference A, which the MLX baseline already passes.

### MLX's own precision (real MLX, unsandboxed)

`experiments/a1_mlx_probe.py` (`a1_mlx_probe.json`) runs the planned per-segment step at the C3
shape (2²⁰ points, 438 bins of 2048) on synthetic data. It compares bin-peak magnitudes against
float64.

| MLX device | max rel error of the bin peaks |
|---|---|
| CPU | 2.4–2.7e-7 |
| GPU | 2.0–2.3e-6 |

GPU float32 at 2²⁰ points costs about 2e-6, well inside 1e-5. The MLX baseline's check A
(tapered, 431 templates) gave 2.5e-6, consistent with this. The full stock-template check runs
on the GPU in A2 (§6).

## 2. Structure

New module `mf_fast/mlx_backend.py`, selected with `--backend`:

```
mf_fast ... --backend matchedfilter   (default; today's short-block path, --device cpu|gpu)
mf_fast ... --backend mlx [--mlx-device gpu|cpu] [--templates stock|tapered]
            [--tmpl-batch 8] [--save-cache F | --load-cache F]
```

- **`--backend matchedfilter`:** the existing `Search` + `TemplateCache`, unchanged. `--device cpu`
  is the CPU path.
- **`--backend mlx`:** `MLXSearch` + `FullLengthCache` (below). The method comes from
  `tools/mlx_baseline.py`'s full mode.
  - The MLX import is deferred to this backend, so the CPU path never imports MLX. MLX can't
    import inside Claude's sandbox at all.
  - `--mlx-device cpu` runs the same code on MLX's CPU device, for tests.
- **Shared with the existing path:**
  - `FrontEnd`;
  - `symmetric_cluster` (from mf_inspiral);
  - PyCBC's `findchirp_cluster_over_window`;
  - `output.write`;
  - the timing JSON.
- **`tools/mlx_baseline.py`** stays as the standalone baseline already measured. It isn't imported
  by the package.

### The full-length template cache (`FullLengthCache`)

What's stored per template (stock):
- the FilterBank spectrum (complex64, segment Δf = 1/512 Hz), stored only on
  [kmin, min(kmax, last nonzero bin + 1)). The data are zero outside [kmin, kmax), so nothing
  else enters the product. Stock IMRPhenomD templates run from 20 Hz to between 665 Hz and
  Nyquist, so bands run up to 514,048 bins (3.9 MB). The whole bank is about **1.7 GB**;
- `chirp_length` (stock's `template_duration` column), `f_lower`, `end_frequency`, `min_f_lower`,
  `approximant`, the template hash.

Where it lives and how it's read:
- **On disk:** one uncompressed HDF5 dataset, one row per template, plus offsets.
- **Per batch:** it's read row by row with h5py, not memory-mapped, so the process only ever holds
  one batch of templates (16 × 3.9 MB = 63 MB). Memory-mapping would pull the whole 1.7 GB into
  the job's resident memory as it went.

How it's built:
- **Building** (once per bank and segment length, ≈ stock's waveform cost, 10 s) generates with
  stock FilterBank, exactly as stock does.
- **Without `--load-cache`,** templates are generated per batch in memory, as stock does. That's
  slower, but it needs no disk.

σ² is bit-identical to stock:
- For IMRPhenomD, PyCBC's `sigma_cached` computes `sigma_view = |h|²·4Δf` on
  [kmin(f_lower), kmax(end_frequency)), then `sigma_view.inner(1/psd)` in float32.
- The backend rebuilds a PyCBC FrequencySeries from the cached band (zero-padded to flen), sets
  the same attributes, and calls PyCBC's own `sigma_cached`. That's the same code on the same
  inputs.
- The cost is one squared norm per template and one inner product per PSD, the same work stock
  does (0.5 s for the job).

**`--templates tapered`:** the band comes from `templates.tapered_template` instead. σ² comes from
`templates.sigmasq` (mf_fast's definition) and duration from
`get_waveform_filter_length_in_time`, as reference A. Check A applies.

## 3. Batching and memory

### Per batch of B templates (from `a1_mlx_probe.json`, real MLX, peak memory per segment step)

| B | GPU peak for the step | of which transient |
|---|---|---|
| 8 | 0.38 GB | 0.28 GB |
| **16** | **0.77 GB** | **0.61 GB** |
| 32 | 1.50 GB | 1.21 GB |

That's about 38 MB of transient memory per template: the product, the transform, the real and
imaginary parts, |z|², and the reductions, all at 2²⁰ points. The MLX baseline measured
**2.05 ms per pair** at B = 16. The session will sweep B ∈ {8, 16, 32} once (§7) before fixing it.

**Choice at design time: B = 16 by default.** The session sweep (see `docs/mlx-results.md`) found B = 8
faster (filter step 2.93 s against 3.58 s) and lighter (MLX peak 384 MB against 748 MB), so the
default is now **B = 8**.

| item | size |
|---|---|
| segments on the GPU (4 × 514k complex64, band only) | 16 MB |
| one batch of templates (band) | 63 MB |
| transient per segment step (B = 16) | ≈ 0.6 GB |
| front end on the host (stock-like) | ≈ 1.1–1.3 GB |
| cache: one batch read at a time | ≈ 0.07 GB |
| **expected peak footprint** | **≈ 2.0–2.3 GB** |

That's under 10% of empire's 24 GB, and about stock's footprint (1.1–1.3 GB) plus one GPU batch.

### Guards
- `mx.set_cache_limit(512 MB)`: MLX otherwise keeps freed buffers, so its footprint only grows.
- `mx.set_memory_limit(4 GB)`: a ceiling. MLX holds work back rather than exceeding it.
- Both are command-line options with these defaults.

### Recorded on every run
In the timing JSON:
- `mx.get_peak_memory()`;
- `mx.get_cache_memory()` at the end;
- the process's maximum resident set size (`resource.getrusage`).

The session also records `/usr/bin/time -l`'s maximum resident set size and **peak memory
footprint**. The footprint is the one that counts Metal's unified-memory buffers.

## 4. Output and clustering (identical to mf_fast)

Per (template, segment), it's the same chain as the existing path:
1. Bin maxima over the analyze window in 2048-sample bins: max |z|², argmax, and the complex value
   at the argmax. These are computed on the GPU.
2. Threshold: kept where |z| > 3.8 / norm, norm = 4Δf/√σ² (matchedfilter's and mf_inspiral's
   rule). The cut is made on the host from the bin maxima, so σ² never has to be on the GPU.
3. `symmetric_cluster`, on the host.
4. Per template, across segments: `findchirp_cluster_over_window` with a 2048-sample window.
5. `output.write`: mf_fast's HDF layout and core columns (snr, coa_phase, end_time, sigmasq,
   template_duration, template_hash, search start/end).

The 1e-5 comparison against stock C3 is therefore `compare_triggers.py --core`. These columns
don't carry PSD variation or χ², so those fields aren't compared.

A1's NumPy check went through mf_inspiral's event manager, not this chain. A2 re-checks with this
chain's output (§6).

## 5. Timing breakdown (timing JSON, both backends)

The steps are:
- `front_end`;
- `templates`: cache load or build, plus σ²;
- `filter`: the GPU work, synchronised, including copying bin maxima to the host;
- `host_prep`: reading and uploading template bands;
- `cluster`: symmetric rule plus findchirp;
- `write`;
- `total`.

The wall time from `/usr/bin/time` adds interpreter start-up and imports.

**Expected (not measured):**

| step | estimate |
|---|---|
| start-up and imports | about 2 s |
| front end | 7.0 s |
| cache load (1.7 GB, warm page cache) | 0.5–1.5 s |
| σ² | 0.5 s |
| filter | 3.5 s |
| cluster | 0.05 s |
| write | 0.07 s |
| **total** | **about 14–15 s** |

That's about 2.6× stock's 38.4–39.4 s per job. **Once the filter is fast, the front end is
about half the job.** Its biggest part is reading the frame file: 4.5 s of stock's log,
"Reading Frames" to "Highpass Filtering". Speeding up the front end is outside route A as you've
scoped it. I'll report it, not change it.

## 6. Tests (Phase A2, real MLX through unsandboxed runs)

1. **Unit test**, `tests/test_mlx_backend.py`: synthetic data at n = 2¹⁴ and 2¹⁶, run on **MLX's
   CPU device and the GPU**. It checks:
   - bin peaks and indices against a float64 NumPy correlator;
   - band placement (offset bands give the same result as full-length spectra);
   - the threshold and the symmetric-rule output, identical to the NumPy path at the same inputs;
   - the memory guards.
2. **Whole job on MLX's CPU device**, C3 inputs, stock templates: `compare_triggers.py --core`
   against stock C3, at 1e-5.
3. **The same job on the GPU:** a correctness run only. Its timings aren't reported; you run those
   (§7).
4. **The tapered option on the GPU:** check A against reference A.
5. **The matchedfilter backend:** `tests/test_fast_blocks.py` still passes, and a CPU run still
   gives identical output to the last committed one.

Commit at the end of A2, with the repo's identity and no trailers.

## 7. Phase A3 session (`~/benchmark-runs/mlx-session.zsh`, run by you; shown before running)

- **Gate block first:** an MLX CPU-device and GPU self-test (the unit test). On any failure the
  script stops immediately and prints why, before any timing.
- **Correctness:**
  - the MLX backend, stock templates, against stock C3 (`--core`, 1e-5);
  - the tapered option against reference A (check A).
- **Whole job, 3 repeats each, quiet check before each:**
  - stock pycbc_inspiral (C3 options, PSD variation on, as in C3);
  - mf_inspiral;
  - mf_fast CPU (matchedfilter backend, cache loaded);
  - mf_fast MLX (stock templates, cache loaded);
  - mf_fast with the matchedfilter GPU backend (cache loaded).

  Each run records the per-step breakdown from §5 and both peak-memory figures.
- **Once:** MLX cache build time (for the cost per job); a template batch sweep B ∈ {8, 16, 32}.
- **Environment:** a fresh `MPLCONFIGDIR` is warmed by a throwaway import before the first timed
  run, so no run pays for Matplotlib building its font cache.

## Decisions (A1 review)

1. **Cache on disk, under `~/benchmark-runs/`, kept out of git.** The session times the MLX whole
   job twice: with the cache loaded, and with templates regenerated every job, as stock does.
2. **Columns: stock's full SNR-only output,** so the comparison with stock C3 is like for like.
   - `psd_var_val` is computed with PyCBC's own PSD-variation code and C3's options.
   - Stock's χ² placeholder values are filled.
   - The stock C3 comparison then uses all columns, not `--core`.
3. **Batch size: B = 16 by default,** with the 8/16/32 sweep in the session.
4. **The checkpoint adds a projection** of whole-job speed-up against templates per job (431,
   2,000, 10,000, 50,000), from the measured front-end, per-template and fixed costs. It also
   says whether the first run's frame read was cold.

## A2 results (2026-11-09; real MLX, run unsandboxed under your GPU rule)

Built:
- `mf_fast/mlx_backend.py`: `TemplateSource`, `MLXSearch`, `segment_peaks`, `threshold_bins`.
- `--backend mlx` and its options in `mf_fast/main.py`.
- `FrontEnd`:
  - keeps the conditioned strain;
  - carries C3's PSD-variation and event-manager options;
  - skips the block series when the MLX backend doesn't need them.
- `tests/test_mlx_backend.py`.

The output goes through PyCBC's EventManager exactly as mf_inspiral's does, so the file is stock's
SNR-only layout.

| test | result |
|---|---|
| `tests/test_mlx_backend.py`, MLX CPU device and GPU | 4 cases: n 2¹⁴ and 2¹⁶, a partial last bin, 1024 and 2048 bins, templates on offset sub-bands, 10–59 bins above threshold. Indices identical to matchedfilter's `run()`. Values within 1.9e-7 (CPU) / 1.1e-6 (GPU) of `run()`, and 1.9e-7 / 1.2e-6 of float64. Memory guards OK. **PASS** |
| whole C3 job, MLX CPU device, stock templates, cache built | against stock C3, **all columns**: structure, gating, attributes and every exact column (end_time, sigmasq, psd_var_val, template_duration, template_hash, all χ²) **identical**. SNR max 4.8e-7. 4 unmatched: two one-sample near-ties. **PASS** |
| same job, **GPU**, cache loaded | all columns identical, **221,314 of 221,314 matched**, SNR max 2.4e-6, phase max 9.5e-7, 0 unmatched. **PASS**. σ² rebuilt from the cache is bit-identical to stock's. |
| tapered option, GPU | check A against reference A (`--core`): SNR max 2.4e-6, 2 unmatched (one known one-sample near-tie). **PASS** |
| matchedfilter CPU backend (regression) | output **bit-identical**, every dataset, to the 9 Nov session's `fastcpu-load-1.hdf` (same cache, same options). `tests/test_fast_blocks.py` all OK. **Note:** the 5 Nov `reference/fast/cpu-load` file differs by up to 2.4e-6 in SNR. That predates f6c2fe6, whose batching changes moved the float32 rounding, so it isn't a valid baseline for this regression. |

**Memory** (timing JSON, correctness runs):

| run | MLX peak | max RSS |
|---|---|---|
| GPU, stock, cache loaded | 748 MB | 1.42 GB |
| GPU, tapered | 748 MB | 1.09 GB |
| MLX CPU device | 620 MB | 2.33 GB |

On the MLX CPU device, MLX's buffers are ordinary process memory. MLX's buffer cache ended at
519 MiB against the 512 MiB limit, so the limit is approximate.

**Cache:** `~/benchmark-runs/cache/mlx-stock-c3.h5`, 1.77 GB, outside the repo.
