# Phase B design: `mf_fast`, batched short-block matched filtering

2026-11-05. Design only; no executable code is written yet. Inputs:
- `matchedfilter-internals.md` (Phase A);
- `short-block-check.md` (why stock templates can't be used in short blocks);
- `experiments/` (`b_taper.py`, `b_blocks.py` and their JSON outputs).

**Status: awaiting approval.** Correctness criterion B (§2) also needs the advisor's sign-off.

## 0. Decisions already taken (from review)

- **Option 2:** smooth the template band edges and use short blocks.
- Correctness is split into **A** (algorithm, held at 1e-5) and **B** (physics against stock C3,
  reported, not pass/fail). This replaces the 1e-5-against-stock criterion **for mf_fast only**;
  mf_inspiral keeps it.
- **Normalisation:** pre-scale the templates per segment (§4).

## 1. Edge smoothing: use PyCBC's standard taper, out of band

### Is there a standard taper? Yes, in PyCBC. None in matchedfilter.

- PyCBC turns frequency-domain approximants into time-domain filters with
  `pycbc.waveform.waveform.get_td_waveform_from_fd` (`waveform.py:786-826`). It:
  - generates the waveform from a **lowered** `f_lower`;
  - sets `f_ref` to the original `f_lower`, which keeps the phase convention;
  - applies `pycbc.waveform.utils.fd_taper(htilde, f_lowered, f_lower, beta=8, side='left')`,
    a half-Kaiser roll-on (`utils.py:397`).
- The same `fd_taper` with `side='right'` is PyCBC's right-edge taper (used in `premerger.py`).
- matchedfilter has no taper: `time_domain.py` takes FIR taps as given.
- So no new taper is invented. Both edges use `fd_taper` (half-Kaiser, β = 8), following PyCBC's
  conventions.
- Note: `fd_taper` needs a double-precision input (its window is float64). FilterBank templates
  are complex64, so they're cast to complex128 first. This is a usage constraint, not a bug.

### Where to put the taper

- **Low edge: out of band, below 20 Hz.** Generate from `20 − w` Hz with `f_ref = 20` Hz, and roll
  on over [20 − w, 20).
  - The data are zero below kmin (D = 0 outside [kmin, kmax)), so this part never enters the
    product.
  - Measured: **every in-band bin is bit-identical to the stock template** (max |ΔH| = 0.0, 6
    templates × 5 variants), so the SNR change from the low taper is exactly zero.
  - Its only cost: the waveform starts at a lower frequency, so the time-domain chirp gets longer.
- **High edge: in band, a right `fd_taper` over the last `w_h` Hz before the template's end** (its
  last nonzero bin, which is Nyquist or the template's f_end). This edge can't be moved out of
  band, so it costs a little SNR (table below).
- **In-band low ramp (20 → 22 Hz):** tested for contrast. It costs about 2e-4 in SNR and saves no
  padding, so it's rejected.

### Trade-off (`experiments/b_taper.json`)

- Padding x is added on each side of the chirp: support = −(chirp + x) .. +x.
- **Algorithm error** = the worst |ΔSNR|/SNR above 3.8 from that truncation, against the
  full-length 2²⁰ filter with the **same** tapered template. Float64, 6 templates spanning the
  bank (0.6–6.6 s), segments 0 and 3.
- The truncation budget is **≤ 5e-6**, which leaves half of the 1e-5 for float32 rounding.
- The high-taper width (4, 16 or 64 Hz) made no difference to the error once a high taper is
  present. Without one, the error stalls at about 1e-4 (rows "high 0" in the JSON). So the table
  uses w_h = 4 Hz.

| low taper (start) | x = 0.5 s | 1 s | 2 s | 4 s | 8 s | x for ≤ 5e-6 | longest chirp | longest support |
|---|---|---|---|---|---|---|---|---|
| none (stock 20 Hz) | 5.4e-3 | 3.4e-3 | 2.2e-3 | 1.3e-3 | 7.7e-4 | none ≤ 8 s | 6.63 s | — |
| 1 Hz (19 Hz) | 2.9e-3 | 6.3e-4 | 8.6e-5 | 8.5e-6 | 2.0e-6 | 8 s | 7.61 s | 23.6 s |
| 2 Hz (18 Hz) | 3.8e-4 | 6.8e-5 | 1.1e-5 | 2.9e-6 | 1.0e-6 | 4 s | 8.80 s | 16.8 s |
| **4 Hz (16 Hz)** | 3.6e-5 | 1.4e-5 | **4.5e-6** | 2.2e-6 | 1.2e-6 | **2 s** | 12.08 s | **16.1 s** |
| 8 Hz (12 Hz) | 1.6e-5 | 7.8e-6 | 4.1e-6 | 2.3e-6 | 1.4e-6 | 2 s | 26.13 s | 30.1 s |
| PyCBC rule (≈ 17 Hz)¹ | 8.1e-5 | 2.7e-5 | 6.8e-6 | 2.5e-6 | 1.1e-6 | 4 s | 10.22 s | 18.2 s |
| in-band 20→22 Hz | 6.5e-4 | 8.7e-5 | 1.6e-5 | 3.5e-6 | 1.7e-6 | 4 s | 6.63 s | 14.6 s, but loses 2e-4 SNR |

¹ PyCBC's own rule: lower f_lower in 1% steps until the duration grows 1.5×
(`waveform.py:786-792`). Here that gives 16.9–17.0 Hz.

**SNR loss for a signal equal to the stock template** (1 − overlap, PSD-weighted over
[20 Hz, Nyquist), **all 431 templates × 4 PSDs**):

| high-edge taper | median | 99th percentile | max |
|---|---|---|---|
| none | 0 | 0 | 0 |
| **4 Hz** | 2.1e-10 | 2.1e-6 | **2.8e-6** |
| 16 Hz | 8.7e-10 | 8.4e-6 | 1.1e-5 |
| 64 Hz | 3.3e-9 | 3.0e-5 | 3.9e-5 |

For scale, the bank was built with minimal match 0.95 (`search/bank.sh`), i.e. up to **5% SNR
loss** from template spacing. The chosen taper's worst case is about 2×10⁴ smaller.

### Alternative start frequencies (added in Phase C at your request; report only)

- Same 4 Hz high taper throughout.
- "Group" means the 159 templates whose support is 12–16.1 s under the chosen taper.
- The error column is the worst of the original 6-template sample (`b_taper.json`).
- The in-band SNR change is **0 for every out-of-band start**: the in-band bins are
  bit-identical to stock.

| low-edge start | padding | cut-off error (6 templates) | longest template chirp | longest support | valid fraction at 65536: group | all |
|---|---|---|---|---|---|---|
| **16 Hz (chosen)** | 2 s | 4.5e-6 | 12.10 s | 16.1 s | 0.55 | 0.65 |
| PyCBC 1% rule (≈ 17 Hz) | 2 s | 6.8e-6 (over) | 10.52 s | 14.5 s | 0.59 | 0.68 |
| PyCBC 1% rule (≈ 17 Hz) | 4 s | 2.5e-6 | 10.52 s | 18.5 s | 0.47 | 0.55 |
| 18 Hz | 2 s | 1.1e-5 (over) | 8.82 s | 12.8 s | 0.63 | 0.70 |
| 18 Hz | 4 s | 2.9e-6 | 8.82 s | 16.8 s | 0.51 | 0.58 |
| 19 Hz | 2 s | 8.6e-5 (over) | 7.63 s | 11.6 s | 0.66 | 0.73 |
| 19 Hz | 8 s | 2.0e-6 | 7.63 s | 23.6 s | 0.29 | 0.35 |

- 18 Hz and 19 Hz shorten the chirp but need more padding to meet the budget. Within budget,
  16 Hz keeps the best valid fraction.
- The full-bank measurement (`fast-cpu-results.md` §1) showed that the worst cut-off errors are
  in short templates, not the six above. Only the 16 Hz / 2 s / 4 s combinations were measured
  on the full bank.
- Kept at 16 Hz, as instructed.

### Choice

- **Low:** `fd_taper` β = 8 over **16–20 Hz**, out of band, generated from 16 Hz with
  `f_ref` = 20 Hz.
- **High:** `fd_taper` β = 8, right side, over the **last 4 Hz**.
- **Padding:** x = **2 s** on each side.
- That gives the shortest worst-case support (16.1 s) and the best valid fraction (§3), and the
  SNR loss is ≤ 2.8e-6 for signals.
- **Risk:** 4.5e-6 is close to the 5e-6 budget, and only 6 templates were tested. Check A in
  Phase C runs over **all 431 templates and all 4 segments**. If any exceeds the budget, the
  fallback is x = 4 s, measured 2.2e-6. That costs about 25% more per valid sample (§3:
  valid fraction 0.65 → 0.52).

## 2. Correctness tests

### A. Algorithm check: pass/fail at 1e-5 (Phase C on CPU, then on the GPU output)

- **Reference A:** the same tapered templates at full length, 2²⁰, filtered the stock way.
  - That's matchedfilter CPU `MatchedFilter(2**20)` with the existing `MFControl` /
    `symmetric_cluster` logic, followed by PyCBC's per-template cross-segment clustering.
  - The PyCBC pieces are reused unchanged.
  - It's built by a small reference driver, `tools/ref_full_length.py`, which shares the
    template cache with mf_fast.
- **Test:** mf_fast's triggers against reference A, using the design-c §7 criteria:
  - matched keys (template_hash, sample);
  - SNR and phase within 1e-5;
  - every unmatched trigger explained as a threshold-band case or near-tie;
  - `end_time`, `sigmasq` and `template_hash` exact.
- Plus a float64 spot check of a few templates, as in `b_taper.py`, to separate truncation error
  from float32 rounding.

### B. Physics impact against stock C3: reported, not pass/fail (**flag for advisor sign-off**)

Against `~/projects/example-workflow/reference/c3/pycbc/H1-INSPIRAL-1186740108-1800.hdf`:

1. The SNR-difference distribution of matched triggers: max, 99.9th percentile and median of
   |ΔSNR|/SNR, and its dependence on template duration.
2. Matched, missed and new trigger counts above 3.8, plus counts above 5 and 6, which matter most
   physically. Each missed or new trigger is classified as threshold band or near-tie, as in
   §7(5).
3. A comparison with template-bank mismatch:
   - the deterministic signal SNR loss (above: ≤ 2.8e-6);
   - the observed trigger-level |ΔSNR|/SNR;
   - both set against 0.95 minimal match (5%) and the bank's actual typical mismatch. I'll
     compute the median nearest-neighbour match for a sample of templates if you want that
     number; it isn't done yet.

Expected (not measured): B is dominated by the stock filter's own integration of the 1/t tails,
which we've removed. In noise those tails contribute about 1e-4 to 1e-3 in SNR
(`short-block-check.md` §3). So B is likely to show triggers that differ at about the 1e-4
level, and a few threshold-band swaps. That is exactly what needs the sign-off.

### Hierarchical loss (Phase D)

- Per threshold (3.8 and 5.5): loss = peak-only triggers above threshold with no hierarchical
  trigger at the same key, divided by the peak-only triggers above threshold.
- Reported binned in SNR (3.8–4, 4–5, 5–6, > 6), and against the gate's modelled `fd`.
- Both runs use the same tapered templates and blocks, so the only difference is the gate.

## 3. Block length, duration grouping, and efficiency per valid output sample

Chosen taper and padding:
- support = chirp(from 16 Hz) + 4 s;
- chirp is 1.13–12.10 s (median 6.65 s) over the 431 templates, so support is 5.1–16.1 s.

Each block yields `valid = floor((n − support)/2048)·2048` searchable lags. Those are whole PyCBC
bins, aligned to `analyze.start`. **All speed figures will be quoted per valid output sample**:
template × valid samples per second of filter time. The stock equivalent is 431 templates × the
analysed samples of the C3 job.

Valid fraction (valid/n) by duration band (`experiments/b_blocks_x2_w4.json` and the inline
table):

| support band | templates | n = 16384 (8 s) | 32768 (16 s) | 65536 (32 s) | 32768 beats 65536 only if t₃₂/t₆₅ < |
|---|---|---|---|---|---|
| 5–6 s | 14 | 0.25 | 0.62 | 0.81 | 0.38 |
| 6–8 s | 73 | 0.04 | 0.52 | 0.76 | 0.34 |
| 8–10 s | 101 | 0 | 0.41 | 0.70 | 0.29 |
| 10–12 s | 84 | 0 | 0.28 | 0.64 | 0.22 |
| 12–14 s | 79 | 0 | 0.15 | 0.58 | 0.13 |
| 14–16.1 s | 80 | 0 | 0.03 | 0.51 | 0.03 |
| all, one n | 431 | 40 fit | 389 fit | **431 fit, mean 0.65** | |

- t₃₂/t₆₅ is the measured time per (template, block) pair at n = 32768 divided by the time at
  65536.
- **With an FFT-work cost model (n·log₂n per pair), n = 65536 is best for every template**, and
  grouping by duration gains nothing (optimal 2- and 3-group splits: cost 1.00 of single n = 65536).
- That's because even the shortest template needs about 5 s of support. A 2× shorter block then
  loses more valid fraction than it saves in work: the break-even ratios in the table are all
  ≤ 0.38, against 0.47 for n·log₂n.
- **Grouping only pays if the M2 is much faster per pair at 32768 than at 65536.** For example,
  that could happen if a 32768 batch stays in cache and a 65536 batch doesn't, or if the 65536
  `_lds32` kernel is disproportionately slow.
- So the Phase C GPU sweep measures t(n) per pair at 32768 and 65536. Duration groups are used
  only where t₃₂/t₆₅ falls below the break-even column. That would plausibly be only the 87
  templates with support ≤ 8 s.
- Also, n = 65536 is the GPU peak-only maximum. Longer blocks (better valid fraction) aren't
  available on the GPU.
- The fallback x = 4 s gives a mean valid fraction of 0.52 at 65536 (`b_blocks_x4_w4.json`).

## 4. Normalisation and threshold: pre-scale per segment (agreed)

- For each 512 s segment, multiply each cached template spectrum by its own
  `4Δf/√σ²(t, PSD_segment)`, together with the block-FFT scale constant (fixed by a unit test).
  The filter output is then complex SNR, and a single exact threshold of 3.8 holds for every pair.
- σ² is PyCBC's `sigmasq` of the **tapered** template, with `f_lower` = 20 Hz, so it integrates
  over the data band. The low taper is out of band, so σ² differs from stock only through the
  4 Hz high taper, which is ≤ 3e-6.
- Each call takes blocks from one segment only. Rescaling is 431 × 65536 complex multiplies per
  segment, done 4 times per job.
- Also required by hierarchical mode, whose gate is calibrated in unit-normalised output units.

## 5. The executable

- **Name:** `mf_fast`: package `mf_fast/` and launcher `bin/mf_fast`, in `~/projects/mf-inspiral`.
  `mf_inspiral` is untouched.
- **CLI** (its own; no pycbc_inspiral option parity):

  ```
  mf_fast --frame-file F --channel H1:LOSC-STRAIN --bank BANK.hdf
          --gps-start S --gps-end E --output OUT.hdf
          [--conditioning c3]                      # fixed C3 conditioning/PSD settings (JSON preset)
          --threshold 3.8 --mode {peak,hier} --device {cpu,gpu}
          [--block-length 65536] [--pad 2.0] [--taper-low 4] [--taper-high 4]
          [--tmpl-batch N] [--data-batch N]
          [--hier-snr 5.5] [--hier-fd 1e-2]
          [--template-cache DIR] [--timing-json T.json] [--verbose]
  ```

  The conditioning preset holds the C3 values (pad 8, high-pass 15, 2048 Hz, 512 s segments,
  32/16 s pads, median PSD 16/8 s × 63, inverse length 16, autogating as C3). There are no per-flag
  overrides at first.
- **Front end (once per job):** PyCBC `strain.from_cli` and `StrainSegments`, then
  `associate_psds_to_segments`, the same as C3. Each segment is overwhitened (s̃/S on
  [kmin, kmax)) and inverse-FFT'd **once at 2²⁰**, giving the complex series that `run_series`
  filters.
- **Template cache (once per bank, reusable across jobs):**
  - generate each template at the 2²⁰ grid from 16 Hz with `f_ref` = 20 Hz;
  - apply both `fd_taper`s, IFFT, and keep lags −(chirp + 2 s)..+2 s;
  - place them in an n-point buffer, FFT, and store complex64 spectra (431 × 512 KB = 221 MB at
    n = 65536);
  - also store chirp, support, σ² inputs and template_hash;
  - optionally written to `--template-cache` (an `.npz`) so repeated runs skip the ~10 s of
    generation;
  - waveform time is reported separately from filter time.
  - A later speed-up (not in the first version): generate directly at the block Δf = 1/32 Hz.
    That has to be verified against this cache with check A first.
- **Loop:**

  ```
  for each segment s (one PSD):
      scale cached spectra by norm(t, s)                    # §4
      blocks: start_b = analyze.start + b*valid - L_group, window [L_group, L_group+valid)
      for each template batch (tmpl-batch, ≤ GPU pair limit / nblocks):
          run_series(series_s, starts, win_start, win_end, binsize=2048, threshold=3.8,
                     templates=batch)
          per (template): concatenate the blocks' bins → symmetric_cluster (reused from
          mf_inspiral/control.py) → peaks
  per template: PyCBC cross-segment clustering (EventManager.cluster_template_events, 2048)
  ```

  - Valid windows are whole bins, so no PyCBC bin is ever split across blocks. The last block's
    window is clipped at `analyze.stop`, giving the same partial last bin as PyCBC.
  - `L_group` is the largest support in the template batch. Batches are formed from templates
    sorted by duration, so a batch's support is close to its members'.
  - The data batch is all blocks of the segment (about 23 at n = 65536), or `--data-batch`
    blocks per call.
- **Batch sizes (to sweep):**
  - template batch ∈ {4, 16, 64, 178}. 178 is about the largest that keeps blocks × templates
    ≤ 4096, the GPU pair limit at 65536. Segments have 18–31 blocks, depending on the batch's
    support.
  - data batch ∈ {4, all}. The advisor's 4 × 4 is included.
  - n ∈ {32768, 65536}, plus 16384 for the 40 templates that fit, as one data point.
- **Memory budget:**
  - about 221 MB of template spectra, plus about 221 MB scaled per segment, plus the library's
    ingest copy of each batch;
  - 4 × 8 MB of segment series;
  - **< 1 GB total**.
- **Output:** HDF5 `H1/{snr, coa_phase, end_time, sigmasq, template_duration, template_hash}`
  and `H1/search/{start_time, end_time}`, compressed like PyCBC's. No chi-squared columns (this
  isn't a workflow drop-in). Plus `--timing-json`: front end, template generation, scaling and
  ingest, filter (device), host clustering, write, and counts (pairs, valid samples, triggers).
- **Comparison tooling:** `tools/compare_triggers.py` gets a `--core` mode that checks only the
  columns above (the full 23-dataset structure check stays the default for mf_inspiral). A new
  `tools/physics_report.py` produces B.

## 6. Baselines and timing (agreed: both CPU and GPU)

- **Stock (C3):** the matched-filter step was 18.62 s for 431 templates × 4 segments: 10.8 ms
  per (template, 512 s segment), with single-thread FFTW at 2²⁰, measured with cProfile in
  `c3-results.md`. Expressed per template × valid output sample over the same analyse region.
- **mf_inspiral CPU (C3):** 7.00 s for the same work.
- **MLX GPU baseline:** the standard FFT approach on the GPU. For each (template, segment): the
  product, an inverse FFT at 2²⁰, |z|² and a per-bin max/threshold, all in MLX at the stock
  layout. Also at n = 65536 blocks, so the gain from matchedfilter's fusion is separated from the
  gain from shorter transforms. A script, run by you outside the sandbox (MLX can't see Metal
  inside it).
- Every timing run: `top -l 1 -n 10` first (load recorded), plus
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 MF_AUTOTUNE=0`.
  The median of 3 runs.
- **Work per valid sample is about the same as stock**, so any speed-up has to come from
  execution, not from doing less arithmetic:
  - stock does log₂(2²⁰) = 20 FFT stages per sample at a valid fraction of 464/512 = 0.91,
    i.e. 22 per valid sample;
  - mf_fast at n = 65536 does 16 stages at a mean valid fraction of 0.65, i.e. 25 per valid
    sample.

  The 8× peak-only target therefore has to come from fusion, the GPU, and batching or cache
  reuse. (The n·log₂n proxy; measured in Phase C.)

## 7. Phase C plan

1. Unit tests (plain python, `tests/`):
   - the taper matches `fd_taper`;
   - out-of-band taper is in-band identical;
   - block layout and bin alignment;
   - the `run_series` lag and window convention against a direct float64 correlation;
   - the norm constant;
   - short block = full length with the same truncated template.
2. `mf_fast --device cpu --mode peak` on the C3 span. Then check A against reference A (all 431
   templates, 4 segments), then B against stock C3. Write `docs/fast-cpu-results.md` and commit.
3. STOP: GPU commands for the correctness run, the MLX baseline and the sweep, with paste-back
   instructions. Then check A on the GPU output, the speed per valid sample, and grouping only if
   t₃₂/t₆₅ is below break-even. Commit.

## Open items

- Typical bank mismatch, for the comparison in B(3): computing it needs a template-match run
  (CPU, a few minutes). Not done.
- x = 2 s margin: settled by check A over the full bank.
- One reference profile per template batch for hierarchical mode: the gate's calibration
  assumption, measured in Phase D.
