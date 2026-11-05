# Phase B pre-check: short blocks vs the 512 s stock correlation

2026-11-05, CPU only, float64 NumPy on the C3 front end:
- PyCBC conditioning, PSDs and FilterBank exactly as in the C3 run;
- GPS 1186740108–1186741908, 4 segments;
- PSD variation omitted, since it does not enter z.

Scripts (session scratchpad, not in the repo): `b_trunc.py`, `b_diag2.py`, `b_cause.py`.

**Result: short blocks do NOT reproduce the stock correlation within 1e-5, for any block
length the GPU peak-only filter supports (≤ 65536 samples = 32 s). Per instruction, I'm
stopping here. The cause is identified below. The tolerance is unchanged.**

## What was compared

- **Truth:** stock `z = IFFT_N(D · conj(H))`, with N = 2²⁰. D is the overwhitened segment
  (s̃/S, after inverse-spectrum truncation), zeroed outside [kmin, kmax). H is the FilterBank
  template.
- **Short-block method:**
  - the overwhitened series `d = ifft(D)` is formed once per 512 s segment, at full
    resolution, as stock does;
  - the template's time-domain response `h = ifft(H)` is kept on lags [−L, +R], i.e. its
    chirp duration plus padding;
  - each n-sample block is correlated circularly, and only lags [L, n − R) are kept
    (overlap-save).
- Errors are in SNR units, `|Δz|·4Δf/√σ²`. "rel" means `|ΔSNR|/SNR` over every sample with
  SNR > 3.8 in the segment's analyze region.

## Findings

### 1. The overlap-save arithmetic is exact. Not wraparound, not block layout.

Template 427, support −(dur+4 s)..+4 s, n = 65536 (17 s valid per block). Blocks against a
full-length filter with the *same* truncated template: **max relative difference 9.2e-16**.

### 2. PSD resolution, PSD interpolation and inverse-spectrum truncation are not involved.

- In this method the data side is whitened once per 512 s segment, with the stock PSD at the
  stock Δf = 1/512 Hz, and inverse-spectrum truncation is applied exactly as stock.
- Every error below comes from comparing the full-length data with the untruncated template
  against the same data with the truncated template.
- So it's **purely the template's time-domain truncation**.

### 3. Cause: the stock template's time-domain response has 1/t tails, from its sharp spectral edges.

- Every FilterBank template starts abruptly at f_lower = 20 Hz (bin 10240).
- Templates either run to the Nyquist bin (e.g. 229 and 427: up to bin 524287, 1024 Hz) or
  are cut at their end frequency (template 101: 524.95 Hz).
- A step in the spectrum gives a 1/t ("sinc") tail in time. The tail energy beyond ±T then
  falls only as about 1/T, and the SNR error about as 1/√T.

Measured truncation error (segment 0, max rel over SNR > 3.8), support −(dur+x)..+x:

| template (dur) | x = 0.25 s | 1 s | 4 s | 16 s (≈ 32–39 s support) |
|---|---|---|---|---|
| 101 (0.60 s) | 8.2e-3 | 3.2e-3 | 9.6e-4 | 3.4e-4 |
| 229 (4.34 s) | 4.0e-3 | 2.4e-3 | 8.8e-4 | 2.8e-4 |
| 427 (6.63 s) | 3.0e-3 | 1.8e-3 | 7.4e-4 | 2.9e-4 |

- Even 32–39 s of support (more than the 32 s GPU maximum) is **about 30× over 1e-5**.
- At the observed ~1/√T rate, 1e-5 would need on the order of 10⁴ s of support. That's longer
  than the 512 s segment, i.e. no truncation at all.
- The first scan (unweighted energy criterion, ε = 1e-6 to 1e-14) said the same thing:
  L ≈ R ≈ 249–256 s for every template.

Confirming the cause (template 427, same supports, smoothing an edge with a short cosine
ramp). **These are diagnostics only. A smoothed template is not the stock template.**

| variant | x = 0.25 s | 1 s | 4 s | 16 s |
|---|---|---|---|---|
| stock | 3.0e-3 | 1.8e-3 | 7.4e-4 | 2.9e-4 |
| low edge smoothed (20→22 Hz) | 3.9e-4 | 2.4e-4 | 1.4e-4 | 8.2e-5 |
| high edge smoothed (1000→1023 Hz) | 3.0e-3 | 1.6e-3 | 7.2e-4 | 2.7e-4 |
| both edges smoothed | 7.9e-5 | 1.8e-5 | **9.5e-7** | 5.6e-8 |

- The 20 Hz start edge dominates.
- The Nyquist edge is the next limit once the low edge is smooth.
- With both smooth, the tail collapses: 4 s of padding is already under 1e-5.

## Consequence for the plan

Exact reproduction of stock (1e-5) with short blocks is impossible with the stock template
definition. The difference is not numerical noise: the stock 2²⁰ filter really does integrate
the template's 1/t tails over the whole 512 s segment.

Options. **Your decision; nothing has been implemented.**

1. **Keep stock templates, keep 1e-5:** no short blocks.
   - GPU peak-only can't do 2²⁰ (its limit is 65536).
   - The only GPU route is `CorrelationFilter` at 2²⁰ (two-stage full output) plus a host
     peak scan: the same work layout as stock, with no fusion of the peak scan.
   - This would test hardware, not the batching idea.
2. **Redefine the search template with smooth band edges** (e.g. cosine ramps at f_lower and
   below Nyquist, widths to be chosen), and use short blocks.
   - Correctness is then checked at 1e-5 against a **full-length 2²⁰ run with the same
     smoothed templates**: CPU, our own float64 or PyCBC-equivalent reference. That keeps the
     tolerance strict for what the fast code computes.
   - Separately, I'd report how much smoothing changes SNR against stock C3. That's a
     physics-level difference, not a filter error, and I haven't measured it yet.
   - Block efficiency then comes from duration + padding, e.g. 6.63 s + 2 × 4 s ≈ 14.6 s of
     support for the longest template.
3. Ask Alex Nitz how `TimeDomainFilterBank` users handle this. It takes FIR taps, which
   implies templates already truncated or windowed in time. **I have not contacted anyone.**

## Independent of that decision: normalisation and threshold (your item 5)

**I'll pre-scale the templates, not threshold at the lowest value and cut on the host.**
- Every `run()` / `run_series()` call takes data blocks from one 512 s segment, so one PSD.
  Before each segment, I'd multiply each cached template spectrum by its own normalisation
  4Δf/√σ²(t, PSD_segment), with σ² from PyCBC. The filter output is then SNR directly, and one
  exact threshold (3.8) is correct for every pair in the call.
- Cost: one complex multiply per template sample per segment, plus re-ingest. That's 4
  rescalings per C3 job, against hundreds of filter calls.
- Why not the lowest threshold with a host cut:
  - It's also exact, but the kernel admits bins above the batch's lowest threshold. Those
    bins are written back and then discarded, which costs readback for templates with large
    σ.
  - More importantly, **hierarchical mode needs pre-scaling.** Its gate thresholds are derived
    from the single run threshold, in the units of a unit-normalised output (`gatemodel.py`
    noise convention: Re and Im each of variance 1). Unscaled templates would put every pair's
    gate in different units.
- Cost to accuracy: multiplying by the norm before the filter rather than after changes
  float32 rounding at about the 1e-7 level, well inside 1e-5. To be checked in Phase C.

## Your item 3 (block efficiency): deferred to the design, because it depends on the support above

- Valid fraction per block = (n − support)/n, rounded down to whole 2048-sample bins.
- Speed will be reported per valid output sample (template × valid seconds per second).
- The duration-grouping table needs the support rule (option 2's padding), so it waits for
  your decision.
