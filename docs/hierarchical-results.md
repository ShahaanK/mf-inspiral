# Hierarchical mode: results

Phase D. GPU results will be added in Phase D; this part is CPU-only, from existing trigger
files.

## 1. Refine fraction at n = 65536 (32 s blocks): estimate before any hierarchical run

Script: `experiments/d_refine_fraction.py`. Output: `experiments/d_refine_fraction.json`.

### Method

- Each template uses its block geometry from the approved design:
  - support = chirp (from 16 Hz) + 2 × 2 s;
  - valid window = whole 2048-sample bins in the 65536-sample block, **15–26 s** (median 21 s).
- The 1752 s trigger span is cut into consecutive valid windows per template.
- A (template, block) pair **must be refined** if at least one stock C3 trigger of that template
  with SNR > ρ lies in its window
  (`~/projects/example-workflow/reference/c3/pycbc/H1-INSPIRAL-1186740108-1800.hdf`).
- **Gaussian estimate** per pair: `1 − exp(−N·exp(−ρ²/2))`, with N = valid samples per block.
  It's weighted over templates by their block counts.

### Refine fraction (pairs that must be refined)

All templates, plus the 10% with the **shortest** valid windows (15–17 s; the longest, low-mass
templates) and the 10% with the **longest** (24–26 s; the short, high-mass templates):

| ρ | measured, all | Gaussian, all | measured, 15–17 s | Gaussian, 15–17 s | measured, 24–26 s | Gaussian, 24–26 s |
|---|---|---|---|---|---|---|
| 3.8 | 99.88% | 100.00% | 99.59% | 100.00% | 99.85% | 100.00% |
| 4.0 | 99.76% | 100.00% | 99.40% | 100.00% | 99.66% | 100.00% |
| 4.5 | 63.2% | 80.5% | 58.1% | 74.0% | 65.1% | 86.7% |
| 5.0 | 11.3% | 14.3% | 9.1% | 11.8% | 12.1% | 17.1% |
| 5.5 | 1.49% | 1.11% | 0.77% | 0.90% | 1.78% | 1.35% |
| 6.0 | 0.48% | 0.063% | 0.059% | 0.051% | 1.07% | 0.077% |

### Reading it

- **At 3.8 and 4.0, essentially every pair must be refined.** The gate can't dismiss anything, so
  hierarchical mode can only add its coarse-tier cost on top of the peak-only cost. Expect a
  **slow-down** at the C3 threshold. That's the explanation for missing ~100× there.
- **From 4.5 to 5.0 the measured fraction is below the Gaussian estimate.** Neighbouring samples
  of a matched-filter output are correlated, so N overstates the number of independent trials,
  and clustering removes some crossings. Both make the estimate conservative there.
- **At 5.5 and especially 6.0, the measured fraction is above Gaussian**: 7.6× at 6.0 overall.
  - The excess sits in the short, high-mass templates: 1.07% measured against 0.077% Gaussian.
  - The long templates match Gaussian: 0.059% against 0.051%.
  - This is the signature of non-Gaussian transients (glitches), which short templates respond
    to. I haven't verified it against a glitch list.
- So even an ideal gate at 5.5 must refine about 1.5% of pairs. A real gate also passes some
  noise-only pairs.
- An upper bound on the speed-up, if the coarse tiers were free:
  - about 1/0.015 ≈ 67× over peak-only at 5.5;
  - about 200× at 6.0, limited by glitches, not Gaussian noise.

  With realistic coarse-tier costs the real figure is lower. It will be measured in Phase D.
- **Caveat:** these are stock-template triggers (no taper), and the stock clustering
  undercounts crossings. The tapered templates' triggers differ at about the 1e-4 level in SNR,
  which doesn't change these fractions materially.
