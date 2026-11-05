# Experiments behind the fast-path design (Phase B)

All CPU-only, float64 NumPy. The front end is the C3 run's PyCBC conditioning (GPS
1186740108–1186741908, H1, 4 segments, 431-template bank), without PSD variation,
which doesn't affect z. Run each with `~/miniforge3/envs/gw/bin/python SCRIPT OUT`
and the four single-thread variables set. `MPLCONFIGDIR` must point at a writable
directory inside the sandbox.

| file | what it shows | write-up |
|---|---|---|
| `b_trunc.py`, `b_trunc.json` | First scan: an unweighted energy criterion (ε = 1e-6 to 1e-14) needs ±249–256 s of support for every template, so no short block fits. | `../short-block-check.md` |
| `b_diag2.py` | The truncation error in SNR units, for fixed supports −(dur+x)..+x. It falls only as ~1/√support and is still 3e-4 at 32–39 s. | `../short-block-check.md` §3 |
| `b_diag.py` | Fragment: the tail of `b_diag2.py`. `b_diag2.py` is the first 60 lines of `b_trunc.py` (front end) plus this. Kept as run. | — |
| `b_cause.py` | (a) Overlap-save equals the full-length filter with the same truncated template (9e-16). (b) Smoothing the 20 Hz edge, then the high edge, collapses the tail. | `../short-block-check.md` §1, §3 |
| `b_taper.py`, `b_taper.json`, `b_taper.log` | The taper trade-off: PyCBC `fd_taper` variants, padding needed against SNR loss. | `../design-fast.md` |
| `b_blocks.py`, `b_blocks_x*_w*.json` | Valid fraction and cost per valid sample, against block length and duration grouping. | `../design-fast.md` §3 |
| `b_bank_mismatch.py`, `.json`, `.log` | Bank fitting factor (200 random signals) and nearest-neighbour match, C3 and design PSDs. | `../fast-cpu-results.md` §3 |
| `c_cutoff_errors.json`, `.log` | `../../tools/cutoff_errors.py` output: float64 cut-off error of every template, 4 segments, 2 s and 4 s padding. | `../fast-cpu-results.md` §1 |
| `c_physics_cpu.json` | `../../tools/physics_report.py` output, mf_fast CPU against stock C3 (criterion B). | `../fast-cpu-results.md` §3 |
| `d_refine_fraction.py`, `.json` | Refine fraction from the C3 triggers at 32 s blocks, against the Gaussian estimate. | `../hierarchical-results.md` §1 |

These are evidence, not production code. They are kept as run, so their output can be
reproduced.
