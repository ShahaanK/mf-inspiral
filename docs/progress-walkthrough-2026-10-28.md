# mf-inspiral progress walkthrough for Prof. Capano

Oct 28, 2026 · @Shahaan Khan

## Summary

The CPU version of the drop-in `pycbc_inspiral` replacement, `mf_inspiral`, reproduces stock `pycbc_inspiral` trigger for trigger, and its matched-filter step is 2.7× faster (measured).

- **Correctness:** on a 4-segment, 1800 s H1 run, 221,312 of 221,314 triggers match at the same sample. SNR agrees to 4.5e-7 relative, phase to 3.6e-7 rad. The other 2 pairs are near-ties one sample apart, from float32 rounding.
- **Speed (single thread, CPU):** matched-filter step 18.62 s (stock) vs 7.00 s (mf), 10.8 vs 4.1 ms per template-segment.
- **New bottleneck:** waveform generation, about 10 s in both, is now 57% of mf's template loop.
- **GPU:** visible outside Claude's sandbox (`gpu:0`, Apple M2, Metal). Not yet run: GPU peak-only mode caps at 2¹⁶ points and segments are 2²⁰.
- **Example workflow:** copied out of the PyCBC source tree, stats step done; generation blocked on `lalapps_inspinj`.
- **Repo:** `~/projects/mf-inspiral`, one local commit, not pushed, no LICENSE yet (licence question below).

**Decisions needed today:** licence (MIT vs GPLv3), how to provide `lalapps_inspinj`, chi-squared columns for workflow use, GPU route, and whether waveform generation is in scope.

## Where everything lives

All work is under `/Users/szkhan` on `empire`, in the conda env `gw`. Nothing outside the home folder was changed.

| Path | What it is | Look at it with |
| --- | --- | --- |
| `~/projects/example-workflow/search/` | Copy of PyCBC's `examples/search`, moved out of the source tree as agreed Oct 13 | `ls -la ~/projects/example-workflow/search` |
| `~/projects/example-workflow/notes/` | Design notes and results for every phase | `ls ~/projects/example-workflow/notes` |
| `~/projects/example-workflow/reference/` | Reference and comparison trigger files (stock runs, `run2/`, `mf-cpu/`, C3 runs) | `ls -R ~/projects/example-workflow/reference` |
| `~/projects/mf-inspiral/` | The new repo: `mf_inspiral` executable, filter class, tests, `tools/compare_triggers.py` | `git -C ~/projects/mf-inspiral log --oneline` |
| `~/projects/pycbc-2.10.0/` | PyCBC 2.10.0 source clone, read-only reference | `git -C ~/projects/pycbc-2.10.0 status` |
| `~/miniforge3/envs/gw/` | Python 3.11, PyCBC 2.10.0, MLX, matchedfilter 0.1.0a9, HTCondor 25.13.2, Pegasus 5.1.3 | `conda list -n gw pycbc` |
| `~/.claude/` | Claude Code config: global `CLAUDE.md`, `settings.json`, guard hook and its log | `ls -la ~/.claude ~/.claude/hooks` |

**Notes files, in the order they were written:**

1. `pycbc-inspiral-map.md`: how `pycbc_inspiral` works (arguments, where the filter runs, output layout, chi-squared dependencies)
2. `reference-run.md`: the exact reference command and its results
3. `design-c.md`: the design and acceptance criteria for `mf_inspiral`
4. `c2-results.md`: single-segment CPU comparison
5. `c3-results.md`: 4-segment CPU comparison and profiles

## Getting in and checking the environment

Every command below only reads; nothing changes. Run them in a zsh session on `empire`.

1. **Log in** (RDS PowerShell):

```powershell
ssh -i "$env:USERPROFILE\OneDrive - Syracuse University\Documents\ssh\empire_ed25519" szkhan@empire.syr.edu
```

2. **Activate the environment and check versions** (`empire`):

```zsh
conda activate gw
python -c "import pycbc; print(pycbc.__version__)"
python -c "import matchedfilter as mf; print(mf.devices())"
```

Expected: `2.10.0`, and a device list with `cpu:0` (NEON) and `gpu:0` (Apple M2, Metal).

3. **Check Condor and Pegasus** (`empire`):

```zsh
condor_status
condor_q
pegasus-version
```

Expected: one 8-core, 24 GB slot (`slot1@empire.syr.edu`), an empty queue, and `5.1.3`. The slot reads `X86_64` because the official macOS HTCondor build runs under Rosetta; jobs still run the `gw` Python natively, and MLX reached the GPU inside a test job on Oct 14.

## Step 1: example workflow

The workflow now runs from its own folder and has passed `get.sh`, `bank.sh` and `stats.sh`; `gen.sh` is blocked because `lalapps_inspinj` isn't installed.

**Show it:**

```zsh
cd ~/projects/example-workflow/search
ls -la
cat master.sh
ls -la stat*.hdf
```

- **Moved out of the source tree** (Oct 28), as agreed on Oct 13: copied, not regenerated, since all paths are relative.
- **`stats.sh`** runs `pycbc_dtphase` four times (HL, LV, HV, HLV) and wrote `statHL.hdf`, `statLV.hdf`, `statHV.hdf`, `statHLV.hdf` in a few minutes.
- **Template bank:** first `bank.sh` run exited without writing `bank.hdf`; the rerun succeeded. `bank.log` and `bank2.log` are both exactly 58,167 bytes, which suggests the first run may have finished too. Cause still unknown.

**The blocker:** `executables.ini` line 15 is `injections = ${which:lalapps_inspinj}`.

```zsh
grep -n inspinj executables.ini
which lalapps_inspinj
```

| Option | Effect | Checked by |
| --- | --- | --- |
| Install `lalapps` into `gw` | Downgrades PyCBC 2.10.0 → 2.8.2 | conda dry run |
| Install into `gw` with `pycbc=2.10.0` pinned | Resolves, but downgrades GSL 2.8 → 2.7 and swaps LAL / LALSimulation builds, which would muddy the reference comparison | conda dry run |
| Separate env `inj` with only `lalapps` 10.1.0, pointed at by absolute path | `gw` untouched; `bin/lalapps_inspinj` confirmed inside the `lalapps` package | conda dry run + archive listing |
| `pycbc_create_injections` | Already installed; needs `injections_minimal.ini` rewritten | not tried |
| Skip injections | Gets `pycbc_inspiral` jobs running; changes the example | not tried |

Nothing has been installed. The proposed option is the separate `inj` env, pending your OK.

## Step 2: SNR-only reference run

Stock `pycbc_inspiral`, run directly without the workflow, gives a bit-for-bit reproducible SNR-only reference: 57,919 triggers on 512 s of H1 data.

**Show it:**

```zsh
cd ~/projects/example-workflow
cat notes/reference-run.md
ls -la reference reference/run2
```

**How it differs from the example's `[inspiral]` settings:**

- `--processing-scheme cpu:1` instead of `mkl`, which can't run on `empire` (MKL is Intel-only). This only changes the FFT backend.
- Chi-squared off (no `--chisq-bins`, `--chisq-snr-threshold`, `--sgchisq-*`) and no `--newsnr-threshold`, which pycbc only applies with chi-squared on. Triggers then carry chisq 0, chisq\_dof 1, sg\_chisq 0.
- Span GPS 1186740108–1186740620 (512 s), the minimum for one filter segment, one PSD estimate and `psdvar-long-segment = 512`. The PSD comes from the same 512 s, so these triggers won't equal a workflow job's.

**Two findings:**

- **`cpu:1` alone isn't single-threaded.** User time was 1.9× real time. Prefixing `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1` brought it to 0.94×; template filtering went from 12.4 s to 15.2 s. Which library was threading is unknown.
- **Reproducible:** two runs gave bit-identical triggers in all 23 datasets except four wall-clock timing values (`search/run_time` and three rate fractions).

## Step 3: mf\_inspiral, the drop-in replacement

`mf_inspiral` swaps only the matched filter: a replacement for PyCBC's `MatchedFilterControl`, with the same method signature, calls `matchedfilter` once per template and segment and then applies PyCBC's own clustering rules. Everything else (data loading, PSD, σ², the HDF writer) is PyCBC's code.

**Show the repo:**

```zsh
cd ~/projects/mf-inspiral
git log --format='%h %an %ad %s' --date=short
git ls-files
git status --short
```

- **CLI:** built from PyCBC's own option-group functions plus the remaining `pycbc_inspiral` option names, so the same command line parses the same way. Options it doesn't implement (chi-squared and other vetoes, keep-loudest, checkpointing, non-symmetric clustering) are refused with an error. No code is copied from `pycbc_inspiral`.
- **Clustering test:** matches PyCBC's `ThresholdCluster` exactly in all 10 cases.
- **Comparison tool:** `tools/compare_triggers.py` matches triggers on (template, sample) and checks the acceptance criteria in `notes/design-c.md`.

**Correctness against stock `pycbc_inspiral`:**

| Run | Span | Triggers | Matched | Max SNR diff (relative) | Max phase diff (rad) |
| --- | --- | --- | --- | --- | --- |
| C2, 1 segment | 512 s | 57,919 | all | 4.5e-7 | 4.8e-7 |
| C3, 4 segments, 7 PSDs | 1800 s | 221,314 | 221,312 | 4.5e-7 | 3.6e-7 |

The 2 unmatched C3 pairs are each the same template one sample apart in the same 1 s bin: an exact tie in one case, a 1.3e-7 difference in the other, with float32 rounding deciding which sample wins. All other columns, including σ², are bit-identical. The acceptance tolerance was 1e-5.

**Timing, C3, single thread, both runs profiled with `cProfile`** (PyCBC unmodified; `--psdvar-*` dropped for profiling only, because NumPy's `ndarray.resize()` fails under `cProfile`):

| Step | stock | mf\_inspiral |
| --- | --- | --- |
| Waveform generation | 9.97 s | 10.08 s |
| σ² | 0.49 s | 0.46 s |
| Matched-filter step | 18.62 s | 7.00 s |
| Per template-segment | 10.8 ms | 4.1 ms |

- Stock's step is mostly the inverse FFT (16.83 s); mf's is mostly `matchedfilter.run` (6.20 s) plus template upload (0.77 s).
- Unprofiled, the whole template loop takes 28.94 s (stock) vs 17.68 s (mf), 1.64×.
- At 4 segments, waveform generation limits the whole-loop gain to about (10 + 18.6) / 10 ≈ 2.8× however fast the filter gets. More segments per job raise that ceiling, because each template is generated once per job.

## Step 4: the GPU and the Condor plan

The Metal GPU is usable outside Claude's sandbox, but `matchedfilter`'s fast peak-only GPU mode can't take the workflow's segment size directly.

```zsh
conda activate gw
python -c "import matchedfilter as mf; print(mf.devices())"
```

- **Sizes:** 2048 Hz × 512 s = 2²⁰ = 1,048,576 points per segment. GPU peak-only mode supports powers of two up to 65,536 (2¹⁶); GPU full-output mode goes to 4,194,304; CPU peak-only goes to exactly 2²⁰.
- **Precision:** `matchedfilter` is single precision only, with its own FFT rather than FFTW. CPU results still match to 4.5e-7, so "identical" is better read as "within 1e-5".
- **Plan, D1 (correctness):** GPU full-output mode, then PyCBC's clustering on the CPU. This proves the GPU path but copies every lag back to memory, so it won't show the fused-kernel speedup.
- **Plan, D2 (speed):** `run_series()` with peak-only output on the GPU, checked against D1.
- **Where it runs:** as HTCondor jobs, with `request_cpus = 1` and the four thread variables. A first tiny job will just print `mf.devices()` to confirm Metal works inside a Condor job. One identical run outside Condor will check that Condor doesn't change the timing.

Not yet tested: the GPU path, injection runs, peak memory and workflow use.

## Step 5: Claude Code on empire

Claude Code (2.1.312) runs on `empire` inside `screen`, as you suggested on Oct 6. Three independent layers keep it to the shared-machine rules, and each has been checked.

| Layer | Covers | Enforces | Verified |
| --- | --- | --- | --- |
| Global `~/.claude/CLAUDE.md` | Claude's own decisions | Hard rules, environment facts, working style | Claude declined `ls /Users` on its own |
| PreToolUse hook `~/.claude/hooks/guard.py` | Every Bash, Read, Edit and Write call, before it runs | Denies: other users' homes (including `ls /Users`), `sudo`, `su`, `brew`, `pkill`/`killall`, killing other users' PIDs, writes outside `~`, `~/.ssh`, conda `base`, edits to its own config. Asks: real package installs, writes into the PyCBC clone, `condor_submit` / Pegasus submits | 53/53 self-tests; live deny logged 11:04:09 |
| Bash sandbox (macOS Seatbelt) | Every Bash command and child process, at the OS level | No reads under `/Users/` except my home; writes only to `~/projects`, `~/benchmark-runs` and the `gw` env; `~/.ssh` unreadable | GPU invisible inside it, as expected |

**Show it:**

```zsh
head -20 ~/.claude/CLAUDE.md
ls -la ~/.claude/hooks
~/miniforge3/bin/python3 ~/.claude/hooks/test_guard.py | tail -3
tail -3 ~/.claude/hooks/guard.log
```

- The hook fails closed: `onFailure: "block"`, and internal errors become denies. It runs on miniforge's base Python, so breaking `gw` can't disable it.
- Condor jobs run outside the sandbox and the hook, which is why every submit asks first.
- Claude is blocked from editing any of these files; I change them by hand.

## Decisions needed

- [ ] **Licence.** Matching `pycbc_inspiral`'s arguments exactly is easiest by copying its argument block, which is GPLv3. Keep MIT and avoid copying (current approach), or switch the repo to GPLv3?
- [ ] **Where the repo lives.** My GitHub account, or a group organisation? Nothing is pushed until this and the licence are settled.
- [ ] **`lalapps_inspinj`.** Separate `inj` env (proposed), rewrite injections for `pycbc_create_injections`, or run the example without injections for now?
- [ ] **Chi-squared columns in the workflow.** With chi-squared off, stock writes chisq\_dof 1, which makes downstream reduced chi-squared 0/0 = NaN and the `traditional_chisq` cut drops every trigger. Fill neutral values (chisq\_dof 16, chisq 30, sg\_chisq 1, so the rankings reduce to SNR), or something else?
- [ ] **Tolerance.** You expected identical CPU output. With single precision and a different FFT, agreement is 4.5e-7. Is "within 1e-5, near-ties listed" acceptable?
- [ ] **GPU route.** Full output first for correctness, then `run_series()` peak-only for speed. Is that what Alex intended, and is it OK to contact him about bugs in the alpha?
- [ ] **Waveform generation.** At 57% of the loop it now caps the gain. In scope, or report the filter-step speedup alone?
- [ ] **Setup sign-off.** Miniforge in `~`, personal HTCondor daemons under my account, and Claude Code with the guard rails above: all OK on `empire`?

## Next steps

1. **D1:** a tiny Condor job confirming Metal is visible inside a job, then the GPU full-output path, compared against the C3 reference.
2. **D2:** `run_series()` peak-only on the GPU, checked against D1, then the GPU benchmark.
3. **Workflow:** unblock `gen.sh` per your injection decision, run it, and test `mf_inspiral` against one of the workflow's own `pycbc_inspiral` job scripts.
4. **Write-up:** a Markdown report of the checks and a speedup plot, for the end-of-semester report.
