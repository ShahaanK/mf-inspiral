#!/bin/zsh
# One timing session for the fast path (docs/fast-cpu-results.md, Phase C).
# Run on empire, inside screen, outside Claude:  zsh ~/projects/mf-inspiral/tools/bench_session.zsh
#
# Every run in one session, so all numbers share conditions:
#   1. stock pycbc_inspiral (C3 inputs) x3, plus 1 cProfile run (PSD variation off, as in C3)
#   2. mf_inspiral x3, plus 1 cProfile run
#   3. mf_fast CPU, cache built x3 (the first also saves the cache), then cache loaded x3
#   4. mf_fast GPU, cache loaded x3 (also the GPU correctness run)
#   5. MLX GPU baseline: full 2^20 layout x3, 65536 blocks x3
#   6. GPU sweep: n in {65536, 32768}, template batch {4, 16, 64, 178}, data batch {4, all}, x3
# Before each timed run: a quiet check (top, two samples). Each run's output is tee'd to
# $OUT/<timestamp>-<name>.log; /usr/bin/time -l adds wall/user/sys and peak memory.
# CPU and GPU runs alike use the four single-thread variables (the front end is CPU).

set -u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gw

REPS=${REPS:-3}
OUT=~/benchmark-runs/$(date +%Y%m%d-%H%M%S)-fastpath
mkdir -p $OUT
REPO=~/projects/mf-inspiral
S=~/projects/example-workflow/search
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
export MF_AUTOTUNE=0
export MPLCONFIGDIR=$OUT/mplconfig

stamp() { date +%Y%m%d-%H%M%S }

quiet() {
  top -l 2 -o cpu -n 8 -stats pid,user,cpu,command | tail -9 | tee $OUT/$(stamp)-top-$1.txt
}

# run NAME CMD...  (quiet check, then the command under /usr/bin/time -l, tee'd)
run() {
  local name=$1; shift
  quiet $name
  local log=$OUT/$(stamp)-$name.log
  print -r -- "+ $*" | tee $log
  /usr/bin/time -l "$@" 2>&1 | tee -a $log
  print "exit ${pipestatus[1]}" | tee -a $log
}

{
  date; hostname; sw_vers; sysctl -n machdep.cpu.brand_string
  python -c "import sys, pycbc, matchedfilter as mf; print(sys.version); print('pycbc', pycbc.__version__); print('matchedfilter', mf.__version__, mf.backend()); print(mf.devices())"
  python -c "import mlx.core as mx; print('mlx', mx.__version__, mx.default_device())"
  git -C $REPO log --oneline -1; git -C $REPO status --short
} 2>&1 | tee $OUT/environment.txt

GPS=(--gps-start-time 1186740108 --gps-end-time 1186741908 --trig-start-time 1186740140 --trig-end-time 1186741892)
INSPIRAL_COMMON=(--verbose --processing-scheme cpu:1 --bank-file $S/bank.hdf
  --frame-files $S/H-H1_LOSC_CLN_4_V1-1186740069-3584.gwf --channel-name H1:LOSC-STRAIN $GPS
  --pad-data 8 --strain-high-pass 15 --sample-rate 2048 --segment-length 512
  --segment-start-pad 32 --segment-end-pad 16 --allow-zero-padding --taper-data 1
  --psd-estimation median --psd-segment-length 16 --psd-segment-stride 8 --psd-inverse-length 16
  --psd-num-segments 63
  --autogating-threshold 50 --autogating-cluster 0.5 --autogating-width 0.25
  --autogating-taper 0.25 --autogating-pad 16 --low-frequency-cutoff 20
  --enable-bank-start-frequency --snr-threshold 3.8 --cluster-window 1
  --cluster-function symmetric --filter-inj-only --finalize-events-template-rate 500
  --injection-window 4.5 --injection-filter-rejector-chirp-time-window 5)
PSDVAR=(--psdvar-segment 8 --psdvar-short-segment 0.25 --psdvar-long-segment 512
  --psdvar-psd-duration 8 --psdvar-psd-stride 4 --psdvar-low-freq 20 --psdvar-high-freq 480)

FAST_COMMON=(--frame-file $S/H-H1_LOSC_CLN_4_V1-1186740069-3584.gwf --channel H1:LOSC-STRAIN
  --bank $S/bank.hdf --gps-start 1186740108 --gps-end 1186741908 --trig-start 1186740140
  --trig-end 1186741892 --verbose)
CUTOFF=$REPO/docs/experiments/c_cutoff_errors.json
CACHE65=$OUT/cache-n65536-pad2-measured.npz
CACHE32=$OUT/cache-n32768-pad2-estimate-fitonly.npz

for i in $(seq $REPS); do
  run stock-$i pycbc_inspiral $INSPIRAL_COMMON $PSDVAR --output $OUT/stock-$i.hdf
done
run stock-profile python -m cProfile -o $OUT/stock.prof $(which pycbc_inspiral) $INSPIRAL_COMMON --output $OUT/stock-profile.hdf

for i in $(seq $REPS); do
  run mfinspiral-$i python $REPO/bin/mf_inspiral $INSPIRAL_COMMON $PSDVAR --mf-device cpu --output $OUT/mfinspiral-$i.hdf
done
run mfinspiral-profile python -m cProfile -o $OUT/mfinspiral.prof $REPO/bin/mf_inspiral $INSPIRAL_COMMON --mf-device cpu --output $OUT/mfinspiral-profile.hdf

for i in $(seq $REPS); do
  if [[ $i == 1 ]]; then SAVE=(--save-cache $CACHE65); else SAVE=(); fi
  run fastcpu-build-$i python $REPO/bin/mf_fast $FAST_COMMON --device cpu --cutoff-measurements $CUTOFF $SAVE --output $OUT/fastcpu-build-$i.hdf --timing-json $OUT/fastcpu-build-$i.json
done
for i in $(seq $REPS); do
  run fastcpu-load-$i python $REPO/bin/mf_fast $FAST_COMMON --device cpu --load-cache $CACHE65 --output $OUT/fastcpu-load-$i.hdf --timing-json $OUT/fastcpu-load-$i.json
done

for i in $(seq $REPS); do
  run fastgpu-load-$i python $REPO/bin/mf_fast $FAST_COMMON --device gpu --load-cache $CACHE65 --output $OUT/fastgpu-load-$i.hdf --timing-json $OUT/fastgpu-load-$i.json
done

for i in $(seq $REPS); do
  run mlx-full-$i python $REPO/tools/mlx_baseline.py --mode full --tmpl-batch 16 --output $OUT/mlx-full-$i.json
done
for i in $(seq $REPS); do
  run mlx-blocks-$i python $REPO/tools/mlx_baseline.py --mode blocks --tmpl-batch 16 --cache $CACHE65 --output $OUT/mlx-blocks-$i.json
done

run fastcpu-build-n32768 python $REPO/bin/mf_fast $FAST_COMMON --device cpu --block-length 32768 --fit-only --save-cache $CACHE32 --output $OUT/fastcpu-build-n32768.hdf --timing-json $OUT/fastcpu-build-n32768.json
for n in 65536 32768; do
  if [[ $n == 65536 ]]; then C=$CACHE65; else C=$CACHE32; fi
  for tb in 4 16 64 178; do
    for db in 4 0; do
      for i in $(seq $REPS); do
        name=sweep-n$n-tb$tb-db$db-$i
        run $name python $REPO/bin/mf_fast $FAST_COMMON --device gpu --block-length $n --load-cache $C --tmpl-batch $tb --data-batch $db --output $OUT/$name.hdf --timing-json $OUT/$name.json
      done
    done
  done
done

quiet end
print "SESSION DONE: $OUT" | tee $OUT/DONE.txt
