#!/bin/bash -l
#SBATCH --account=tra26_poliex
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=0:30:00
#SBATCH --job-name=scaling
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# ---------------------------------------------------------------------------
# Problem-size scaling study: GInteractions/s vs N for serial / omp-32 / cuda.
# Grids (same Mandelbrot window, finer sampling -> N grows ~ grid area):
#   S   600x500    steps=200   (N ~  0.56k)
#   M  1200x1000   steps=200   (N =  2231, official input)
#   L  2400x2000   steps=200   (N ~  8.9k)
#   XL 4800x4000   steps=50    (N ~ 35.7k)  -- fewer steps so serial stays feasible
#   XXL 9600x8000  steps=10    (N ~143k)    -- omp/cuda only (serial ~12 min/run)
# GInt/s is a rate (N(N-1)*steps / pure-dynamics-time), so different step
# counts across sizes remain comparable. All runs benchmark mode (`none 0`).
# Ordered cheap->expensive; the two long serial-XL runs go LAST so a 30-min
# timeout can only clip the least critical datapoint.
# ---------------------------------------------------------------------------
set -uo pipefail
source scripts/env.leonardo.sh
BIN=./install/bin
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "############################################################"
echo "# host : $(hostname)   date: $(date)"
echo "# GPU  :"; nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader 2>/dev/null | sed 's/^/#        /'
echo "############################################################"

declare -A INPUT=( [S]=input/scaling/S_600x500.in [M]=input/Particles.in
                   [L]=input/scaling/L_2400x2000.in [XL]=input/scaling/XL_4800x4000.in
                   [XXL]=input/scaling/XXL_9600x8000.in )

run() { # model size rep cmd...
  echo "----- BENCH $1 size=$2 rep=$3 -----"
  shift 3
  "$@"
}

for sz in S M L; do
  in=${INPUT[$sz]}
  echo; echo "=================== SIZE $sz ($in) ==================="
  for r in 1 2;   do run serial $sz $r "$BIN/particles_serial" "$in" none 0; done
  for r in 1 2 3; do OMP_NUM_THREADS=32 run omp32 $sz $r "$BIN/particles_omp" "$in" none 0; done
  for r in 1 2 3; do run cuda   $sz $r "$BIN/particles_cuda"   "$in" none 0; done
  date
done

echo; echo "=================== SIZE XL (${INPUT[XL]}) — omp/cuda first ==================="
for r in 1 2 3; do OMP_NUM_THREADS=32 run omp32 XL $r "$BIN/particles_omp" "${INPUT[XL]}" none 0; done
for r in 1 2 3; do run cuda XL $r "$BIN/particles_cuda" "${INPUT[XL]}" none 0; done
date

echo; echo "=================== SIZE XXL (${INPUT[XXL]}) — omp/cuda only ==================="
for r in 1 2;   do OMP_NUM_THREADS=32 run omp32 XXL $r "$BIN/particles_omp" "${INPUT[XXL]}" none 0; done
for r in 1 2 3; do run cuda XXL $r "$BIN/particles_cuda" "${INPUT[XXL]}" none 0; done
date

echo; echo "=================== SIZE XL — serial (longest, last) ==================="
for r in 1 2;   do run serial XL $r "$BIN/particles_serial" "${INPUT[XL]}" none 0; done

echo; echo "scaling DONE @ $(date)"
