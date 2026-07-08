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
#SBATCH --job-name=cpp_all
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# ---------------------------------------------------------------------------
# C++ family: correctness (HDF5 out) + benchmark sweep (none 0, no HDF5).
# One boost/dbg job; cpus-per-task=32 + mem=0 already reserves the whole node,
# so --exclusive is unnecessary (and dbg forbids it).
# ---------------------------------------------------------------------------
set -uo pipefail
source scripts/env.leonardo.sh

IN=./input/Particles.in
BIN=./install/bin
mkdir -p output

# Clean OpenMP pinning for reproducible strong-scaling numbers
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "############################################################"
echo "# host   : $(hostname)"
echo "# date   : $(date)"
echo "# GPU    :"; nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader 2>/dev/null | sed 's/^/#          /'
echo "# CPUs   : allocated=${SLURM_CPUS_PER_TASK:-?}  node cores: $(nproc)"
echo "############################################################"

echo; echo "==================== CORRECTNESS (HDF5, outputEvery=20) ===================="
echo "----- [serial] -> output/Particles_cpp.h5 -----"
$BIN/particles_serial "$IN" output/Particles_cpp.h5
echo "----- [omp, 32 threads] -> output/Particles_omp.h5 -----"
OMP_NUM_THREADS=32 $BIN/particles_omp "$IN" output/Particles_omp.h5
echo "----- [cuda] -> output/Particles_cuda.h5 -----"
$BIN/particles_cuda "$IN" output/Particles_cuda.h5

echo; echo "==================== BENCHMARK (none 0, no HDF5) ===================="

echo ">>>>> SERIAL baseline (x3)"
for r in 1 2 3; do
  echo "----- BENCH serial rep=$r -----"
  $BIN/particles_serial "$IN" none 0
done

echo ">>>>> OpenMP strong scaling {1,2,4,8,16,32} threads (x3 each)"
for t in 1 2 4 8 16 32; do
  export OMP_NUM_THREADS=$t
  for r in 1 2 3; do
    echo "----- BENCH omp threads=$t rep=$r -----"
    $BIN/particles_omp "$IN" none 0
  done
done
unset OMP_NUM_THREADS

echo ">>>>> CUDA (x3)"
for r in 1 2 3; do
  echo "----- BENCH cuda rep=$r -----"
  $BIN/particles_cuda "$IN" none 0
done

echo; echo "cpp_all DONE @ $(date)"
