#!/bin/bash -l
#SBATCH --account=tra26_poliex
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=0:30:00
#SBATCH --job-name=mpi_all
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# ---------------------------------------------------------------------------
# MPI port, single node (course MPI day):
#   1. correctness: 8 ranks with HDF5 -> Particles_mpi.h5, strict-validated
#      inline against the serial reference (expect bitwise).
#   2. strong scaling: ranks {1,2,4,8,16,32} on the official input, x3.
#   3. weak scaling (constant work per rank, N ∝ sqrt(p)):
#      p=1 -> official (N~2231), p=4 -> W2 (N~4462), p=16 -> L (N~8924), x2.
# ---------------------------------------------------------------------------
set -uo pipefail
source scripts/env.leonardo.sh
BIN=./install/bin/particles_mpi
IN=./input/Particles.in
mkdir -p output

echo "############################################################"
echo "# host : $(hostname)   date: $(date)"
echo "# mpi  : $(mpirun --version 2>/dev/null | head -1)"
echo "############################################################"

echo; echo "==================== CORRECTNESS (8 ranks, HDF5) ===================="
srun -n 8 $BIN "$IN" output/Particles_mpi.h5
echo "----- strict validation vs serial reference (rtol=1e-12) -----"
python tools/validate_particles_h5.py output/Particles_cpp.h5 output/Particles_mpi.h5 \
       --rtol=1e-12 --atol=1e-14 | tail -12

echo; echo "==================== STRONG SCALING (official input) ===================="
for P in 1 2 4 8 16 32; do
  for r in 1 2 3; do
    echo "----- BENCH mpi ranks=$P size=M rep=$r -----"
    srun -n $P $BIN "$IN" none 0
  done
done

echo; echo "==================== WEAK SCALING (N ~ sqrt(p) * N0) ===================="
declare -A WIN=( [1]=$IN [4]=input/scaling/W2_1698x1414.in [16]=input/scaling/L_2400x2000.in )
for P in 1 4 16; do
  for r in 1 2; do
    echo "----- BENCH mpi_weak ranks=$P size=W$P rep=$r -----"
    srun -n $P $BIN "${WIN[$P]}" none 0
  done
done

echo; echo "mpi_all DONE @ $(date)"
