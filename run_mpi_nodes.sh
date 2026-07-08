#!/bin/bash -l
#SBATCH --account=tra26_poliex
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=0:30:00
#SBATCH --job-name=mpi_nodes
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# ---------------------------------------------------------------------------
# MPI port, MULTI-NODE (up to 4 nodes x 32 ranks = 128 ranks over the
# Dragonfly+ network). Inter-node scaling for three problem sizes:
#   M  (N~2231,  official): latency-dominated at high rank counts --
#                            expected efficiency drop (course: comm overhead)
#   L  (N~8996):  compute still dominates, should scale
#   XL (N~35919): plenty of work per rank, best inter-node efficiency
# plus the p=64 weak-scaling point (W8, N~17850).
# ---------------------------------------------------------------------------
set -uo pipefail
source scripts/env.leonardo.sh
BIN=./install/bin/particles_mpi
mkdir -p output

echo "############################################################"
echo "# nodes: $SLURM_JOB_NUM_NODES  ($SLURM_JOB_NODELIST)   date: $(date)"
echo "############################################################"

echo; echo "==================== INTER-NODE STRONG SCALING ===================="
for P in 32 64 128; do
  for r in 1 2 3; do
    echo "----- BENCH mpi ranks=$P size=M rep=$r -----"
    srun -n $P ./install/bin/particles_mpi input/Particles.in none 0
  done
done
for P in 32 64 128; do
  for r in 1 2 3; do
    echo "----- BENCH mpi ranks=$P size=L rep=$r -----"
    srun -n $P $BIN input/scaling/L_2400x2000.in none 0
  done
done
for P in 64 128; do
  for r in 1 2 3; do
    echo "----- BENCH mpi ranks=$P size=XL rep=$r -----"
    srun -n $P $BIN input/scaling/XL_4800x4000.in none 0
  done
done

echo; echo "==================== WEAK SCALING p=64 (W8) ===================="
for r in 1 2; do
  echo "----- BENCH mpi_weak ranks=64 size=W64 rep=$r -----"
  srun -n 64 $BIN input/scaling/W8_3394x2828.in none 0
done

echo; echo "mpi_nodes DONE @ $(date)"
