#!/bin/bash -l
#SBATCH --account=tra26_poliex
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --job-name=python_ref
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# ---------------------------------------------------------------------------
# OPTIONAL, off critical path. Pure-Python reference (scalar O(N^2),
# single-threaded) -> output/Particles_python.h5 for the "Python vs Numba"
# validation. ~20-40 min. This account has no serial/DCGP access, so it runs
# on boost with the normal QoS (NOT the 2-job debug queue).
#
# CPU-only by intent (no GPU needed). If boost_usr_prod rejects a job without a
# GPU, add:  #SBATCH --gres=gpu:1
# ---------------------------------------------------------------------------
set -uo pipefail
source scripts/env.leonardo.sh
mkdir -p output
echo "host: $(hostname)  date: $(date)"
python ./src/python/particles.py ./input/Particles.in output/Particles_python.h5
echo "python_ref DONE @ $(date)"
