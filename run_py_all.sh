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
#SBATCH --job-name=py_all
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# ---------------------------------------------------------------------------
# Python GPU/JIT family: correctness (HDF5 out) + benchmark (none 0).
# Requires particles_venv (numba, h5py). env.leonardo.sh activates it.
# ---------------------------------------------------------------------------
set -uo pipefail
source scripts/env.leonardo.sh

IN=./input/Particles.in
PY=./src/python
mkdir -p output

# numba CPU threads
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

echo "############################################################"
echo "# host   : $(hostname)   date: $(date)"
echo "# GPU    :"; nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader 2>/dev/null | sed 's/^/#          /'
echo "# python : $(command -v python)  ($(python --version 2>&1))"
echo "# numba  : $(python -c 'import numba,sys; print(numba.__version__)' 2>&1)"
echo "# numpy  : $(python -c 'import numpy,sys; print(numpy.__version__)' 2>&1)"
echo "# h5py   : $(python -c 'import h5py,sys; print(h5py.__version__)' 2>&1)"
echo "# cuda ok: $(python -c 'from numba import cuda; print(cuda.is_available())' 2>&1)"
echo "############################################################"

echo; echo "==================== CORRECTNESS (HDF5, outputEvery=20) ===================="
echo "----- [numba CPU] -> output/Particles_numba.h5 -----"
python "$PY/particles_numba.py" "$IN" output/Particles_numba.h5
echo "----- [numba CUDA] -> output/Particles_numba_cuda.h5 -----"
python "$PY/particles_numba_cuda.py" "$IN" output/Particles_numba_cuda.h5

echo; echo "==================== BENCHMARK (none 0, no HDF5) ===================="
echo ">>>>> NUMBA CPU (x3) [use reported 'Pure dynamics time', excludes JIT warm-up]"
for r in 1 2 3; do
  echo "----- BENCH numba rep=$r -----"
  python "$PY/particles_numba.py" "$IN" none 0
done

echo ">>>>> NUMBA CUDA (x3)"
for r in 1 2 3; do
  echo "----- BENCH numba_cuda rep=$r -----"
  python "$PY/particles_numba_cuda.py" "$IN" none 0
done

echo; echo "py_all DONE @ $(date)"
