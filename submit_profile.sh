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
#SBATCH --job-name=profile_cuda
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# ---------------------------------------------------------------------------
# Handoff §5 / course profiling lectures: nsys timeline + ncu kernel analysis
# (incl. the roofline section) for particles_cuda on the A100.
#   * nsys: whole-app timeline at the official size (kernels vs memcpy vs API)
#   * ncu:  computeForcesKernel at M (N=2231, under-occupied) and
#           XL (N=35919, saturated) -- the contrast explains the CPU/GPU
#           crossover seen in the scaling study.
# ---------------------------------------------------------------------------
set -uo pipefail
source scripts/env.leonardo.sh
module load nvhpc          # provides nsys / ncu (loaded after env's module purge)
command -v nsys && nsys --version | head -1
command -v ncu  && ncu  --version | tail -1

BIN=./install/bin/particles_cuda
mkdir -p reports/nsys reports/ncu

echo "=== [1/4] nsys timeline, official input (benchmark mode) ==="
nsys profile --trace=cuda --stats=true --force-overwrite=true \
     -o reports/nsys/particles_cuda_M \
     $BIN input/Particles.in none 0

echo; echo "=== [2/4] ncu full set + roofline, computeForcesKernel @ M (N=2231) ==="
ncu --set full --section SpeedOfLight_HierarchicalDoubleRooflineChart \
    -k computeForcesKernel -c 3 -f -o reports/ncu/force_M \
    $BIN input/Particles.in none 0

echo; echo "=== [3/4] ncu full set + roofline, computeForcesKernel @ XL (N=35919) ==="
ncu --set full --section SpeedOfLight_HierarchicalDoubleRooflineChart \
    -k computeForcesKernel -c 3 -f -o reports/ncu/force_XL \
    $BIN input/scaling/XL_4800x4000.in none 0

echo; echo "=== [4/4] text summaries (details page) ==="
ncu --import reports/ncu/force_M.ncu-rep  --page details > reports/ncu/force_M.txt  2>&1
ncu --import reports/ncu/force_XL.ncu-rep --page details > reports/ncu/force_XL.txt 2>&1
echo "--- key metrics M ---";  grep -E "Compute \(SM\)|Memory Throughput|DRAM Throughput|Achieved Occupancy|Theoretical Occupancy|Duration|Elapsed Cycles|Block Size|Grid Size|FP64|Waves Per SM" reports/ncu/force_M.txt | head -20
echo "--- key metrics XL ---"; grep -E "Compute \(SM\)|Memory Throughput|DRAM Throughput|Achieved Occupancy|Theoretical Occupancy|Duration|Elapsed Cycles|Block Size|Grid Size|FP64|Waves Per SM" reports/ncu/force_XL.txt | head -20

echo; echo "profile_cuda DONE @ $(date)"
