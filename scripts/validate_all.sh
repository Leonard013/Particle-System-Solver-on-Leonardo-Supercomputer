#!/usr/bin/env bash

# Run every comparison so one failure does not hide later diagnostics, then
# return a single trustworthy status to callers and SLURM.
set -uo pipefail

RTOL="${RTOL:-2e-3}"
ATOL="${ATOL:-2e-3}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

VALIDATOR="tools/validate_particles_h5.py"
status=0

compare() {
  local label="$1"
  local reference="$2"
  local candidate="$3"

  echo
  echo "$label"
  if ! "$PYTHON_BIN" "$VALIDATOR" "$reference" "$candidate" \
      --rtol="$RTOL" --atol="$ATOL"; then
    status=1
  fi
}

compare "Cpp vs Omp" output/Particles_cpp.h5 output/Particles_omp.h5
compare "Cpp vs Cuda" output/Particles_cpp.h5 output/Particles_cuda.h5
compare "Cpp vs Python" output/Particles_cpp.h5 output/Particles_python.h5
compare "Cpp vs Numba" output/Particles_cpp.h5 output/Particles_numba.h5
compare "Cpp vs NumbaCuda" output/Particles_cpp.h5 output/Particles_numba_cuda.h5
compare "Omp vs Cuda" output/Particles_omp.h5 output/Particles_cuda.h5
compare "Omp vs Python" output/Particles_omp.h5 output/Particles_python.h5
compare "Omp vs Numba" output/Particles_omp.h5 output/Particles_numba.h5
compare "Omp vs NumbaCuda" output/Particles_omp.h5 output/Particles_numba_cuda.h5
compare "Cuda vs Python" output/Particles_cuda.h5 output/Particles_python.h5
compare "Cuda vs Numba" output/Particles_cuda.h5 output/Particles_numba.h5
compare "Cuda vs NumbaCuda" output/Particles_cuda.h5 output/Particles_numba_cuda.h5
compare "Python vs Numba" output/Particles_python.h5 output/Particles_numba.h5
compare "Python vs NumbaCuda" output/Particles_python.h5 output/Particles_numba_cuda.h5
compare "Numba vs NumbaCuda" output/Particles_numba.h5 output/Particles_numba_cuda.h5

echo
if (( status == 0 )); then
  echo "COURSE GATE PASS: all 15 comparisons satisfy rtol=$RTOL, atol=$ATOL."
else
  echo "COURSE GATE FAIL: one or more comparisons failed." >&2
fi
exit "$status"
