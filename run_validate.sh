#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run on the LOGIN NODE (pure CPU + h5py, no GPU/SLURM needed):
#     bash run_validate.sh
# after the jobs have written output/*.h5.
#
#   Pass 1: course gate, RTOL=ATOL=2e-3 over the full pair matrix.
#   Pass 2: STRICT (rtol=1e-12) on the pairs claimed near-bitwise
#           (Cpp-Omp-Cuda within C++ family; Python-Numba-NumbaCuda within Py).
# ---------------------------------------------------------------------------
set -uo pipefail
source scripts/env.leonardo.sh          # activates particles_venv (h5py, numpy)
V=tools/validate_particles_h5.py
O=output

echo "############ available outputs ############"
ls -la "$O"/*.h5 2>&1
echo

echo "############ particle counts per file (HDF5 attr) ############"
python - <<'PY'
import glob, h5py
for f in sorted(glob.glob("output/*.h5")):
    try:
        with h5py.File(f,"r") as h:
            n = h.attrs.get("particles", "?")
            steps = h["/step"][()] if "/step" in h else []
            print(f"  {f:40s} particles={n}  frames={len(steps)}  last_step={steps[-1] if len(steps) else '?'}")
    except Exception as e:
        print(f"  {f:40s} ERROR: {e}")
PY
echo

echo "################################################################"
echo "### PASS 1 - course gate: RTOL=ATOL=2e-3 (full matrix) ###"
echo "################################################################"
RTOL=2e-3 ATOL=2e-3 bash scripts/validate_all.sh

echo
echo "################################################################"
echo "### PASS 2 - STRICT near-bitwise (rtol=1e-12 atol=1e-14) ###"
echo "################################################################"
strict() {
  local a=$1 b=$2
  echo; echo "--- STRICT $a vs $b ---"
  if [[ -f "$O/Particles_$a.h5" && -f "$O/Particles_$b.h5" ]]; then
    python "$V" "$O/Particles_$a.h5" "$O/Particles_$b.h5" --rtol=1e-12 --atol=1e-14
  else
    echo "SKIP: missing one of Particles_$a.h5 / Particles_$b.h5"
  fi
}
strict cpp omp
strict cpp cuda
strict omp cuda
strict python numba
strict numba numba_cuda
echo; echo "validate DONE @ $(date)"
