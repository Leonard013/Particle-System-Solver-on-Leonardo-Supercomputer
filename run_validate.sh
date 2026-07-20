#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run on the LOGIN NODE (pure CPU + h5py, no GPU/SLURM needed):
#     bash run_validate.sh
# after the jobs have written output/*.h5.
#
#   Pass 1: course gate, RTOL=ATOL=2e-3 over the full pair matrix.
#   Pass 2: STRICT (rtol=1e-12) on pairs expected to agree. The known
#           Numba-vs-Numba-CUDA strict mismatch is reported but is not a failure.
# ---------------------------------------------------------------------------
set -uo pipefail
if ! source scripts/env.leonardo.sh; then
  echo "FATAL: could not load the Leonardo validation environment." >&2
  exit 1
fi
V=tools/validate_particles_h5.py
O=output
PYTHON_BIN="${PYTHON_BIN:-python3}"
status=0

echo "############ available outputs ############"
ls -la "$O"/*.h5 2>&1
echo

echo "############ particle counts per file (HDF5 attr) ############"
"$PYTHON_BIN" - <<'PY'
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
if ! RTOL=2e-3 ATOL=2e-3 PYTHON_BIN="$PYTHON_BIN" bash scripts/validate_all.sh; then
  status=1
fi

echo
echo "################################################################"
echo "### PASS 2 - STRICT near-bitwise (rtol=1e-12 atol=1e-14) ###"
echo "################################################################"
strict_required() {
  local a=$1 b=$2
  echo; echo "--- STRICT $a vs $b ---"
  if [[ -f "$O/Particles_$a.h5" && -f "$O/Particles_$b.h5" ]]; then
    if ! "$PYTHON_BIN" "$V" "$O/Particles_$a.h5" "$O/Particles_$b.h5" \
        --rtol=1e-12 --atol=1e-14; then
      status=1
    fi
  else
    echo "FAIL: missing Particles_$a.h5 or Particles_$b.h5" >&2
    status=1
  fi
}

strict_optional() {
  local a=$1 b=$2
  echo; echo "--- STRICT OPTIONAL $a vs $b ---"
  if [[ -f "$O/Particles_$a.h5" && -f "$O/Particles_$b.h5" ]]; then
    if ! "$PYTHON_BIN" "$V" "$O/Particles_$a.h5" "$O/Particles_$b.h5" \
        --rtol=1e-12 --atol=1e-14; then
      status=1
    fi
  else
    echo "SKIP: optional MPI output is not present."
  fi
}

strict_informational() {
  local a=$1 b=$2
  echo; echo "--- STRICT INFORMATIONAL $a vs $b ---"
  if [[ -f "$O/Particles_$a.h5" && -f "$O/Particles_$b.h5" ]]; then
    if "$PYTHON_BIN" "$V" "$O/Particles_$a.h5" "$O/Particles_$b.h5" \
        --rtol=1e-12 --atol=1e-14; then
      echo "INFO: this pair also satisfies the strict threshold."
    else
      echo "INFO: strict mismatch expected; the course-gate result above is authoritative."
    fi
  else
    echo "SKIP: missing one of Particles_$a.h5 / Particles_$b.h5"
  fi
}

strict_required cpp omp
strict_required cpp cuda
strict_required omp cuda
strict_required python numba
strict_optional cpp mpi
strict_informational numba numba_cuda

echo
if (( status == 0 )); then
  echo "VALIDATION PASS @ $(date)"
else
  echo "VALIDATION FAIL @ $(date)" >&2
fi
exit "$status"
