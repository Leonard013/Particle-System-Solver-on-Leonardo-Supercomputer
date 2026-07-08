#!/usr/bin/env bash
# ===========================================================================
# Launch the Particles experiments on Leonardo (boost-only).
#
# Account tra26_poliex has ONLY boost QoS (dbg/bprod/lprod/normal) and no
# DCGP/serial access, so every job runs on boost_usr_prod. boost_qos_dbg allows
# at most 2 jobs (running+pending) per user, so the two essential jobs are:
#     cpp_all  (C++ correctness + benchmark)      -> boost/dbg   [slot 1]
#     py_all   (numba + numba_cuda corr + bench)  -> boost/dbg   [slot 2]
#
# Prereqs (already done in this repo):
#   * install/bin/{particles_serial,particles_omp,particles_cuda}  (cmake build)
#   * particles_venv/  (created on the login node; numba, h5py, numpy, mpl)
#
# Robust against transient "Socket timed out on send/recv" from slurmctld.
# Usage:  bash scripts/submit_pipeline.sh
# ===========================================================================
set -uo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"   # -> Particles/
mkdir -p logs output

rsbatch() {
  local n=0 out rc
  while :; do
    out="$(sbatch "$@" 2>&1)"; rc=$?
    [[ $rc -eq 0 ]] && { printf '%s\n' "$out"; return 0; }
    if [[ "$out" == *"timed out"* || "$out" == *"Unable to contact"* || "$out" == *"try again"* ]]; then
      n=$((n+1)); [[ $n -ge 10 ]] && { echo "$out" >&2; return 1; }
      sleep $((2*n)); continue
    fi
    echo "$out" >&2; return $rc
  done
}
jid(){ rsbatch "$@" | grep -oE '[0-9]+' | tail -1; }
req(){ [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo "FATAL: submit failed at '$2'."; \
       echo "       If the error was 'Invalid account or account/partition combination',"; \
       echo "       the project tra26_poliex is not yet enabled for submission --"; \
       echo "       see FinalProjects/Particles/LEONARDO_RUN_STATUS.md."; exit 1; }; }

echo "[1/2] cpp_all  (boost/dbg): C++ correctness (HDF5) + benchmark sweep"
CPP=$(jid run_cpp_all.sh); req "$CPP" cpp_all; echo "      -> $CPP"
echo "[2/2] py_all   (boost/dbg): numba + numba_cuda correctness + benchmark"
PY=$(jid run_py_all.sh);   req "$PY"  py_all;  echo "      -> $PY"
printf '%s %s\n' "$CPP" "$PY" > logs/.jobids

cat <<EOF

Submitted:  cpp_all=$CPP  py_all=$PY   (ids in logs/.jobids)

Next:
  * OPTIONAL pure-Python reference (boost/normal, ~20-40 min, off critical path):
        sbatch submit_python_ref.sh
  * After the jobs finish, VALIDATE on the login node (no GPU needed):
        bash run_validate.sh
  * Parse benchmarks + make plots:
        source particles_venv/bin/activate
        python tools/parse_benchmarks.py logs/cpp_all_*.out logs/py_all_*.out --outdir reports --plots
  * Watch progress:
        squeue --me ; tail -f logs/cpp_all_*.out
EOF
squeue --me -o "%.10i %.14j %.15P %.13q %.9T %.10M %.11l %R" 2>&1 | head