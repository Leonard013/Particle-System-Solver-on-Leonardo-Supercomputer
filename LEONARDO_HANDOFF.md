# Leonardo handoff — Particles final project: remaining experiments

Assignment for the Claude Code session running on the Leonardo cluster (CINECA).
Everything below assumes the working directory is `FinalProjects/Particles/`.

## State (verified on macOS, 2026-07-02 — do NOT redo)

Four parallel ports of the serial baseline exist and are committed:

| File | Model | Local verification |
|---|---|---|
| `src/cpp/particles_omp.cpp` | OpenMP | Built warning-clean via CMake preset; `/pos`,`/vel` **bitwise identical** to serial (official validator, rtol=1e-12); 4.2× on 11 cores |
| `src/python/particles_numba.py` | Numba CPU | **Bitwise identical** to `particles.py` and to serial C++ |
| `src/cpp/particles.cu` | CUDA (sm_80) | Deep code review only — **never compiled/run on a real GPU** |
| `src/python/particles_numba_cuda.py` | Numba CUDA | Ran under `NUMBA_ENABLE_CUDASIM=1`, passed strict validation — **never run on a real GPU** |

Design invariant (do not break): only the outer particle index is parallelized; the
inner force-sum order (j = 0..N-1) is preserved, so results are near-bitwise vs serial.
No fast-math anywhere; `particles.cu` uses `__d*_rn` intrinsics deliberately — keep
`PARTICLES_FAST_MATH_CUDA=OFF`.

## Known issue to check for (pre-existing in the course material, not in the ports)

On the full official `input/Particles.in`, C++ compiled with default flags FMA-contracts
the Mandelbrot loop and generates **2232** particles, while the Python family generates
**2231** (one borderline cell, i=1157 j=489, crosses threshold 2900). Proven on macOS:
`-ffp-contract=off` makes C++ match Python **bitwise**.

Consequences on Leonardo (gcc 12):
- `Cpp vs Python/Numba/NumbaCuda` in `validate_all.sh` may FAIL with a `/pos` **shape
  mismatch** on the official input. This is expected; it affects the shipped baselines
  themselves. Validate within families, or add `-ffp-contract=off` to the C++ flags to
  align families (verified remedy).
- **Check that `particles_cuda` prints the same `Particles:` count as `particles_serial`**
  (nvcc's host pass could contract differently from the g++ build). If it differs, add
  `-Xcompiler -ffp-contract=off` (CUDA host) / `-ffp-contract=off` (CXX) and rebuild.

## Tasks

### 0. Account setup
The old training reservation (`tra26_polimi`) is expired. Marco Celoria is creating a
dedicated project. Add `#SBATCH --account=<NEW_PROJECT>` to every `submit_*.sh` used
(or `export SBATCH_ACCOUNT=<NEW_PROJECT>`). Keep `--qos=boost_qos_dbg` for short runs;
drop it for the long benchmark sweeps if queue limits bite.

### 1. Build (login node)
```bash
source scripts/env.leonardo.sh
cmake --preset leonardo-a100 -DPARTICLES_BUILD_OPENMP=ON -DPARTICLES_BUILD_CUDA=ON
cmake --build --preset leonardo-a100 -j
cmake --install build/leonardo-a100        # → install/bin/{particles_serial,particles_omp,particles_cuda}
```
Expected: all three targets build; warnings-clean. `PARTICLES_STRICT_CUDA` behavior: ON
in the preset means a missing CUDA toolchain is fatal — the `cuda/12.2` module must be loaded.

### 2. Python env (once, compute or serial node)
```bash
sbatch submit_install_pyenv.sh            # creates particles_venv (system-site-packages)
```

### 3. Correctness on the A100 (the critical new result)
Run all five with HDF5 output on the official input, then validate:
```bash
sbatch submit_cpp.sh && sbatch submit_omp.sh && sbatch submit_cuda.sh
sbatch submit_numba.sh && sbatch submit_numba_cuda.sh
sbatch submit_validate.sh                 # runs scripts/validate_all.sh (RTOL=ATOL=2e-3)
```
Success criteria:
- `Cpp vs Omp`, `Cpp vs Cuda`, `Omp vs Cuda`: PASS (expect near-bitwise, far below 2e-3).
- `Python vs Numba`, `Numba vs NumbaCuda`: PASS.
- Cross-family pairs: PASS only if particle counts match (see known issue above);
  otherwise document the shape mismatch + FMA explanation in the report.
- Also compare the printed `Final validation quantities:` blocks (17 digits).

### 4. Benchmarks (benchmark mode: `none 0`, no HDF5)
All on the official input unless noted. One node, `--exclusive`.
- **Serial baseline**: 1 run, record `Pure dynamics time` and GInteractions/s.
- **OpenMP strong scaling**: `OMP_NUM_THREADS in {1,2,4,8,16,32}`, 3 repetitions each,
  report median; compute speedup and parallel efficiency vs serial.
- **CUDA**: 3 repetitions; report GInteractions/s and speedup vs serial and vs omp-32.
- **Numba / Numba-CUDA**: 1–3 runs each (exclude JIT warm-up: use the reported
  `Pure dynamics time`, not wall time).
- **Problem-size scaling (optional)**: scale the generating grid (e.g. 600×500,
  1200×1000, 2400×2000 with proportional screen grid) to grow N; plot GInteractions/s
  vs N for serial/omp-32/cuda.

### 5. Profiling (optional, for the presentation)
```bash
module load nvhpc   # for nsys/ncu
nsys profile --trace=cuda --stats=true -o reports/nsys_cuda ./install/bin/particles_cuda input/Particles.in none 0
ncu --set full -o reports/ncu_force ./install/bin/particles_cuda input/Particles.in none 0
```
Kernel of interest: `computeForcesKernel` (expect compute-bound, high occupancy).

### 6. Deliverables
- `logs/` from all runs (job scripts write `logs/%x_%j.out`).
- A results table: model → dynamics time → GInteractions/s → speedup vs serial.
- Speedup + efficiency plots (OpenMP threads; CPU vs GPU bar chart).
- Validation summary (which pairs pass at 2e-3, max diffs, plus the FMA note if hit).
- Report hardware (A100-64GB, node config), modules/compilers, and flags used.

## Numerical expectations (from local verification)
`omp` and `cuda` vs `serial`: positions/velocities should be bitwise or within a few ulp;
any deviation beyond ~1e-12 relative indicates an FMA/fast-math regression — investigate
before benchmarking. The 2e-3 course tolerance is a very generous outer gate.
