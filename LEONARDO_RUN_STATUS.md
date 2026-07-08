# Leonardo run status — Particles final project

Session on Leonardo login node (`login07`), user `lscappat`, 2026-07-08.

## TL;DR — ✅ COMPLETED 2026-07-08

The Leonardo scheduler outage cleared ~17:09 CEST and the account submitted
normally (the `Invalid account` errors were outage symptoms, **not** an account
problem). Both jobs ran on an **NVIDIA A100-SXM-64GB** node and **COMPLETED (exit 0)**:

- **Correctness:** all 5 implementations generate **2231 particles**; `serial`,
  `omp`, `cuda`, `numba` are **bitwise identical** (`max_rel_diff=0`); `numba_cuda`
  agrees to `1.16e-3` (within the 2e-3 gate). The **A100 CUDA port is bitwise-identical
  to the serial baseline** — the critical never-before-run result. (The macOS FMA
  particle-count caveat did NOT occur: gcc-12 + nvcc both agree with Python at 2231.)
- **OpenMP strong scaling** vs serial: 2→2.00×, 4→3.99×, 8→7.97×, 16→15.3×,
  32→**29.2× (91% efficiency)**.
- **Throughput:** CUDA A100 **31.1×** (8.8 GInt/s) · numba-CPU 27.5× · numba_cuda 11.9×.
  (At N=2231 the A100 is under-occupied, so CUDA only just beats omp-32 — a
  problem-size scaling study would show the GPU pulling ahead at larger N.)

Artifacts: `reports/{summary.md, benchmarks.csv, speedup_omp.png, efficiency_omp.png,
throughput.png}`, full validation in `reports/validation.log`, job logs in `logs/`.
Rerun anytime once modules/venv are in place: `bash scripts/submit_pipeline.sh`.

**Exact build configuration** (exam deliverable — hardware/compiler/flags):
- Node: Leonardo Booster `lrdn1384`/`lrdn0016`, 1× Intel Xeon Platinum 8358 (32 cores),
  NVIDIA **A100-SXM-64GB** (driver 535.274.02, compute cap 8.0), 1 GPU used.
- Modules: `gcc/12.2.0`, `cuda/12.2` (nvcc), `cmake/3.27.9`,
  `hdf5/1.14.3--gcc--12.2.0-spack0.22`, `python/3.11.7`; venv: numpy 2.4.6,
  numba 0.65.1, h5py 3.16.0.
- C++ (serial): `g++ -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic -DUSE_HDF5`
- C++ (OpenMP): same + `-fopenmp`
- CUDA: `nvcc -O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_80,code=[compute_80,sm_80] -rdc=true -DUSE_HDF5` (no fast-math; `PARTICLES_FAST_MATH_CUDA=OFF`)
- No `-ffast-math`/`-march=native` anywhere (`PARTICLES_NATIVE_ARCH=OFF`); default
  gcc FMA contraction did **not** change the particle count on this platform (2231
  everywhere). Timing runs: HDF5 output **disabled** (`none 0`); OpenMP pinning
  `OMP_PROC_BIND=close`, `OMP_PLACES=cores`.

**Printed 17-digit `Final validation quantities:` blocks** (handoff §3, compared
across the five correctness runs): `cuda` is **identical to `serial` in all 11
quantities to all 17 digits**. `omp`, `python`, `numba` differ from serial only in
the last 1–2 digits (≈1e-15 relative) — their `/pos`,`/vel` arrays are bitwise
identical (validator max_rel_diff=0); the tiny print differences come from the
summation order *of the validation reductions themselves* (OpenMP parallel
reduction, numpy pairwise sums) — the expected parallel-reduction caveat.
`numba_cuda` differs at ≤1e-6 relative on the energy aggregates, consistent with
its 1.16e-3 component-level deviation, within the 2e-3 gate. (`momentum_x/y` are
catastrophic-cancellation quantities ~1e-8 on ~1e11-scale sums; their digits are
not meaningful.)

**Profiling & model analysis (handoff §5 + course lectures on nsys/ncu/roofline/Amdahl)**
(job 48943318; artifacts in `reports/nsys/`, `reports/ncu/`, `reports/amdahl_omp.png`):

- **nsys timeline** (official input): `computeForcesKernel` = **99.1% of GPU time**
  (201 launches, 610 µs each); halfKick kernels 0.9%; `cudaMemcpy` totals **0.19 ms
  in 9 calls** — data transfer is negligible, so streams/async overlap would buy
  nothing (course module on streams: applies when transfer ≈ compute; not here).
- **ncu kernel analysis** (`computeForcesKernel`, 128-thread blocks, 52 reg/thread):

  | metric | M (N=2231) | XL (N=35919) |
  |---|---|---|
  | Grid size (blocks on 108 SMs) | 18 | 281 |
  | Waves per SM | 0.02 | 0.25 |
  | Achieved occupancy | 6.1% | 13.8% |
  | SM (compute) throughput | 4.5% | 53.5% |
  | DRAM throughput | ~0.01% | ~0.00% |

  This is the scaling-study crossover, quantified: at the official size the GPU is
  ~95% idle (18 blocks can't feed 108 SMs); at XL it is busy and **unambiguously
  compute-bound** (DRAM ≈ 0% — all particle data fits in L1/L2, so arithmetic
  intensity is effectively huge; the kernel sits under the compute roof, far from
  the memory roof).
- **Roofline / pipe analysis @ XL**: FP64 is the hottest pipe at **63% utilization**
  ("kernel is FP64-bound" per ncu); achieved **37% of the device FP64 peak**. ncu's
  optimizer note: converting the 686M non-fused FP64 ops to FMA would gain ≈ **+31%**
  — we *deliberately decline* this (`__d*_rn` intrinsics, no `-ffast-math`) because
  non-contracted arithmetic is what makes the GPU results **bitwise identical** to
  serial. Profiler-quantified cost of reproducibility.
- **Amdahl's law fit** to the OpenMP strong scaling (`tools/amdahl_fit.py`):
  parallel fraction **f = 99.69%**, serial fraction **0.21%**, asymptotic max
  speedup ≈ **484×**; measured points match the model within ~3% at every p.

**Optional items — both completed later the same day:**

- **Pure-Python reference** (job 48940659, boost/normal, 14 min): `Particles_python.h5`
  generated; the full 15-pair validation matrix now **PASSes**, and `serial C++`,
  `omp`, `cuda`, `python`, `numba` are **all bitwise identical** (`max_rel_diff=0`).
  Pure Python runs at 0.00128 GInt/s — **~221× slower than serial C++**.
- **Problem-size scaling study** (job 48940656, boost/dbg, 14 min): same Mandelbrot
  window at grid 600×500 → 9600×8000, N = 567 → 143,768 (`input/scaling/*.in`,
  `run_scaling.sh`, parsed by `tools/parse_scaling.py` → `reports/scaling_*`):

  | N | serial GInt/s | omp-32 GInt/s | CUDA GInt/s | CUDA speedup vs serial |
  |---|---|---|---|---|
  | 567 | 0.282 | 5.78 | 1.78 | 6.3× (GPU under-occupied, loses to omp-32) |
  | 2,231 | 0.283 | 8.37 | 8.86 | 31× (crossover: GPU ≈ 32-core CPU) |
  | 8,996 | 0.283 | 8.39 | 36.8 | 130× |
  | 35,919 | 0.283 | 8.44 | 105.4 | 372× |
  | 143,768 | (0.283 extrap.) | 8.07 | **153.9** | **543×** (GPU 19× faster than omp-32) |

  serial is size-independent (flat 0.283); omp-32 saturates at ~8.4 GInt/s
  (~29–30×); the A100 keeps climbing with occupancy — the official N=2231 sits
  right at the CPU/GPU crossover, which explains why CUDA "only" tied omp-32
  in the headline benchmark. Steps per size: 200/200/200/50/10 (GInt/s is a
  rate, so different step counts remain comparable; serial XL measured with 2
  reps, XXL serial extrapolated).

---
_Historical context (the outage that blocked the first attempts):_

```bash
cd FinalProjects/Particles
bash scripts/submit_pipeline.sh
```

## ✅ Done (verified this session)

| Item | Status |
|---|---|
| CMake build (`leonardo-a100` preset) | ✅ `particles_serial`, `particles_omp`, `particles_cuda` built **warnings-clean** (HDF5 ON, OpenMP ON, CUDA sm_80 ON, Release) → `install/bin/` |
| Toolchain | gcc 12.2.0, cuda 12.2 (nvcc), cmake 3.27.9, hdf5 1.14.3, python 3.11.7 |
| Python venv (`particles_venv/`, system-site) | ✅ installed on login node: numpy 2.4.6, numba 0.65.1, llvmlite 0.47.0, h5py 3.16.0, matplotlib 3.10.9 (cupy/cuda-pathfinder skipped — unused by the 3 scripts) |
| Driver scripts | ✅ `run_cpp_all.sh`, `run_py_all.sh`, `submit_python_ref.sh`, `run_validate.sh`, `scripts/submit_pipeline.sh` |
| Benchmark parser/plotter | ✅ `tools/parse_benchmarks.py` |
| Account line added to all `submit_*.sh` | ✅ `--account=tra26_poliex`; redundant `--exclusive` removed (cpus=32+mem=0 already fills the node, and dbg forbids exclusive) |

## ⛔ Blocker: account not submittable

`saldo -b` shows the project **active**:
```
account         start       end        total  consumed
tra26_poliex    20260625    20261130   3000    0
```
and my association carries boost QoS:
```
leonardo | tra26_poliex | (no partition) | boost_qos_bprod,boost_qos_dbg,boost_qos_lprod,normal
```
**But every `sbatch` is rejected, tested with a healthy controller (so not a transient glitch):**

| Partition (combo tried) | Error |
|---|---|
| `boost_usr_prod` — dbg & normal, gpu & no-gpu, default acct | `Invalid account or account/partition combination specified` |
| `dcgp_usr_prod` — normal | `invalid account or expired budget` |
| `lrd_all_serial` | `Invalid account or account/partition combination specified` |

`boost_usr_prod` itself is open (`AllowAccounts=ALL`, `AllowQos=ALL`), so this is
**not** a partition ACL — the project association is not fully enabled in the
scheduler (accounting record + QoS exist, but the partition/budget link the
scheduler needs is missing). This is the "Marco Celoria is creating a dedicated
project" step from `LEONARDO_HANDOFF.md` — **not finished on CINECA's side.**

### Most likely cause: the announced scheduler outage (not the account)
CINECA HPC support emailed on **2026-07-08**: *"the slurm controller of Leonardo
is experiencing some instabilities ... we will notify you as soon as ... normal
production operations, including job submission and execution, have been
restored."* A half-restored slurmctld returns `Invalid account` for valid
accounts. **Primary action: wait for the "resolved" notification, then launch**
(a background watcher is polling and will auto-submit `cpp_all`+`py_all` the
moment submission recovers).

Backup — only if it still fails *after* CINECA declares it resolved: ask Marco
Celoria / **superc@cineca.it** to enable submission for `tra26_poliex` on
`boost_usr_prod`. Quick self-check either way:
```bash
sbatch -A tra26_poliex -p boost_usr_prod -q boost_qos_dbg --gres=gpu:1 \
       -N1 -t 00:02:00 --wrap 'hostname'     # should print "Submitted batch job <id>"
scancel <id>
```

> Note: the SLURM controller was intermittently returning
> `Socket timed out on send/recv` this session. All our scripts retry through
> that automatically; if you submit by hand and see it, just retry.

## ▶ How to launch once the account works

From `FinalProjects/Particles/`:

```bash
# 1) the two essential GPU jobs (boost/dbg): C++ + Python-GPU, correctness + benchmark
bash scripts/submit_pipeline.sh          # submits cpp_all + py_all, prints job ids

# 2) OPTIONAL pure-Python reference for the "Python vs Numba" check (~20-40 min)
sbatch submit_python_ref.sh

# 3) after jobs finish: validate on the LOGIN node (pure CPU, no GPU)
bash run_validate.sh

# 4) results table + speedup/efficiency/throughput plots
source particles_venv/bin/activate
python tools/parse_benchmarks.py logs/cpp_all_*.out logs/py_all_*.out --outdir reports --plots
```

Monitor: `squeue --me` · `tail -f logs/cpp_all_*.out`

## What each job does

- **`run_cpp_all.sh`** (boost/dbg, ~10 min): writes `Particles_{cpp,omp,cuda}.h5`
  (correctness, outputEvery=20), then benchmarks in `none 0` mode —
  serial ×3, OpenMP strong scaling `{1,2,4,8,16,32}` ×3 (pinned:
  `OMP_PROC_BIND=close`, `OMP_PLACES=cores`), CUDA ×3. Prints `Particles:` count
  per model — **check `cuda` matches `serial`** (FMA/particle-count caveat in
  `LEONARDO_HANDOFF.md`).
- **`run_py_all.sh`** (boost/dbg, ~10-15 min): `Particles_{numba,numba_cuda}.h5`
  (correctness) + numba/numba_cuda benchmarks ×3 (use reported *Pure dynamics
  time*, excludes JIT warm-up).
- **`run_validate.sh`** (login): particle counts + PASS1 (2e-3 course gate, full
  matrix) + PASS2 (strict 1e-12 near-bitwise on Cpp-Omp-Cuda and Python-Numba-NumbaCuda).

## Success criteria (from LEONARDO_HANDOFF.md §3)
`Cpp vs Omp`, `Cpp vs Cuda`, `Omp vs Cuda` PASS (near-bitwise); `Python vs Numba`,
`Numba vs NumbaCuda` PASS. Cross-family pairs PASS only if particle counts match
(else document the FMA/`-ffp-contract` shape-mismatch caveat). **`particles_cuda`
on a real A100 is the critical never-before-run result.**
