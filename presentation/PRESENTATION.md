# Exam presentation spec — Particles final project

**Audience for this file:** the Claude Code session that will BUILD the PowerPoint.
Everything you need is in this file + the PNGs under `reports/`. Do not re-derive
numbers; every figure below was verified against the committed logs/CSVs.

## Build instructions

- Output: `presentation/Particles_exam_fixed.pptx`, 16:9 (13.33 × 7.5 in).
- Tooling: `@oai/artifact-tool`; preserve the inherited deck objects and styling.
- Style: white background, dark near-black text, ONE accent color (e.g. #1f6feb)
  for emphasis/numbers; sans-serif (Calibri/Helvetica); title ~32 pt, body ~18 pt.
- ≤ 6 bullets per slide; bullets are FINAL TEXT — do not paraphrase or pad.
- Figures: paths are relative to `FinalProjects/Particles/`. Figure slides use
  title + image right-half (or full-width if no bullets); keep native aspect.
- Tables: render as native pptx tables, header row shaded, numbers right-aligned.
- Every slide has SPEAKER NOTES below — put them in the pptx notes pane verbatim.
- The presentation must satisfy the 5 exam points from `FinalProjects/README.md`:
  (1) original application, (2) numerical correctness, (3) optimization work,
  (4) benchmark results incl. hardware/flags, (5) speed-up & efficiency plots.
  Slide ↔ requirement mapping: (1)=S2–S3, (2)=S5–S7, (3)=S4,S9,S12,S15,
  (4)=S8, (5)=S10–S11,S13–S14,S16.

---

## S1 — Title

**Particles: parallelizing an O(N²) N-body solver**
C++/Python baselines → OpenMP, CUDA, MPI, Numba CPU and Numba CUDA

Leonardo Scappatura — Polimi PhD School, HPC course final project — July 2026

Metric cards: **7** runnable programs · **6** bitwise-identical states ·
**543×** max GPU speed-up · **113.7×** MPI at 128 ranks.

NOTES: One-liner of the result up front: seven runnable programs—two baselines plus
five parallel ports. Six produce bitwise-identical state arrays; Numba CUDA passes
the course tolerance. Speedup reaches 543× on one GPU and 114× across four MPI nodes.

## S2 — The application

Bullets:
- 2D particle-dynamics toy solver from the course repository (C++17 + Python baselines).
- Particle generation: Mandelbrot field on a 1200×1000 grid (3000 iters); cells above
  threshold become particles → **N = 2231**, weight = mass.
- Dynamics: **all-pairs softened force** — kForce·wᵢwⱼ/(r²+ε²)^{3/2}, kForce=10⁻³, ε²=10⁻⁴.
- Integrator: velocity-Verlet, 200 steps, dt=10⁻³ → **O(N²)·steps ≈ 10⁹ interactions**.
- Output: HDF5 (/pos /vel /step /weight) + 11 printed validation quantities
  (sums, momenta, kinetic/potential-like energy).

NOTES: The force kernel is 99+% of the runtime — the whole project is about that
loop. Weight doubles as mass; the numerical model, input format, and validation
quantities are contractual and must not change.

## S3 — Strategy: reproducibility first

Bullets:
- Design invariant used in EVERY port: **parallelize only the outer particle index i;
  keep the inner j-sum in serial order (j = 0…N−1)**.
- Each fx[i] then accumulates in exactly the serial order → results can be
  **bitwise identical**, not just "within tolerance".
- Corollaries: no fast-math anywhere; no FMA contraction (CUDA uses `__dadd_rn`/
  `__dmul_rn` intrinsics); no pair-symmetry tricks that create write races.
- Baseline untouched: each port is a separate self-contained source file, exactly
  as the CMake scaffold expects.

NOTES: This is the thesis of the talk. Parallel ≠ nondeterministic: if you preserve
the accumulation order per output element, an embarrassingly parallel outer loop
reproduces serial floating point exactly. The profiler later estimates a +31% FMA
optimization opportunity, which we decline to retain that reproducibility.

## S4 — Five parallel ports + two baselines

Table (model | file | key techniques):
| Serial baselines | src/cpp/particles.cpp + src/python/particles.py | C++ and NumPy references; identical model and HDF5 schema |
| OpenMP | src/cpp/particles_omp.cpp | `parallel for` on outer loops; reductions for validation sums |
| CUDA | src/cpp/particles.cu | 1 thread = 1 particle; device-resident SoA; `__d*_rn` intrinsics; sm_80 |
| MPI | src/cpp/particles_mpi.cpp | block decomposition of i; per-step `MPI_Allgatherv` of positions |
| Numba CPU | src/python/particles_numba.py | `@njit(parallel=True)` + `prange`, fastmath=False |
| Numba CUDA | src/python/particles_numba_cuda.py | `@cuda.jit` kernels, float64, device-resident |

NOTES: Five optimized ports plus two serial baselines. The six-way course matrix
excludes MPI and includes both baselines; MPI is validated separately against C++ serial.

## S5 — Correctness methodology

Bullets:
- Every implementation writes the same HDF5 datasets; the course validator compares
  `/step /weight /pos /vel` pairwise.
- **15-pair matrix** across C++ serial, OpenMP, CUDA, Python, Numba CPU and Numba
  CUDA at the course gate rtol=atol=2·10⁻³.
- MPI checked separately against C++ serial from 1 to 128 ranks.
- Strict pass at rtol=10⁻¹² for expected bitwise pairs; Numba-CUDA is informational.
- Plus: 17-significant-digit comparison of the 11 printed validation quantities.

NOTES: Validation is dataset-level over full trajectories (11 frames × N × 2), not
just final scalars — position errors would compound over 200 steps and get caught.

## S6 — Correctness results (Leonardo A100, official input)

Bullets:
- **15/15 pairs PASS** the six-way course-gate matrix; MPI also passes its separate
  serial comparison.
- C++ serial, OpenMP, CUDA, MPI, Python and Numba CPU produce **bitwise-identical**
  /pos and /vel arrays.
- CUDA = serial in **all 11 validation quantities to all 17 printed digits**.
- MPI bitwise-identical from **1 to 128 ranks** (decomposition preserves sum order).
- Numba-CUDA max relative velocity difference is **1.158·10⁻³**, below the
  2·10⁻³ course gate (position max absolute difference: 4.56·10⁻⁵).

NOTES: "Bitwise" is the headline — it validates the design invariant of S3. The
Numba-CUDA deviation is the one place we can't control contraction as precisely
(NVVM codegen); its worst relative difference is about 1.7× below the course limit.
Momentum components are catastrophic-cancellation quantities (~10⁻⁸ on 10¹¹-scale
sums), so their trailing digits are not meaningful — expected and documented.

## S7 — Case study: one FMA, one particle

Bullets:
- On macOS (clang, default `-ffp-contract`), the **baseline itself** generates
  **2232** particles; Python generates **2231**.
- Root cause: FMA contraction in the Mandelbrot loop changes 14 of 1.2 M cells;
  ONE cell (i=1157, j=489) crosses the selection threshold (2900).
- Fix/proof: `-ffp-contract=off` → C++ matches Python **bitwise**.
- On Leonardo (gcc 12 + nvcc): all seven runnable programs agree at **N = 2231**.

NOTES: Found during local verification before touching the cluster. A single fused
multiply-add changed the particle COUNT — the dataset shapes stop matching, not just
values. This motivates the no-contraction discipline of S3 and shows validation must
compare structure, not only tolerances. On the exam platform the issue does not
arise, but the report documents it for reproducibility on other machines.

## S8 — Benchmark setup (hardware / compilers / method)

Bullets:
- Leonardo Booster node: Xeon Platinum 8358 (32 cores) + **NVIDIA A100-SXM-64GB**
  (108 SMs, driver 535.274.02); MPI: up to 4 nodes / 128 ranks over Dragonfly+.
- Toolchain: gcc 12.2.0, CUDA 12.2 (nvcc, sm_80), cmake 3.27.9, hdf5 1.14.3,
  python 3.11.7, numpy 2.4.6, numba 0.65.1.
- Flags: `-O3 -std=c++17` (+`-fopenmp`); **no fast-math, no -march=native**.
- Method: benchmark mode = HDF5 **disabled** (`none 0`); metric = "pure dynamics"
  time (excludes JIT warm-up & I/O); **median of 3 runs**; threads pinned
  (`OMP_PROC_BIND=close, OMP_PLACES=cores`).

NOTES: Exam deliverable #4 verbatim. Serial reference: 3.516 s for 200 steps at
N=2231 = 0.283 GInteractions/s, flat across problem sizes (compute-bound, cache-resident).

## S9 — OpenMP: what was parallelized

Bullets:
- `#pragma omp parallel for schedule(static)` on: force outer loop, both Verlet
  half-kick loops, Mandelbrot rows (`collapse(2)`).
- Validation reductions: `reduction(+:…)` incl. the O(N²) potential.
- Each thread writes only its own fx[i]/fy[i] → race-free by construction, inner
  order untouched → bitwise result for ANY thread count.

NOTES: ~10 pragmas on the baseline; deliberately minimal so the diff vs serial is
reviewable. Bitwise at every thread count tested (1…32).

## S10 — OpenMP strong scaling  【figures: reports/speedup_omp.png + reports/efficiency_omp.png side by side】

Table (threads | speedup | efficiency):
| 2 | 2.00× | 100% |
| 4 | 3.99× | 100% |
| 8 | 7.97× | 100% |
| 16 | 15.31× | 96% |
| 32 | **29.20×** | **91%** |

NOTES: Near-ideal to 8 threads; 91% at the full socket. The 16/32-thread dips are
memory-subsystem sharing, not algorithmic — the Amdahl fit next quantifies it.

## S11 — Amdahl's law fit  【figure: reports/amdahl_omp.png】

Bullets:
- Least-squares fit of 1/S vs 1/p over p = 1…32.
- **Serial fraction ≈ 0.21% → speedup ceiling ≈ 480×.**
- Measured points match the model within ~3% at every p.
- The 32-core node exploits only ~6% of the parallel headroom → the algorithm
  scales far beyond one CPU (motivates GPU + MPI).

NOTES: Course topic "measuring performance". The tiny serial fraction is the
integrator + loop bookkeeping. Ceiling ≈480× is the Amdahl asymptote 1/(serial
fraction) — the fitted intercept of the linearized model.

## S12 — CUDA port: what was offloaded

Bullets:
- One thread per particle i (128-thread blocks); inner j-loop in registers,
  serial order preserved.
- Whole SoA state **device-resident across all 200 steps**; copies only for HDF5
  frames and the final validation (9 memcpys, 0.19 ms total).
- Round-to-nearest intrinsics (`__dadd_rn`, `__dmul_rn`, `__drcp_rn(sqrt)`) —
  immune to `--fmad` contraction by construction.
- Result (S6): **bitwise identical to serial** on the A100.

NOTES: No shared-memory tiling: at these N the kernel is compute-bound with all
data in L1/L2 (profiling slide), so tiling adds complexity for ~nothing — the
honest engineering answer, backed by ncu data two slides ahead.

## S13 — CPU vs GPU at the official size  【figure: reports/throughput.png】

Table (implementation | GInt/s | speedup vs serial):
| serial | 0.283 | 1× |
| OpenMP-32 | 8.31 | 29.2× |
| **CUDA A100** | **8.80** | **31.1×** |
| Numba CPU | 7.83 | 27.5× |
| Numba CUDA | 3.42 | 11.9× |

Bullets:
- At N=2231 the A100 **barely beats 32 CPU cores** — why?
- ncu: grid = 18 blocks on **108 SMs** → occupancy **6.1%**, SM throughput 4.5% —
  the GPU is ~95% idle.

NOTES: Key teaching moment: the official problem is too small for the GPU. N=2231
sits almost exactly at the CPU/GPU crossover. Numba-CPU at 27.5× ≈ hand-written
OpenMP is remarkable for Python; Numba-CUDA pays kernel-launch + codegen overheads
at tiny N.

## S14 — Problem-size scaling: the GPU needs work  【figures: reports/scaling_giups.png + reports/scaling_speedup.png】

Table (N | serial GInt/s | OpenMP-32 | CUDA | CUDA speedup):
| 567 | 0.282 | 5.78 | 1.78 | 6.3× |
| 2,231 | 0.283 | 8.37 | 8.86 | 31× ← crossover |
| 8,996 | 0.283 | 8.39 | 36.8 | 130× |
| 35,919 | 0.283 | 8.44 | 105.4 | 372× |
| 143,768 | (0.283) | 8.07 | **153.9** | **543×** |

NOTES: Serial is size-independent (0.283 flat); OpenMP saturates at ~8.4 GInt/s
(~29×); the A100 climbs with occupancy and is 19× faster than the full CPU socket
at N=144k. XXL serial is extrapolated from the flat rate (stated in the report).
Different step counts per size are fine — GInt/s is a rate.

## S15 — GPU profiling: nsys + ncu (roofline view)

Bullets:
- nsys: `computeForcesKernel` = **99.1%** of GPU time; memcpy total **0.19 ms**
  → streams/overlap would buy nothing.
- ncu @ XL: SM throughput 53.5%, **DRAM ≈ 0%** → purely **compute-bound**
  (all particle data lives in L1/L2; effectively infinite arithmetic intensity).
- Hottest pipe: **FP64 at 63%**; achieved ≈ **37% of A100 FP64 peak**.
- ncu optimizer: fusing to FMA would gain ≈ **+31%** — **declined**: non-contracted
  arithmetic is what makes the GPU bitwise-identical to serial.
- Occupancy M → XL: 6.1% → 13.8%; waves/SM 0.02 → 0.25 (explains S13/S14).

NOTES: The profiler reports an estimated +31% FMA optimization opportunity; we
decline it deliberately to preserve bitwise agreement. On the memory/compute
roofline the kernel sits under the compute roof, far from the memory roof.

## S16 — MPI: distributed memory across nodes

Bullets:
- Block decomposition of the outer index; each rank computes its slice against ALL
  positions; per-step `MPI_Allgatherv` (positions only).
- Preserves the inner sum order → **bitwise identical to serial, 1 → 128 ranks**.
- Same 32-way node: MPI 27.5× (86%) vs OpenMP 29.2× (91%) — the gap IS the
  Allgatherv that shared memory avoids → argument for hybrid MPI+OpenMP.
- Communication cost isolated: 32 ranks packed on 1 node = **27.5×** vs spread
  over 4 nodes = **19.2×** (same ranks, same N).

NOTES: The course's MPI day applied. The 1-node-vs-4-node pair at fixed rank count
is the cleanest single number for "what does the network cost".

## S17 — MPI scaling  【figures: reports/mpi_speedup.png + reports/mpi_weak.png】

Table (strong scaling, speedup @ ranks):
| N | 32 | 64 | 128 |
| 2,231 (M) | 27.5× (1 node) | 38.6× | 38.2× — saturated |
| 8,996 (L) | 30.3× | 57.4× | 89.7× (70%) |
| 35,919 (XL) | — | 61.1× (95%) | **113.7× (89%)** |

Bullets:
- Weak scaling (work/rank const): p=4 → 99%, p=16 → 95%, p=64 → **92%**.
- At N=2231, 128 ranks ≈ 17 particles/rank → communication-bound, speedup saturates.
- Scaling ceiling moves with problem size — the MPI twin of the GPU occupancy story.

NOTES: 32/64/128-rank multi-size runs used a 4-node allocation with ranks spread
across nodes (Dragonfly+). Same unifying lesson as S14: performance = keeping every
processing element fed, whether SMs or ranks.

## S18 — Conclusions

Bullets:
- Seven runnable programs: six produce **bitwise-identical state arrays**;
  Numba-CUDA passes the 2·10⁻³ course gate.
- One design rule made that possible: parallelize the outer loop, preserve the
  inner sum order, refuse contraction — an estimated +31% FMA opportunity knowingly declined.
- Performance: **29.2×** (OpenMP, 32 cores) · **31→543×** (CUDA, N-dependent) ·
  **113.7×** (MPI, 128 ranks) · Numba within 6% of hand-written OpenMP.
- One lesson everywhere: speedup = work per processing element (Amdahl ceiling,
  GPU occupancy, MPI ranks — the same curve three times).

NOTES: Close with the FMA anecdote callback: correctness in parallel computing is
a design choice, not luck — and it is measurable.

## S19 — Backup: artifacts

Bullets:
- Repo: `github.com/Leonard013/polimi-phd-school` → `FinalProjects/Particles`.
- Raw evidence: `logs/*.out` (SLURM jobs), `reports/validation.log` (15-pair matrix),
  `reports/*.csv` (every measurement), `reports/ncu/*.ncu-rep`, `reports/nsys/*`.
- Run: `scripts/submit_pipeline.sh`; validate: `run_validate.sh`; regenerate reports:
  `tools/parse_*.py`.
- Run environment details: `LEONARDO_RUN_STATUS.md`.

NOTES: Everything in the talk is regenerable from the committed logs by
`tools/parse_benchmarks.py`, `tools/parse_scaling.py`, `tools/parse_mpi.py`,
`tools/amdahl_fit.py`.
