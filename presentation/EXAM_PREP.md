# Exam prep — the course, the assignment, the project, the questions

Study guide for the oral presentation. Everything here is consistent with the
committed code, logs, and reports. Read top to bottom once, then drill §12–§13.

---

## 1. The course in one page

The Polimi PhD school course is a CINECA-run *parallel computing* school. Its arc:

1. **Intro to HPC + Leonardo** — why parallelism exists (single-core frequency
   scaling ended ~2005; performance now comes from more cores/wider units), and
   how a national supercomputer is used in practice (login vs compute nodes,
   batch scheduler, environment modules).
2. **OpenMP** — *shared-memory* parallelism: one process, many threads on one
   node, communicating through shared RAM. You parallelize loops with pragmas
   (`#pragma omp parallel for`), combine partial results with `reduction`, and
   worry about *race conditions* (two threads writing the same location).
3. **MPI** — *distributed-memory* parallelism: many processes ("ranks"), each
   with private memory, possibly on different nodes, exchanging data with
   explicit messages (point-to-point `Send/Recv`, collectives like `Bcast`,
   `Reduce`, `Allgather`). This is how you scale beyond one node.
4. **GPU computing (CUDA day, OpenACC day)** — offloading data-parallel kernels
   to GPUs: thousands of lightweight threads organized in *blocks* on *streaming
   multiprocessors (SMs)*; separate device memory; host↔device copies; CUDA is
   explicit (you write kernels), OpenACC is directive-based (compiler generates
   them).
5. **Profiling** — measuring instead of guessing: `gprof` (CPU call profile),
   Nsight Systems `nsys` (timeline: kernels, copies, overlap), Nsight Compute
   `ncu` (per-kernel hardware counters), the **roofline model** and **Amdahl's
   law** as the two interpretive frameworks.
6. **GPU Python** — Numba (`@njit` JIT-compiles Python to machine code via LLVM;
   `prange` = OpenMP-like threading; `@cuda.jit` = CUDA kernels from Python) and
   CuPy (NumPy API on GPU).

**Vocabulary you must own:** speedup S(p)=t₁/tₚ · efficiency E=S/p · strong
scaling (fixed problem, more processors) vs weak scaling (fixed work *per*
processor) · Amdahl's law S(p)=1/((1−f)+f/p) where f = parallel fraction ·
race condition · reduction · occupancy · arithmetic intensity (FLOP/byte) ·
memory-bound vs compute-bound · FMA.

---

## 2. The machine: Leonardo, SLURM, modules

- **Leonardo** (CINECA, Bologna) — EuroHPC pre-exascale system. We used the
  **Booster** partition: each node = 1× Intel Xeon Platinum 8358 (32 cores) +
  4× **NVIDIA A100 SXM 64 GB** GPUs (we used 1 GPU/node); nodes connected by a
  **Dragonfly+** InfiniBand network. There is also a CPU-only partition (DCGP).
- **Workflow**: you `ssh` to a *login node* (edit + compile only — never run
  there), load the toolchain with *environment modules*
  (`module load gcc/12.2.0 cuda/12.2 cmake/3.27.9 hdf5/... python/3.11.7`),
  and submit *batch jobs* to the **SLURM** scheduler: a shell script with
  `#SBATCH` headers (partition `boost_usr_prod`, QoS `boost_qos_dbg` for short
  debug jobs, `--gres=gpu:1` to request a GPU, `--account=tra26_poliex` = the
  budget), submitted with `sbatch script.sh`; output lands in `logs/`.
- `srun -n P ./exe` inside a job launches P MPI ranks. Multi-node jobs allocate
  `--nodes=4 --ntasks-per-node=32` → up to 128 ranks.
- Build system: CMake *presets* — `cmake --preset leonardo-a100` pins CUDA
  architecture `sm_80` (= A100, "compute capability 8.0").

---

## 3. The assignment (what the exam requires)

From `FinalProjects/README.md`: pick **Cooling** (2D stencil solver) or
**Particles** (chosen) and:

1. Describe the original serial application.
2. **Preserve the numerical model** (force law, all-pairs interactions, input
   format, output quantities) — explicitly: "the force law and the inclusion of
   all pair interactions must be preserved". So *no* Barnes-Hut/approximations.
3. Parallelize with course technologies (OpenMP / MPI / CUDA / OpenACC / Numba —
   one or more).
4. **Validate** against the serial baseline (small FP differences from reduction
   order/FMA are declared acceptable; course tools compare HDF5 outputs at
   rtol=atol=2·10⁻³).
5. **Benchmark** reproducibly: same input, I/O disabled during timing, report
   hardware/compiler/flags/threads/ranks, repeat runs.
6. Show **speed-up and efficiency plots**; discuss strong/weak scaling.
7. ~20 h workload; present and defend at the exam.

The repo *scaffold* already expects the port filenames: CMake builds
`particles_omp` from `src/cpp/particles_omp.cpp` and `particles_cuda` from
`src/cpp/particles.cu` *if those files exist* — creating them **was** the
assignment. Submit scripts (`submit_omp.sh`, …) and a validator
(`tools/validate_particles_h5.py`, driven by `scripts/validate_all.sh`) were
provided.

---

## 4. The application, explained end-to-end

One executable, five phases (identical in C++ `particles.cpp` and Python
`particles.py`):

1. **Read input** (`input/Particles.in`): generating-grid size **1200×1000**,
   a window of the complex plane, screen grid, `maxFractalIterations=3000`,
   `timeSteps=200`, `dt=0.001`, `outputEvery=20`.
2. **Mandelbrot generating field**: for every grid cell, iterate z←z²+c from
   z=0 (c = cell's complex coordinate); count iterations until |z|²>4, capped at
   3000. Result: a 1200×1000 field of iteration counts. (Embarrassingly
   parallel; it's the standard course fractal example.)
3. **Particle generation**: threshold = ⌊(29·vmax+vmin)/30⌋ (≈ "cells that
   nearly never escaped", i.e. near the Mandelbrot set boundary). Each such cell
   becomes a particle: position = cell coordinates mapped to a 1200×1000 world,
   **weight w = max(1, 10·count)** — and *weight is the mass*. Velocities start
   at 0. Official input → **N = 2231** particles. The *scan order* (row-major,
   j outer, i inner) fixes each particle's index — part of the output contract.
4. **Dynamics — the hot loop.** Force on particle i:
   F⃗ᵢ = Σ_{j≠i} kForce·wᵢwⱼ·(r⃗ⱼ−r⃗ᵢ)/(|r⃗ⱼ−r⃗ᵢ|²+ε²)^{3/2},
   with kForce=10⁻³, ε²=10⁻⁴. It is **attractive** (points toward j),
   gravity-like, with **Plummer softening**: the +ε² prevents the 1/r²
   singularity when particles overlap. All-pairs ⇒ **O(N²)** per step; with
   N=2231 and 200 steps ≈ **10⁹ pair interactions** per run.
   Integration = **velocity-Verlet** (kick-drift-kick form):
   v += ½·(F/m)·dt → x += v·dt → recompute F → v += ½·(F_new/m)·dt.
   Second-order, symplectic (good long-term energy behaviour), time-reversible.
5. **Validation & output**: 11 printed quantities (Σx, Σy, Σvx, Σvy, weighted
   sums, momentum Σw·v, kinetic energy ½Σw·v², a *potential-like* sum
   Σ_{i<j} k·wᵢwⱼ/√(r²+ε²), and their combination "energy_like") + optional
   **HDF5** file with datasets `/pos` (frames,N,2), `/vel`, `/step`, `/weight`,
   `/screen`. Momentum is physically conserved (internal, antisymmetric forces;
   starts at 0) so it stays ~0 — its printed digits are catastrophic-cancellation
   noise (~10⁻⁸ against 10¹¹-scale intermediate sums). "energy_like" is a
   *validation aid*, not a conserved physical energy (the potential term has the
   wrong sign convention for that) — call it a checksum with physical flavour.
   Performance metric printed: **GInteractions/s** = N·(N−1)·steps / time / 10⁹.

**Benchmark mode**: `./particles input/Particles.in none 0` — `none` disables
HDF5, `0` = final output only ⇒ timing excludes all I/O.

---

## 5. The one design rule (and why it's the thesis of the talk)

Floating-point addition is **not associative**: (a+b)+c ≠ a+(b+c) in rounded
arithmetic. Parallel programs usually change summation *order* (each thread sums
a chunk, partials combined) ⇒ results differ from serial in the last bits, and
those differences **compound over 200 chaotic-ish timesteps**. The course
accepts this (2·10⁻³ tolerance). We aimed higher:

> **Parallelize only ACROSS output elements; never change the order WITHIN one
> output element's accumulation.**

The force loop is `for i { for j { fx[i] += … } }`. Thread/GPU-thread/rank i
computes fx[i] by running j = 0…N−1 **in the original serial order**. No shared
writes (each i is owned by exactly one worker) → no races; same order → **the
same bits**. Two corollaries:

- **No pair-symmetry optimization** (Newton's 3rd law would halve the FLOPs by
  doing `f[i]+=…; f[j]-=…`): it creates concurrent writes to f[j] (needs atomics
  = slow + order-nondeterministic) and reorders j's accumulation. Also the
  baseline doesn't use it, and the assignment freezes the numerical model.
- **No fast-math, no FMA contraction.** Compilers may fuse `a*b+c` into one FMA
  instruction with a *single* rounding — more accurate, but *different* from
  mul-round-add-round. We compile without fast-math and, in CUDA (where nvcc
  contracts by default), use explicit round-to-nearest intrinsics
  (`__dmul_rn`, `__dadd_rn`) that the compiler cannot fuse.

**The FMA war story (tell it — it's the best 60 seconds of the talk).** During
local verification on a Mac, the *provided baseline itself* produced **2232**
particles in C++ vs **2231** in Python on the official input. Root cause: clang
contracts FMAs by default in the Mandelbrot loop; the iteration is chaotic, so a
1-ulp difference grows until 14 of 1.2 M cells get different iteration counts —
and exactly one cell (i=1157, j=489: 3000 fused vs 2654 unfused iterations)
crosses the 2900 selection threshold. One fused instruction ⇒ one extra particle
⇒ the datasets don't even have the same *shape*. Proof: recompiling with
`-ffp-contract=off` makes C++ bitwise-equal to Python. On Leonardo (gcc 12 +
nvcc) all seven runnable programs agree at N=2231, so the exam numbers are clean —
but the anecdote motivates the whole reproducibility discipline, and the
profiler later estimates a **+31% FMA optimization opportunity** that we decline (§9).

---

## 6. The five ports (what I actually wrote)

Each is a complete standalone program (own `main`), sharing the baseline's I/O,
validation, and HDF5 code verbatim; only the hot loops change.

**OpenMP — `src/cpp/particles_omp.cpp`** (shared memory, 1 node)
`#pragma omp parallel for schedule(static)` on: force outer loop, both Verlet
half-kick loops, Mandelbrot rows (`collapse(2)`). Validation sums use
`reduction(+:…)` clauses (including the O(N²) potential loop). `schedule(static)`
= contiguous equal chunks — right choice because iterations have identical cost
(no load imbalance) and it's cache-friendly and deterministic. Threads pinned:
`OMP_PROC_BIND=close, OMP_PLACES=cores`. Bitwise vs serial at every thread count.

**CUDA — `src/cpp/particles.cu`** (1 GPU)
One **thread per particle i**, 128-thread blocks, grid = ⌈N/128⌉. The inner
j-loop runs serially *inside* the thread, accumulating in registers. All particle
arrays (SoA: separate x, y, vx, vy, w arrays — coalesced access, SIMD-friendly)
live **resident in GPU memory across all 200 steps**; the host only copies back
for optional HDF5 frames and the final validation (9 memcpys, 0.19 ms total —
nsys-verified). Three kernels: force, half-kick+drift, half-kick; force-buffer
*pointer swap* instead of copying. Every arithmetic op in the model uses
`__d*_rn` intrinsics ⇒ bitwise-stable regardless of `--fmad`. `CUDA_CHECK` macro
+ `cudaGetLastError`/`cudaDeviceSynchronize` after each launch. Built for sm_80.
No shared-memory tiling — deliberate: profiling shows the kernel is compute-bound
with the whole working set (~90 KB) in L1/L2; tiling would add complexity for no
bandwidth win (§9 has the numbers to defend this).

**MPI — `src/cpp/particles_mpi.cpp`** (distributed memory, up to 4 nodes)
**Block decomposition** of the particle index: rank r owns a contiguous slice of
i's; computes forces for its slice against **all** N positions; integrates its
slice; then a per-step **`MPI_Allgatherv`** redistributes the updated positions
(x and y) to everyone. (Allgatherv, not Allgather, because N=2231 doesn't divide
evenly.) Weights never change → broadcast once. Since each fx[i] is still one
rank's serial j-loop, results are **bitwise identical to serial for any rank
count** — verified 1→128. Rank 0 does I/O/validation after gathering velocities.

**Numba CPU — `src/python/particles_numba.py`**
Hot loops moved to module-level `@njit(parallel=True, fastmath=False, cache=True)`
functions on plain NumPy arrays; `numba.prange` on the outer i. Numba JIT-compiles
through LLVM — effectively "OpenMP written in Python". First call compiles
(warm-up excluded from timings — we time the reported *pure dynamics* counter).
Bitwise vs the Python baseline *and* vs serial C++.

**Numba CUDA — `src/python/particles_numba_cuda.py`**
Same kernel structure as the CUDA port via `@cuda.jit`, float64 everywhere,
`cuda.to_device` once, device-resident across steps. Here we can't control FMA
contraction as surgically as with C++ intrinsics ⇒ the one non-bitwise port
(max relative velocity difference **1.158·10⁻³**, below the 2·10⁻³ course gate;
position max absolute difference **4.56·10⁻⁵**).

*(Why no OpenACC port? Course offers it as the directive-based alternative to
CUDA. We chose explicit CUDA because bitwise reproducibility needed instruction-
level control (`__d*_rn`), which directives don't expose. Legitimate, defensible.)*

---

## 7. Validation: method and results

**Method** (two layers):
1. Every implementation writes the same HDF5 layout; the course validator
   compares `/step /weight /pos /vel` element-by-element over all frames (11
   frames: steps 0,20,…,200). Course gate: `np.isclose` with rtol=atol=2·10⁻³.
   We ran the **full 15-pair matrix** across C++ serial, OpenMP, CUDA, Python,
   Numba CPU, and Numba CUDA. MPI was checked separately against C++ serial.
   Expected bitwise pairs also received a **strict pass at rtol=10⁻¹²**;
   Numba CPU vs Numba CUDA is an explicitly informational strict check.
2. The 11 printed validation quantities compared at 17 significant digits.

**Results (Leonardo A100, official input) — memorize these:**
- **15/15 pairs PASS** the six-way course gate; MPI passes its separate comparison.
- C++ serial ≡ OpenMP ≡ CUDA ≡ MPI ≡ Python ≡ Numba CPU: **bitwise identical**
  (validator max_abs_diff = **0**). CUDA matches serial in *all 11 printed
  quantities to all 17 digits*.
- MPI bitwise from 1 to 128 ranks.
- Numba-CUDA: max relative velocity difference **1.158·10⁻³** and max absolute
  position difference **4.56·10⁻⁵** (passes 2·10⁻³; fails the strict pass).
- Caveat to volunteer if asked about the printed blocks: `momentum_x/y` are ≈0
  by conservation, computed as catastrophic cancellation → their digits are
  noise; OpenMP/NumPy print-level differences (~10⁻¹⁵ relative) in some
  aggregates come from the *validation reductions themselves*, while the
  underlying trajectories are bit-identical.

---

## 8. Benchmarks: method and every number

**Method** (deliverable #4): Leonardo Booster node — Xeon 8358 (32 c) + A100
SXM 64 GB (108 SMs, driver 535.274.02); gcc 12.2.0, CUDA 12.2, `-O3 -std=c++17`
(+`-fopenmp` / nvcc sm_80), **no fast-math, no -march=native**; HDF5 **disabled**
during timing (`none 0`); metric = the program's own "pure dynamics" timer
(excludes setup, I/O, JIT warm-up); **median of 3 repetitions**; pinned threads.
Serial reference: **3.516 s / 0.283 GInt/s**, flat across problem sizes.

**OpenMP strong scaling (N=2231):**
| threads | 2 | 4 | 8 | 16 | 32 |
| speedup | 2.00× | 3.99× | 7.97× | 15.31× | **29.20×** |
| efficiency | 100% | 100% | 100% | 96% | **91%** |

**Amdahl fit** (linearized least squares on 1/S vs 1/p): serial fraction ≈
**0.21%** ⇒ ceiling ≈ **480×**; measured points within ~3% of the model. (Quote
it exactly like that; the 16/32-thread dip is shared memory subsystem, not code.)

**All models at the official size (N=2231):**
| model | GInt/s | speedup |
| serial | 0.283 | 1× |
| OpenMP-32 | 8.31 | 29.2× |
| CUDA A100 | 8.80 | **31.1×** |
| Numba CPU | 7.83 | 27.5× |
| Numba CUDA | 3.42 | 11.9× |
| pure Python | 0.00128 | **221× slower** than serial C++ |

**Problem-size scaling (same fractal window, finer grids):**
| N | 567 | 2,231 | 8,996 | 35,919 | 143,768 |
| omp-32 GInt/s | 5.78 | 8.37 | 8.39 | 8.44 | 8.07 (saturated ~29×) |
| CUDA GInt/s | 1.78 | 8.86 | 36.8 | 105.4 | **153.9** |
| CUDA speedup | 6.3× | 31× ←crossover | 130× | 372× | **543×** |

Story: serial is size-independent; the CPU saturates; the GPU *climbs with
occupancy* — the official size sits exactly at the CPU/GPU crossover, which is
why CUDA "only ties" omp-32 in the headline table.

**MPI:**
- Single node, 32 ranks: **27.5× (86%)** vs OpenMP's 29.2× (91%) — the gap *is*
  the per-step Allgatherv that shared memory avoids ⇒ textbook argument for
  hybrid MPI+OpenMP.
- Communication cost isolated: same 32 ranks, same N — packed on 1 node
  **27.5×** vs spread over 4 nodes **19.2×**.
- Strong scaling across nodes (ranks spread over a 4-node allocation):
  N=2231: 38× at 64 ranks, **saturates ~38×** at 128 (only ~17 particles/rank —
  communication-bound). N=8996: 89.7× @128 (70%). N=35919: **113.7× @128 (89%)**.
- **Weak scaling** (work/rank constant — note: work is O(N²), so N ∝ √p):
  p=4 → 99%, 16 → 95%, 64 → **92%**.

---

## 9. Profiling: what nsys/ncu proved

- **nsys (timeline), official input**: `computeForcesKernel` = **99.1%** of GPU
  time (201 launches — 200 steps + initial force — ~610 µs each); integration
  kernels 0.9%; total `cudaMemcpy` **0.19 ms in 9 calls** ⇒ transfers are
  negligible ⇒ CUDA *streams*/async overlap (a course topic) would buy nothing
  here — right tool only when transfer time ≈ compute time.
- **ncu (kernel counters)**, M = official vs XL = 35,919:
  | | M | XL |
  | blocks (on 108 SMs) | 18 | 281 |
  | achieved occupancy | 6.1% | 13.8% |
  | SM throughput | 4.5% | 53.5% |
  | DRAM throughput | ~0% | ~0% |
  At M the GPU is ~95% idle — *launch geometry*, not code quality. At XL:
  **compute-bound** — DRAM ≈ 0 because the whole working set (5 arrays × N × 8 B
  ≈ 90 KB at M, ~1.4 MB at XL) lives in L1/L2 ⇒ arithmetic intensity is
  effectively huge ⇒ on the **roofline** the kernel sits under the *compute*
  roof, far right of the ridge point.
- Hottest pipe: **FP64 at 63%**; achieved ≈ **37% of A100 FP64 peak** (9.7
  TFLOP/s non-tensor). ncu's own optimizer note: fusing the 686 M non-fused FP64
  ops into FMAs would gain ≈ **+31%** — **we declined**, because non-contracted
  arithmetic is exactly what makes the GPU bitwise-identical to serial. That's
  the profiler-quantified *price of reproducibility* — say this sentence.
- **Unifying lesson** (close the talk with it): Amdahl ceiling, GPU occupancy,
  and MPI saturation are the *same curve* — performance = keeping every
  processing element fed with enough work.

---

## 10. Likely questions — model answers (drill these)

**Theory**

- *Why aren't parallel results usually bitwise?* FP addition isn't associative;
  parallel reductions change summation order, so last-bit differences appear and
  compound over timesteps. We avoided it by never changing the order within any
  single accumulated value — parallelism only across independent outputs.
- *What is a race condition? Where could one occur here?* Concurrent unsynchronized
  writes to the same memory. Would occur if we exploited Newton's 3rd law
  (`f[j] -= …` from thread i). We don't — each f[i] has exactly one writer.
- *Why not halve the work with Newton's 3rd law?* Creates write races (needs
  atomics), changes summation order (breaks bitwise), and the assignment freezes
  the numerical model ("all pair interactions must be preserved"). The 2× would
  also not change the *scaling* story, only the constant.
- *What is FMA?* Fused multiply-add: a·b+c in one instruction with a **single**
  rounding of the exact intermediate. More accurate but different from
  mul-round-then-add-round. Compilers contract by default; in chaotic code
  (Mandelbrot) a 1-ulp change can flip discrete outcomes — our 2232-vs-2231
  particle story.
- *Amdahl vs Gustafson?* Amdahl: fixed problem, speedup bounded by 1/serial-
  fraction (ours: 0.21% ⇒ ~480×). Gustafson: scale the problem with p — the
  serial fraction shrinks relatively; that's the weak-scaling view, and our weak
  scaling holds 92% at 64 ranks.
- *Strong vs weak scaling; why N ∝ √p in your weak scaling?* Strong: fixed total
  work. Weak: fixed work per processor. Our work is O(N²), so constant N²/p
  requires N ∝ √p — that's why the weak-scaling inputs grow like 1698×1414 etc.
- *Define speedup/efficiency.* S=t_serial/t_p; E=S/p. Report both; efficiency
  exposes the knee.
- *Why velocity-Verlet and not Euler/RK4?* Symplectic + time-reversible ⇒ bounded
  long-term energy drift at 2nd order with one force evaluation per step (the
  expensive part). RK4 = 4 force evaluations/step for non-symplectic behaviour.
- *Is energy conserved in your runs?* Momentum is (internal antisymmetric forces)
  — printed ≈0. The printed "energy_like" is a *validation checksum*, not the
  physical energy (its potential term has the opposite sign convention), so it
  isn't a conserved quantity; correctness is asserted by trajectory comparison,
  not conservation.
- *Why the ε² softening?* Removes the 1/r² singularity at close encounters
  (Plummer softening) — keeps forces finite and dt=10⁻³ stable.
- *Could you do better than O(N²)?* Yes in general: cell lists / Barnes-Hut
  O(N log N) / FMM O(N). Not allowed here — they *approximate* the force sum,
  and the assignment requires preserving all pair interactions. Would be the
  natural next step if the model were free.

**GPU**

- *Why is the GPU barely faster than 32 CPU cores at the official size?*
  N=2231 → 18 blocks of 128 threads on 108 SMs → 6.1% occupancy; the GPU is 95%
  idle. It's launch geometry, not code: at N=143k the same kernel does 153.9
  GInt/s = 543× serial and 19× the full CPU socket.
- *What is occupancy?* Ratio of resident warps to the SM's maximum — a measure
  of how much parallelism the hardware has available to hide latency.
- *Memory-bound or compute-bound? Prove it.* Compute-bound: ncu shows DRAM ≈ 0%
  (working set fits in L1/L2) while the FP64 pipe runs at 63%; on the roofline
  we sit under the compute roof. So shared-memory tiling would not help — that's
  why the kernel deliberately has none.
- *Why 128 threads/block?* Multiple warps for latency hiding, small enough for
  good register allocation (52 regs/thread); block size is not the bottleneck —
  total thread count is (occupancy argument).
- *Why don't you use CUDA streams?* Streams overlap copies with compute; nsys
  shows total copies = 0.19 ms vs ~120 ms kernel time — nothing to overlap.
- *Why are results on GPU bitwise? Isn't GPU arithmetic different?* A100 FP64 is
  IEEE-754 compliant; differences normally come from *order* and *contraction*.
  We fixed the order (serial inner loop per thread) and blocked contraction
  (`__d*_rn` intrinsics) ⇒ same bits.
- *What did reproducibility cost?* ncu estimates that enabling FMA could improve
  performance by ≈ +31%; we deliberately decline that optimization opportunity.
- *Why is Numba-CUDA slower than CUDA C++?* Python-side launch overhead ~each
  step and less tuned codegen (NVVM); at small N launch overhead dominates. It's
  also the one port not bitwise (can't control contraction as precisely).

**MPI**

- *How is work distributed?* Contiguous block of particle indices per rank
  (equal cost per particle ⇒ static blocks are balanced). Each rank needs *all*
  positions (all-pairs) → per-step `MPI_Allgatherv` of x,y.
- *Why Allgatherv, not Allgather?* 2231 doesn't divide evenly by the rank count;
  the "v" variant takes per-rank counts/displacements.
- *Why does MPI saturate at 128 ranks on the official input?* 2231/128 ≈ 17
  particles/rank: per-step compute shrinks to ~microseconds while the Allgatherv
  latency across 4 nodes is fixed ⇒ communication-bound; speedup flatlines ~38×.
  At N=35919 the same 128 ranks give 113.7× (89%) — the ceiling moves with work
  per rank.
- *MPI vs OpenMP on one node?* 27.5× vs 29.2× at 32-way: the difference is the
  explicit Allgatherv vs free shared memory ⇒ motivates hybrid MPI(between
  nodes)+OpenMP(within node).
- *What does '32 ranks spread over 4 nodes = 19.2×' show?* Same ranks, same N,
  only the interconnect changed: 27.5→19.2 is the isolated cost of inter-node
  communication.
- *Is MPI also bitwise? Why?* Yes, 1→128 ranks: block decomposition doesn't
  touch the inner j-order; Allgatherv moves bytes, it doesn't do arithmetic.

**Methodology / tools**

- *Why medians of 3? Why exclude JIT warm-up and I/O?* Robustness to scheduler
  noise; warm-up is one-time compilation, not the algorithm; I/O measured
  separately (assignment explicitly says benchmark with HDF5 off).
- *What exactly is GInteractions/s?* N·(N−1)·steps / time / 10⁹ — a rate, so runs
  with different step counts are comparable.
- *What tolerance and why?* Course default rtol=atol=2·10⁻³ via np.isclose;
  we additionally show 10⁻¹² and exact-bit agreement where achieved.
- *What's in the HDF5 file?* `/pos` (frames,N,2), `/vel`, `/step` (frame→step
  map), `/weight` (N), `/screen` (visualization only, excluded from grading
  comparisons); chunked datasets, one frame per chunk.
- *How would you rerun everything?* Build first, then use
  `bash scripts/submit_pipeline.sh` for the C++/Numba jobs and
  `sbatch submit_python_ref.sh` for the full matrix. After completion, run
  `bash run_validate.sh`, then regenerate tables and plots with `tools/parse_*.py`.

**Curveballs**

- *You used AI/tools? / How was the work organized?* Answer honestly per your
  policy; the defensible core is that you can explain every design decision and
  every number — which is what this guide is for.
- *What was the hardest part?* Making parallel execution *reproducible*, not just
  fast — and discovering the baseline itself wasn't (FMA particle-count story).
- *What would you do next?* Hybrid MPI+OpenMP; multi-GPU (one rank per GPU, same
  Allgatherv pattern); if the model were free: Barnes-Hut + comparison of
  accuracy/performance trade-off; occupancy tuning (multiple particles per
  thread at small N).
- *Why is pure Python 221× slower?* Interpreter overhead per loop iteration
  (dynamic dispatch, boxing); Numba removes it by JIT-compiling the same loop to
  machine code — landing within 6% of hand-written OpenMP.

---

## 11. Numbers card (memorize cold)

- N = **2231**, 200 steps, dt=10⁻³, ~**10⁹** interactions/run; grid 1200×1000, 3000 iters.
- Serial: **3.516 s** = **0.283 GInt/s** (flat in N).
- OpenMP-32: **29.2× (91%)** · Amdahl serial **0.21%** ⇒ ceiling **≈480×**.
- CUDA: **31.1×** @N=2231 (6.1% occupancy) → **543×** (153.9 GInt/s) @N=143,768.
- Kernel: **99.1%** of GPU time; copies 0.19 ms; FP64 pipe **63%**; ≈**37% of FP64
  peak**; FMA would add **+31%** — declined.
- MPI: 1 node 32r **27.5× (86%)**; spread 4 nodes same 32r **19.2×**; 128r: M
  saturates **38×**, XL **113.7× (89%)**; weak 99/95/**92%**.
- Numba **27.5×** ≈ OpenMP; Numba-CUDA **11.9×**, max rel vel **1.158·10⁻³**;
  pure Python
  **221× slower**.
- Validation: **15/15 PASS** in the six-way matrix; **six state outputs bitwise**
  (including MPI); CUDA = serial to 17 digits; MPI bitwise 1→128 ranks.
- Platform: Leonardo Booster, Xeon 8358 32c + **A100 SXM 64 GB** (108 SMs),
  gcc 12.2.0 + CUDA 12.2, sm_80, `-O3`, no fast-math, Dragonfly+.

## 12. Repo map (if asked to show something)

`src/cpp/particles.cpp` baseline · `particles_omp.cpp` · `particles.cu` ·
`particles_mpi.cpp` · `src/python/particles{,_numba,_numba_cuda}.py` ·
`tools/validate_particles_h5.py` + `parse_*.py` + `amdahl_fit.py` ·
`reports/` (all CSVs, PNGs, validation.log, ncu/nsys reports) · `logs/` (raw
SLURM output) · `LEONARDO_RUN_STATUS.md` (full run record) ·
`presentation/PRESENTATION.md` (slide spec). Repo:
`github.com/Leonard013/polimi-phd-school`, path `FinalProjects/Particles`.
