# 08 — The Particles project through the course lens

Every design decision and every result in this project instantiates something the
course taught. This chapter walks the project module by module: *concept as taught
(with slide refs into the decks)* → *where it appears in the project (file / report)*.
Read it after the module chapters; it is the strongest preparation for "why did you
do X?" questions, because the answer is always "because the course showed Y".

Slide references use each chapter's convention: `[M1 s38]` = Module 1 chapter, slide 38
of its deck, etc.

---

## 1. From Module 1 (Intro HPC & Leonardo)

**SPMD [M1 s22].** All five parallel ports use data-parallel, SPMD-style execution:
workers run the same kernel and distinguish themselves by an index (OpenMP/Numba
thread → loop chunk, CUDA/Numba-CUDA `blockIdx/threadIdx` → particle, MPI rank →
particle block). Nothing in the project is task-parallel; parallel work is pure
**data parallelism** [M1 s24] over the particle index.

**Memory-bound vs compute-bound [M1 s9, s41].** The course's recap says "HPC
applications are typically memory-bound". Our force kernel is the notable exception —
`ncu` measured DRAM throughput ≈ 0% with the FP64 pipe at 63% [reports/ncu/force_XL.txt]:
the whole working set (5 arrays × N × 8 B ≈ 90 KB at N=2231, ~1.4 MB at XL) fits in
L1/L2, so the kernel is **compute-bound**. Being able to *prove* which side of the
divide you are on (rather than assume) is exactly what Module 7's tools are for.

**Speedup, efficiency, Amdahl [M1 s36-38].** The formulas `S(Nₚ)=tₛ/t(Nₚ)`,
`E=S/Nₚ`, `S=1/[(1−P)+P/Nₚ]` are computed in `tools/parse_benchmarks.py` and
`tools/amdahl_fit.py`. The Amdahl fit (linearized 1/S vs 1/p) gives serial fraction
≈ 0.21% → ceiling ≈ 480×; measured OpenMP points sit within ~3% of the model
[reports/amdahl_omp.png]. The course's statement "the serial fraction caps speedup for
a fixed problem" is our plot.

**Strong vs weak scaling [M1 s36, s39].** We ran both. Strong: fixed official input,
threads 1→32 (OpenMP) and ranks 1→128 (MPI). Weak: constant work per rank — and
because our work is O(N²), constant N²/p requires **N ∝ √p**, which is why the weak
inputs are W2=1698×1414, L, W8=3394×2828 [input/scaling/, run_mpi_all.sh]. The course
recommends weak scaling for memory-bound apps; ours is compute-bound, but the weak test
still shows the textbook behavior: 99/95/92% efficiency because the communication
fraction stays bounded when per-rank work is fixed.

**The machine [M1 Part B].** Benchmarks ran on the Booster node the deck specifies
[M1 s6-7]: 1× Xeon Platinum 8358 (32 cores), 4× A100-64GB (we used 1), Dragonfly+
at 200 Gb/s [M1 s9/s17] — the network our 4-node MPI runs crossed. Submit scripts use
exactly the SLURM grammar of [M1 s19-21]: `--partition=boost_usr_prod`,
`--qos=boost_qos_dbg` (30-min debug QoS), `--gres=gpu:1`, `--account=tra26_poliex`,
logs `%x_%j.out`. The gotcha taught on [M1 s19] (dbg QoS incompatible with
`--exclusive`) is why `--exclusive` was removed from the scripts on Leonardo.

**Static vs dynamic scheduling [M1 s32].** Workload per particle is identical and
known a priori → static distribution everywhere (OpenMP `schedule(static)`, MPI
contiguous blocks). The course's criterion — static when workload is known, dynamic
when not — answers "why not `schedule(dynamic)`?" directly.

---

## 2. From Module 2 (OpenMP)

**`parallel for`, not `parallel` [M2 s27, s32].** The deck's "ACHTUNG: `parallel`
executes redundantly" trap is why every hot loop in `particles_omp.cpp` uses the
combined `#pragma omp parallel for`. The iteration-independence requirement [M2 s32]
holds by construction: iteration i writes only `fx[i]`, `fy[i]`.

**Race conditions [M2 s24-25, s48-49].** The four-condition definition (same location,
concurrent, ≥1 write, unsynchronized) is why we rejected the Newton's-third-law
optimization: `f[j] -= coeff*dx` from thread i is a textbook data race on `f[j]`,
fixable only with atomics (slow) or coloring (reorders sums). The project note "no
pair-symmetry" is the deck's race lesson applied.

**`reduction`, and why never `critical`/`atomic` in hot code [M2 s50-61].** The deck's
own π benchmark — serial 0.14 s, `critical` 10.60 s, `atomic` 8.34 s, `reduction`
0.04 s — is the entire justification for our validation sums using
`reduction(+:...)` / `reduction(max:...)` clauses and nothing else. In the force loop
itself no reduction is needed at all (each thread owns its output element — the best
synchronization is none).

**Data scoping [M2 s37-45].** Arrays are `shared` (defined outside the region — the
default rule), loop iterators private by default; scalar temporaries are declared
inside the loop body, making them private by the "defined inside" rule. This is the
deck's scoping table in action.

**`schedule(static)` [M2 s72-75].** Uniform-cost iterations → static = lowest overhead
and deterministic chunking (also cache-friendly). Matches the deck's best practice
"use static for generic/balanced loops"; our SAXPY-like uniformity makes it optimal.

**`collapse(2)` [M2 s70].** Used on the Mandelbrot generation loops (grid rows ×
columns) to expose ny×nx iterations instead of ny — the deck's remedy for nested
loops, avoiding the disabled-nested-parallelism trap [M2 s69].

**Pinning.** `OMP_PROC_BIND=close, OMP_PLACES=cores` in the benchmark scripts — the
run-to-run stability discipline (the chapter notes affinity was not covered in this
deck; it comes from the benchmark-reproducibility requirement of the assignment).

**Result:** 29.20× on 32 threads, 91% efficiency, bitwise identical to serial at
every thread count [reports/summary.md, validation.log].

---

## 3. From Module 3 (MPI)

**The model [M3 s82-83, s88].** `particles_mpi.cpp` is the deck's distributed-memory
SPMD picture: every rank runs the same binary, owns private memory, and all data
movement is explicit. The deck's banner — **"minimize message passing"** — shaped the
design: one collective per timestep, positions only (velocities and weights never
travel; weights are broadcast once at startup with `MPI_Bcast` [M3 s139]).

**Which collective — and why `Allgatherv` [M3 s145-149].** The deck teaches
`MPI_Allgather` with the matrix-vector example: each process owns a slice of rows but
needs *the whole vector x*. Our force computation is the same shape: each rank owns a
slice of particles but needs *all positions* to compute its forces. Hence per-step
`MPI_Allgatherv` of x and y. The `v` variant because N=2231 does not divide evenly by
the rank count — exactly the unequal-blocks case the deck gives for `Scatterv/Gatherv`
[M3 s149].

**No deadlocks by construction [M3 s126-134].** The deck's deadlock taxonomy
(recv-first, misaligned tags, send-first-unbuffered) applies to point-to-point code.
We use collectives only, so the entire failure class is designed out — the defensible
answer to "how do you know it can't deadlock?" (Collectives' own rule [M3 s136] —
every rank must call — is satisfied trivially since all ranks execute the same loop.)

**Timing idiom [M3 s157-158].** The benchmark harness times the way the deck
prescribes: synchronize, measure per-rank, report the slowest (the program's own
"pure dynamics" timer on rank 0 after collective completion serves this role; medians
over 3 reps per the assignment's repetition requirement).

**The scaling model — our saturation is the deck's table [M3 s159-161].** The deck
models `t_parallel = t_serial/p + t_overhead(comm)` and shows efficiency ≈0.15 at
(p=16, n=1024) vs ≈0.97 at n=16384. Our measurements reproduce it exactly one level
up: at N=2231, 128 ranks = ~17 particles/rank → speedup saturates at ~38× (30%
efficiency); at N=35919 the same 128 ranks give 113.7× (89%)
[reports/mpi_summary.md]. Small-n-large-p is communication-bound; the ceiling moves
with problem size.

**Collectives dominate scaling [M3 s162] + hybrid tip [M3 s163].** Single-node 32-way:
MPI 27.5× (86%) vs OpenMP 29.2× (91%) — the gap *is* the per-step Allgatherv that
shared memory gets for free. That is the deck's closing advice ("use hybrid MPI/OpenMP
to reduce ranks in collectives") demonstrated with numbers; hybrid is our stated
future work. The packed-vs-spread experiment (same 32 ranks, 1 node = 27.5× vs
4 nodes = 19.2×) isolates the pure network cost of the same collective.

**Bitwise across ranks.** Block decomposition never touches the inner j-order:
each fx[i] is one rank's serial loop over the gathered positions; `Allgatherv` moves
bytes, it does no arithmetic. Hence bitwise-identical to serial from 1 to 128 ranks —
a property most MPI codes cannot claim (their reductions reorder sums; we have no
inter-rank reductions in the dynamics).

---

## 4. From Module 4 (CUDA)

**One thread per output element [M4 s74, s117].** `computeForcesKernel` is the deck's
canonical mapping: `i = blockIdx.x*blockDim.x + threadIdx.x`, bounds guard
`if (i >= N) return`, one particle per thread, inner loop in registers. Launch uses
the deck's round-up formula `(N + 128 − 1)/128` blocks of 128 threads — a multiple of
32 per the warp rule [M4 s56].

**The five-step pattern, minimized [M4 s60-62].** Allocate → copy in → kernel → copy
out → free. Our refinement: steps 2 and 4 happen once, not per step — the SoA state
is device-resident for all 200 steps, with D2H copies only for optional HDF5 frames
and the final validation. `nsys` confirms 9 memcpys totalling 0.19 ms against ~120 ms
of kernel time [reports/nsys/]. This is the CUDA translation of Module 5's central
lesson (keep data resident; see §5).

**Occupancy and latency hiding [M4 s14-17, s46].** The deck: an A100 SM hosts 64
warps/2048 threads, and the scheduler hides latency by switching among *resident*
warps. At N=2231 we launch 18 blocks of 128 threads = ~70 warps **for 108 SMs** —
less than one warp per SM. There is nothing to switch to; occupancy 6.1%, SM
throughput 4.5% [reports/ncu/force_M.txt]. At N=143k the same kernel reaches 153.9
GInt/s (543× serial). The deck's oversubscription picture [M4 s95] — queue far more
blocks than SMs — is precisely what the official input fails to provide and the
scaling study restores.

**SIMT and divergence [M4 s44].** The kernel's only branch is `if (i != j)` — uniform
across the warp except for the single warp containing j... actually per-thread j is
the loop variable, so the branch skips one iteration per thread at different times;
its cost is negligible against the ~N-iteration loop body, and there is no
data-dependent divergence. Coalescing [M4 s99]: SoA layout means thread i reads
`x[j]` — all threads in a warp read the *same* `x[j]` at the same time (a broadcast,
served by cache), and write contiguous `fx[i]` — the friendly pattern.

**Why no shared-memory tiling [M4 s154-157].** The deck's tiling criterion: stage
tiles in shared memory when re-reads of *global* memory make you bandwidth-bound. Our
ncu data shows DRAM ≈ 0% — the reuse is already served by L1/L2 (working set ≪ 40 MB
L2). Tiling would add complexity to move a bottleneck we do not have. The roofline
confirms it: the kernel sits under the *compute* roof (§7), where the deck's own
decision rule says tiling is the wrong lever.

**Why no streams [M4 s141-158].** The deck's criterion [M4 s158]: streams help when
transfer time is comparable to kernel time. Ours: 0.19 ms vs ~120 ms. Nothing to
overlap. (Same verdict from Module 7's Mandelbrot example, §7.)

**Pinned/unified memory [M4 s64-70, s135-140].** Not on the critical path for the same
reason — transfers are already negligible. The HDF5 output path uses ordinary pageable
staging; benchmark mode does no I/O at all (the assignment's discipline).

**Error handling and validation [M4 s83-84].** `CUDA_CHECK` macro on every runtime
call + `cudaGetLastError`/`cudaDeviceSynchronize` after launches — the deck's checkCuda
pattern. The deck teaches tolerance-based GPU validation (`fabs(h−g) > 1e-5`); we
exceeded it: the A100 result is **bitwise identical** to serial, because we fixed both
sources of divergence the course names — summation **order** (serial inner loop per
thread) and **FMA contraction** (`__dmul_rn`/`__dadd_rn` intrinsics; FMA is
instruction-level parallelism per [M1 s26], contracted by default by nvcc).

**Build [M4 s36].** `nvcc -arch=sm_80` via the CMake preset (`CUDA_ARCHITECTURES=80`);
`--ptxas-options=-v`-style register data came from ncu instead (52 registers/thread).

---

## 5. From Module 5 (OpenACC)

**Why the project uses CUDA instead of OpenACC.** The course positions directives as
"near-CUDA performance for a fraction of the effort" [M5 s13, s38]. We chose CUDA for
one specific reason directives cannot satisfy: **bitwise reproducibility requires
instruction-level control** — forcing non-contracted `__d*_rn` arithmetic — which a
descriptive `#pragma acc parallel loop` does not expose. Trade-off named, price known.

**But the module's central lesson is implemented.** The Laplace case study — naive GPU
port *slower than serial* because implicit `copyin/copyout` fire every `while`
iteration (98.9% memory / 1.1% kernels), fixed by one enclosing `data` region →
**68×** [M5 s54-79] — is exactly our device-resident design: the CUDA equivalent of
wrapping the time loop in `#pragma acc data`. Our nsys ratio (99.1% kernels / ~0%
memory) is the healthy mirror image of the deck's sick 1.1%/98.9%.

**Gang/worker/vector dictionary [M5 s124].** gang=thread block, vector=thread,
worker≈warp: our launch (blocks of 128 threads) is `num_gangs(18) vector_length(128)`
in OpenACC terms — and 128 respects the deck's Rule of 32.

**The `reduction(max:error)` lesson [M5 s23, s97]** parallels our OpenMP validation
reductions; and the OpenMP-offload dictionary [M5 s101-106] is the answer bank for
"could you have used OpenMP target offload?" (yes: `target teams distribute parallel
for` + `map(...)`; same data-residency requirement; same FMA caveat applies).

---

## 6. From Module 6 (Profiling I: methodology, POP)

**Profiling vs tracing [M6 s13].** We used both, as the deck prescribes: `nsys stats`
summaries (profile) to find *where* time goes, the timeline (trace) to see *how*
kernels/copies interact, and ncu counters for the *why* at kernel level.

**Measurement discipline [M6 s17-20, s41-47].** Wall-time timers around phases
(the program's chrono "pure dynamics" counter = the deck's bracketed-region
instrumentation), benchmark mode with I/O disabled, medians of 3 reps, JIT warm-up
excluded (numba) — each one is a slide: overhead/perturbation/accuracy [s17], wall vs
CPU time [s41-42], the "results depend on the test case" warning [s14] (hence the
size-sweep instead of a single N).

**gprof [M6 s67-69]** was unnecessary — the hotspot is analytically known (O(N²) force
vs O(N) everything else) and the program self-reports phase times; but the flat-profile
concept (exclusive time ranking) is what those phase timers implement by hand.

**The POP model maps onto our MPI results [M6 Deck2 s20-29].**
- **Load balance** `⟨Cᵢ⟩/max(Cᵢ)` ≈ 1 by construction: contiguous equal blocks of
  identical-cost particles (instruction balance perfect).
- **Communication efficiency** `max(Cᵢ)/T` is the eroding term: 86% at 32 ranks packed,
  collapsing to ~30% at 128 ranks on the small input — the POP table's signature
  "load balance flat, communication craters" [Deck2 s29/s47].
- **Transfer vs serialization** [Deck2 s25-26]: our packed-vs-spread pair (27.5× on 1
  node vs 19.2× over 4 nodes, same ranks/same N) isolates **transfer** cost (the
  network); there is essentially no serialization because there are no rank-to-rank
  dependencies — one symmetric collective per step, no pass-the-token staircase
  [Deck2 s14-19].
- The network model **T = L + V/BW** [Deck2 s10-11]: at 128 ranks the per-rank
  Allgatherv contribution is ~17 particles ≈ 280 B — deep in the latency-dominated
  regime, which is *why* M-size saturates. At XL each message is ~16× larger and
  efficiency returns (89%).

**The case-study reasoning pattern [M6 Deck2 s48-57]** — "the metric that moved names
the bottleneck; the fix targets exactly it" — is the structure of our whole results
narrative (occupancy names the GPU crossover; latency names the MPI saturation).

---

## 7. From Module 7 (Profiling II: nsys/ncu, roofline)

**The offloaded-region anatomy [M7 Deck1 s11-20]** (DATA IN / launch / KERNEL / WAIT /
DATA OUT) is our nsys timeline vocabulary: 201 launches of `computeForcesKernel`
(200 steps + initial force), ~610 µs each = 99.1% of GPU time; halfKick kernels 0.9%;
copies 0.19 ms in 9 calls.

**The Kernels-vs-Memory diagnostic [M7 Deck1 s52].** The deck's CFD example shows the
sick pattern (1.9% kernels / 98.1% memory). Ours is the healthy inverse — the
one-line proof that data residency worked.

**Kernel-launch overhead [M7 Deck1 s31-34]** explains numba-CUDA's 11.9× vs CUDA
C++'s 31.1× at small N: a Python-side launch per kernel per step, amortized poorly
when kernels are ~600 µs; "expose as much parallelism as possible" also restates the
occupancy story.

**Streams [M7 Deck1 s54-70].** The deck's Mandelbrot case (22 ms → 14 ms by
overlapping copies with compute) is the *positive* case; the same analysis on our
numbers (0.19 ms copies vs 120 ms kernels) yields the *negative* verdict — streams
would buy nothing. Being able to argue both directions from the same criterion is the
point.

**Roofline [M7 Deck2 s2-10].** `P = min(P_max, I×B)`; ridge point; coalescing moves
you up, tiling moves you right. Our force kernel at XL: DRAM ≈ 0 bytes moved per
FLOP → effective intensity is effectively unbounded (cache-resident working set) →
firmly **right of the ridge**, under the FP64 CUDA-core roof at 63% pipe utilization,
~37% of the A100's FP64 peak. The algorithmic-vs-effective distinction [Deck2 s7]:
our *algorithmic* intensity is already high (O(N) FLOPs per particle loaded), and the
caches deliver it — no tiling required to realize it.

**The pitfalls list [M7 Deck2 s14] reads like our checklist.** "Trivial workloads are
latency-bound and plot under all roofs" = our N=2231 under-occupancy (SM 4.5%);
"isolate the kernel window, exclude PCIe/init" = benchmark mode `none 0` and pure
dynamics timer; "match the precision ceiling" = we compare against the **FP64
non-tensor** peak (9.7 TF), not FP16/TF32 tensor roofs (which are irrelevant: bitwise
FP64 forbids tensor cores).

**The quantified price of reproducibility.** ncu's optimizer note: converting the
686 M non-fused FP64 ops to FMA ≈ **+31%** — declined, because non-contracted
arithmetic is what makes the GPU bitwise-equal to serial. The course teaches the
roofline as a map of what you *could* gain; we used it to price what we *chose not
to* gain. `ncu --set full --section SpeedOfLight_HierarchicalTensorRooflineChart`
[Deck2 s11] is verbatim what `submit_profile.sh` runs.

---

## 8. The FMA case study, placed in the course

Module 1 lists **FMA as instruction-level parallelism** [M1 s26]; Module 4's compiler
pipeline shows codegen choices happen below the source level [M4 s36-37]; Module 7
warns that byte/op accounting differs between the formula and the machine [Deck2 s7,
s14]. Our contribution ties these together with a concrete failure: on macOS/clang,
default FMA contraction in the Mandelbrot loop changed 14 cells of 1.2 M, one crossed
the selection threshold, and the C++ baseline generated 2232 particles vs Python's
2231 — a dataset *shape* mismatch from a single fused instruction. Verified fix:
`-ffp-contract=off` makes C++ bitwise-equal to Python. On Leonardo (gcc 12 + nvcc)
all seven runnable programs agree at N=2231. This is the strongest possible illustration
of the course's reproducibility warnings, discovered *before* touching the cluster —
and the reason every port forbids contraction.

---

## 9. One-paragraph synthesis (the closing argument)

The course teaches three ways to parallelize (threads, messages, GPU threads), two
laws that bound the payoff (Amdahl for fixed problems, the communication model
`t_serial/p + t_comm` for distributed ones), and a measurement method (profile →
identify → target → validate) built on the roofline. The project runs one O(N²)
kernel through all three technologies while holding the *numerics* invariant — outer-
index parallelism, inner-order preservation, no contraction — so that five of six
implementations are bitwise-identical to the serial baseline, and then uses the
course's own instruments to explain every performance number: Amdahl caps OpenMP at
~480× (we reach 29× of it on 32 cores at 91%); occupancy explains why the A100 ties
32 cores at N=2231 and beats them 19× at N=143k; and the latency term of `T=L+V/BW`
explains why 128 MPI ranks saturate on the official input but deliver 114× at 89% on
a 16× larger one. Every claim has a committed artifact; every artifact regenerates
from the committed logs.
