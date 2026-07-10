# Module 5 — OpenACC & directive-based GPU offload

*Based on "Programming GPUs with OpenX" by N. Shukla (CINECA), HPC Summer School 2025. This chapter follows the lecture in order; slide references appear as [s..].*

This module teaches how to move a working serial C/C++/Fortran program onto a GPU **without rewriting it in CUDA**, by adding compiler *directives* (special comment-like pragmas) that tell the compiler which loops to parallelise and how to move data. The vehicle for the whole lecture is one small program — a 2D Laplace heat solver — which is taken from a plain serial version all the way to a GPU version that runs roughly **68× faster**. The four numbered topics of the course are: (1) housekeeping, (2) OpenACC directives, (3) data management, (4) loop optimisation [s2].

The single most important lesson of the module, stated early and proven repeatedly, is this: **on a GPU, data movement — not computation — is usually what kills your performance.** Almost everything after the halfway point is about controlling *when* and *how often* data crosses between CPU and GPU.

---

## 1. Housekeeping and the running example [s1–3]

The course materials (four exercise skeletons, intentionally incomplete) live at `gitlab.hpc.cineca.it/cinecasummerschool/hpc25cineca` under `Day-5/openACC` [s3]. All runs are on **Leonardo**, CINECA's supercomputer (NVIDIA A100 GPUs, compute capability 8.0 → the `cc80` flag you will see later).

---

## 2. The case study: 2D Laplace heat transfer [s4–7, s11]

**The physics.** We simulate heat spreading across a metal plate. A constant temperature is applied to the top edge, and we let the heat diffuse until the plate reaches a steady state [s5]. Mathematically this is the Laplace equation in 2D, ∇²f(x,y) = 0 [s6].

**The numerical method (Jacobi iteration).** The steady-state solution is found *iteratively*: repeatedly replace each interior point with the average of its four neighbours until nothing changes much [s6]. The update rule is the classic 5-point stencil:

```
A_{k+1}(i,j) = ( A_k(i-1,j) + A_k(i+1,j) + A_k(i,j-1) + A_k(i,j+1) ) / 4
```

**The code structure** [s7]. Understanding this loop nest is essential because we parallelise it for the rest of the module:

```c
while (error > tol && iter < iter_max) {   // iterate until converged
  error = 0.0;
  for (int j = 1; j < n-1; ++j)            // sweep interior points
    for (int i = 1; i < m-1; ++i) {
      Anew[j][i] = 0.25 * (A[j][i+1] + A[j][i-1] + A[j-1][i] + A[j+1][i]);
      error = fmax(error, fabs(Anew[j][i] - A[j][i]));  // track largest change
    }
  for (int j = 1; j < n-1; ++j)            // copy Anew back into A ("swap")
    for (int i = 1; i < m-1; ++i)
      A[j][i] = Anew[j][i];
  iter++;
}
```

The four moving parts, all of which recur later: the **`while`** convergence loop; the **calc** loop nest that computes `Anew`; the **`error`** reduction (largest change over all points — this is a `max` reduction, and it will need special handling in parallel); and the **swap** loop that copies `Anew` into `A`. Note the code uses two separate arrays and a double buffer to avoid overwriting values still being read.

**The baseline to beat** [s11]. Running 1000 iterations serially takes **37.69 s**. Profiling shows the time split roughly as: the compute (`calc`) loop and the `swap` loop together dominate, with `init` a small slice — i.e. the two loop nests inside `while` are the hot spots worth parallelising. (Later GPU comparisons use a larger 4096×4096 grid whose serial reference time is **Ts = 137.78 s** [s54].)

---

## 3. Compiling with the NVIDIA HPC SDK (NVHPC / "PGI") [s8–10]

The compilers used throughout are NVIDIA's HPC compilers, historically known as **PGI** [s9]:

- `nvc` for C, `nvc++` for C++, `nvfortran` for Fortran.
- `-fast` turns on standard optimisation. So a plain serial build is `nvc -fast my_program.c`.

**Reading compiler feedback is a core skill in this module.** The `-Minfo` flag makes the compiler *tell you what it did* [s9–10]:

- `-Minfo=accel` — report what was accelerated / offloaded.
- `-Minfo=opt` — report all optimisations.
- `-Minfo=all` — report everything, positive or negative (e.g. "loop not parallelised because…").

You will lean on this constantly: a directive that *looks* applied may silently have been ignored, and only the `-Minfo` output tells you the truth.

---

## 4. Where directives fit: three ways to accelerate on a GPU [s12–14]

There is a spectrum of effort vs. control for programming NVIDIA GPUs [s13]:

| Approach | Character | Optimises for |
|---|---|---|
| **Libraries** (cuBLAS, etc.) | Drop-in acceleration | Productivity |
| **OpenX directives** (OpenACC / OpenMP) | Easy acceleration | Portability |
| **CUDA** | Maximum flexibility | Performance |

Moving left→right increases programming effort. Directives sit in the middle: far less work than CUDA, and *portable* (the same annotated source can compile for CPU or GPU), at the cost of some peak performance. This module is entirely about the middle column.

---

## 5. Refresher: OpenMP (the model directives build on) [s15–18]

OpenACC borrows OpenMP's mental model, so the lecture revisits OpenMP first.

**Timeline** [s15]. OpenMP has evolved since 1997 through three eras: *loop parallelism* → *tasking* → *heterogeneity* (GPU offload, from v4.0). The specification has grown from ~63 pages to 700+.

**How directives work** [s16]. Directives are hints understood only by aware compilers; a non-aware compiler ignores them, so the same source still compiles and runs serially. An OpenMP program uses the **fork-join** model: it runs serially (the original thread becomes "thread 0"), *forks* a team of threads at a parallel region, then *joins* back to serial. Regions can be nested.

**Two key constructs** [s17]:

- `#pragma omp parallel` — creates a team of threads that **all execute the whole block** (redundantly).
- `#pragma omp parallel for` — creates a team **and splits the loop iterations across them** (each thread does a share).

Thread count comes from `OMP_NUM_THREADS`.

**Why worksharing matters** [s18]. For a vector add `C[i]=A[i]+B[i]`:

- *Serial*: one thread, time ∝ N.
- *`parallel` only*: 4 threads each run **all** N iterations redundantly (wrong result — every thread overwrites C — and no faster: still ∝ N).
- *`parallel for`*: 4 threads split the work, ~N/4 each. **This is the point of worksharing.**

The lesson lands as a warning: adding `parallel` without distributing the loop gives you redundant work, not speed.

---

## 6. Checkpoint 0 — parallelise Laplace with OpenMP on the CPU [s19–26]

**The exercise** [s19–20]: add OpenMP directives to the two Laplace loop nests, vary `OMP_NUM_THREADS` from 1 to 8, and record the time.

**Building OpenMP with NVHPC** [s21]: add `-mp`, e.g. `nvc -mp -fast -Minfo=all my_program.c -o bin`.

**The solution** [s23]. The compute loop needs a reduction for `error`:

```c
#pragma omp parallel for shared(m, n, Anew, A) reduction(max:error)
for (int j = 1; j < n-1; ++j)
  for (int i = 1; i < m-1; ++i) { ... }        // compute + error

#pragma omp parallel for shared(m, n, Anew, A)  // swap loop
for (int j = 1; j < n-1; ++j) ...
```

The teaching points: arrays are `shared`, and the `error` accumulation must be a **`reduction(max:error)`** — each thread keeps a partial maximum and the runtime combines them at the end. Without the reduction you get a data race on `error`.

**An improvement — `collapse(2)`** [s24]: `#pragma omp parallel for collapse(2)` fuses the `j` and `i` loops into one larger iteration space, giving the scheduler more parallelism to distribute. (Collapse returns in the OpenACC section.)

**Results on Leonardo** [s25]. Speedup grows sub-linearly with threads: ~1.5× (4 threads), ~2.7× (8), ~3.7× (16), ~4.2× (32). Decent — but this is a CPU using at most ~4× of its serial self.

**The pivotal question** [s26]: *Will your legacy OpenMP (CPU) code perform well on the GPU?* The slide answers in one word: **NO.** This is the hinge of the whole lecture — CPU-style threading does not translate to good GPU performance, which motivates OpenACC.

---

## 7. What is OpenX? OpenACC vs OpenMP [s27–33]

**Course objective for this section** [s27]: enable you to accelerate your application with OpenACC.

**The two "OpenX" standards** [s29]:

- **OpenACC** — its *main focus* is offloading code onto GPUs.
- **OpenMP** — from v4.0 it *also* allows offloading onto GPUs.

The lecturer is explicit and neutral: *this course will not tell you whether OpenMP is better than OpenACC* [s29]. It teaches OpenACC in depth, then shows the OpenMP-offload equivalents at the end.

**Timeline with OpenACC added** [s30]. OpenACC 1.0 (2011) → 3.1, spec growing from 37 to ~149 pages. Governance is comparable in size: OpenMP ARB has 33 members, OpenACC 32.

**Same basic principle, different emphasis** [s31]:

| | OpenACC | OpenMP |
|---|---|---|
| Origin | Specifically targets GPU accelerators; started *after* OpenMP | Designed to replace low-level threading (POSIX/pthreads); targets shared-memory CPUs; offload since 4.0/4.5 |
| Model | Fork-join | Fork-join |
| Style | More **descriptive** | More **prescriptive** |
| Compilers | PGI/NVHPC, Cray, NVIDIA | GCC, Intel, IBM XL, LLVM/Clang |

**"Descriptive vs prescriptive" is a recurring theme.** OpenACC lets you *describe* that a loop is parallel and leaves the how to the compiler; OpenMP tends to have you *prescribe* the mapping.

**The host-device model** [s32]. Both use a host-centric model: one **host** (CPU) plus one or more **devices** (GPUs) of the same type, connected over PCIe or NVLink, with **separate memory spaces** — this separation is the source of all the data-movement pain to come. On the device, a function that runs on the GPU is a **kernel**; a kernel runs as a large set of threads executing concurrently in SPMD (single-program-multiple-data) fashion, each thread mapped to a CUDA core.

**Jargon aside: specification vs standard** [s33]. A *specification* is a detailed description of how something should be done and may or may not be adopted; a *standard* is an official rule that can be adopted *de jure* (by mandate) or *de facto* (by widespread use). OpenACC/OpenMP are specifications that have become de-facto standards.

---

## 8. Offloading with directives: the compute constructs [s34–45]

### 8.1 Offloading serial code [s34]

The most basic offload directive generates one or more **gangs** (OpenACC) / **teams** (OpenMP) of threads:

```c
#pragma acc parallel        // OpenACC
#pragma omp target teams    // OpenMP
{ ... }
```

By itself this makes the compiler generate parallel gangs/teams that **execute the block redundantly** — the same trap as OpenMP `parallel` without `for`. You still have to distribute the loop (next).

### 8.2 What OpenACC *is* [s35–36]

OpenACC is a **directive-based parallel programming model designed for performance and portability** [s36]. You keep your serial code and drop in a hint:

```c
int main() {
  <serial code>
  #pragma acc kernels    // <-- this region automatically runs on the GPU
  { <parallel code> }
}
```

Its selling points [s37]: **incremental** (accelerate one loop at a time), **single source** (one code base for CPU and GPU), **interoperable**, and **performance-portable** across CPU, GPU, and MIC.

### 8.3 The three jobs a directive does [s37]

OpenACC directives fall into three categories, and a real program uses all three together:

1. **Manage data movement** — `#pragma acc data copyin(a,b) copyout(c)`
2. **Initiate parallel execution** — `#pragma acc parallel`
3. **Optimise loop mappings** — `#pragma acc loop gang vector`

```c
#pragma acc data copyin(a,b) copyout(c)   // (1) data
{
  #pragma acc parallel                    // (2) compute
  {
    #pragma acc loop gang vector          // (3) loop mapping
    for (int idx=0; idx<N; idx++) C[idx] = A[idx] + B[idx];
  }
}
```

### 8.4 How good is it vs. CUDA? [s38]

OpenACC can reach up to **98% of CUDA's speed** on real applications (the slide cites the Cloverleaf benchmark on a V100, where the OpenACC and CUDA curves nearly overlap). It has 10,000+ developers and tool support in NVHPC, Cray, and LLVM. **The trade-off it embodies: near-CUDA performance for a fraction of the coding effort.**

### 8.5 Anatomy of a pragma [s39]

`#pragma acc <directive> <clauses>`:

- **`#pragma`** — a compiler *hint* in C/C++. Like a comment the compiler actually reads; if it doesn't understand it, it *ignores* it rather than erroring. This is what makes the code fall back to serial safely.
- **`acc`** — marks this as an OpenACC pragma. Non-OpenACC compilers (and even `nvc` when told to) ignore it, so the same source runs sequentially.
- **directive** — the command (e.g. "parallelise this").
- **clauses** — refinements/optimisations on the directive ("parallelise it *this* way").

Mental model: the directive says *what* general action to take; the clauses say *how specifically* to do it.

### 8.6 GPU programming as a two-step process [s40–41]

Every GPU port is really two problems: the **compute-bound** part (exposing work to the GPU environment) and the **data-bound** part (managing data movement). The lecture tackles compute first, then data.

### 8.7 Parallelism identified by the programmer: `parallel` and `loop` [s42–43]

- **`parallel`** [s42]: marks a parallel region; the compiler generates a parallel kernel for it. On its own, **each gang executes the entire loop** (redundant — the recurring warning).
- **`loop`** [s43]: identifies a loop whose iterations should be **distributed across threads**. `parallel` and `loop` are almost always written together as `#pragma acc parallel loop`. Just like OpenMP, the compiler then translates the region into a kernel that runs in parallel on the GPU.

### 8.8 `kernels`: let the compiler decide [s44]

- **`kernels`**: marks a region and lets the programmer **step back and rely entirely on the compiler** to find and prove the parallelism. `#pragma acc kernels loop independent` can hint that iterations are independent. There is **no explicit OpenMP counterpart** to `kernels` — it is OpenACC's signature "descriptive" construct.

### 8.9 `kernels` vs `parallel` — the difference you must know [s45]

This is the single most exam-worthy distinction in the OpenACC part of the course:

| **`kernels`** (compute directive) | **`parallel`** (compute directive) |
|---|---|
| Gives the **compiler leeway** to parallelise and optimise | **Programmer's responsibility** to identify the parallel region |
| **Only** provided by OpenACC | Has an OpenMP counterpart |
| **Compiler guarantees correctness** (it will refuse to parallelise if unsafe) | **Programmer guarantees correctness** |
| Can fuse loop nests; typically generates a separate kernel per loop | `parallel` alone runs the same code redundantly; you assert no dependencies |
| Implicit barrier at the end **and between each loop** | Implicit barrier at the end of the region |

Crucial nuance: with `kernels`, if the compiler *can't prove* a loop is safe it will run it serially (safe but slow); with `parallel`, the compiler *trusts you* and will parallelise even an unsafe loop (fast but potentially wrong). **When both are fully optimised they give similar performance** — the difference is who is responsible for correctness.

---

## 9. The development cycle [s46]

Accelerating code is iterative, a three-stage loop:

1. **Analyze** — profile to find the hot spots that need parallelising/optimising.
2. **Parallelize** — start with the most time-consuming parts; check correctness; re-analyse.
3. **Optimize** — improve and measure the speedup, then go round again.

This "Analyze → Parallelize → Optimize" cycle is the backbone of every checkpoint in the module.

---

## 10. Compiling & running OpenACC with NVHPC [s47–53]

**Basic build** [s48]:

- NVHPC: `nvc -fast -gpu=<target> -Minfo=accel -o laplace_2d laplace_2d.c`
- GCC: `gcc -fast -fopenacc -foffload=<target> ...`
- Cray: `cc -fast -h pragma=acc -h msgs ...`

**The flags that matter** [s49] — where you build serial, multicore, or GPU from the *same source*:

| Flag | Effect |
|---|---|
| `-acc` | enable OpenACC offload to device |
| `-acc=host` | build to run **serially on the host CPU** |
| `-acc=multicore` | parallelise for a **multicore CPU** |
| `-acc=gpu -gpu=cc80` | map OpenACC parallelism to an **NVIDIA GPU**, compiling for compute capability 8.0 (A100) |
| `-gpu=managed` | put all allocatables in **CUDA Unified (managed) Memory** |
| `-gpu=pinned` | use CUDA pinned memory |
| `-Minfo=acc` | compiler diagnostics for OpenACC |
| `NVCOMPILER_ACC_NOTIFY=1\|2\|3` | env var: runtime notifications of kernel launches / data transfers |

Example: `nvc -acc -gpu=cc80,managed -Minfo=acc -o binary code.c`. (The same table exists for `-mp` to drive **OpenMP** GPU offload [s50].)

**Checkpoint 1** [s51–52]: parallelise the serial Laplace with OpenACC — add `parallel` and/or `kernels` constructs to the two loops, and remember the `reduction` for `error`.

**Sanity check: multicore build** [s53]. Compiling with `-acc=multicore -Minfo=acc` prints exactly what the compiler did, e.g.:

```
main:
  49, Generating Multicore code
      49, #pragma acc loop gang
  49, Generating reduction(+:error)
  51, Loop is parallelizable
```

This confirms the compiler recognised the loop as parallelisable and generated the reduction — the payoff of reading `-Minfo`.

---

## 11. First GPU results and the data-transfer trap [s54–57]

**A promising start** [s54]. On the 4096×4096 grid (serial Ts = 137.78 s): the OpenMP-CPU (`OMP32`) version reaches ~4×, and the OpenACC multicore/first version (`ACC32`) ~4.5×. So far, so good.

**Reading the GPU compiler report** [s55–56]. Set `NV_ACC_TIME=1` (older name `PGI_ACC_TIME=1`) for a lightweight profiler of data-movement and kernel time; `NV_ACC_NOTIFY=1` gives a detailed breakdown of kernel launches and transfers. A `kernels` build's report shows the compiler being helpful and honest:

```
main:
  49, Loop is parallelizable
      Generating implicit copyin(A[:][:])   [if not already present]
      Generating implicit copy(error)       [if not already present]
      Generating implicit copyout(Anew[1:4094][1:4094])
  50, Generating Tesla code
      49, #pragma acc loop gang, vector(128) collapse(2)
          Generating implicit reduction(max:error)   // compiler found the reduction!
  69, FMA (fused multiply-add) instruction(s) generated
```

Two things to notice: the compiler **auto-generated the `max` reduction** for `error`, and it **auto-inserted "implicit" `copyin`/`copy`/`copyout`** for the arrays. That second point is exactly the problem.

**Disaster** [s57]. When you actually run the naive `Kernels` and `Parallel` GPU versions, they are **slower than the serial baseline** — far worse than the CPU versions. The slide asks: *What went wrong?* The answer is those "implicit" copies: the compiler, unable to see the bigger picture, copies the arrays to and from the GPU **on every iteration of the `while` loop**.

---

## 12. Why data movement dominates GPU performance [s58–64]

This section is the conceptual heart of the module.

**The bandwidth reality** [s59]. The numbers explain everything:

| Path | Bandwidth |
|---|---|
| GPU ↔ its own memory (HBM2) | **900 GB/s** |
| GPU ↔ CPU over **PCIe** | **16 GB/s** |
| GPU ↔ GPU over NVLink | 25 GB/s |
| CPU ↔ RAM (DDR4) | 128 GB/s |

The GPU talks to its own memory ~**56× faster** than it talks to the CPU over PCIe. So: **data must be resident on the device when a kernel launches, and you must minimise how often it crosses the PCIe link.** Explicit memory management exists precisely to control this.

**The three processing steps** [s60–62]. Every offloaded computation is: (1) copy input CPU→GPU, (2) execute the kernel, (3) copy results GPU→CPU. Steps 1 and 3 are the expensive PCIe crossings.

**Managed (Unified) Memory — the easy button** [s63–64]. With `-gpu=managed`, CPU and GPU memory are presented as a **single shared pool**; the runtime migrates pages **only when actually needed**, instead of the compiler conservatively copying whole arrays every iteration. Compare `nvc -gpu=cc80 …` (manual) vs `nvc -gpu=cc80,managed …` (managed). For the Laplace code [s64]: *without* managed memory the compiler must determine the sizes of `A` and `Anew` and copy them to/from the GPU each iteration "to be safe"; *with* managed memory the runtime moves data lazily. Managed memory is the quickest way to get a correct, faster GPU run — but the fully explicit approach (next) is faster still.

---

## 13. Explicit data management [s65–79]

Now the "data-bound" half of the two-step process [s65]: controlling data with **data clauses** and **data regions**.

### 13.1 The three-step data lifecycle [s66]

For an array on the GPU you: (1) **allocate** it on the device, (2) **copy** host→device, (3) at the end **copy back** and deallocate. OpenACC clauses express each combination:

```c
#pragma acc parallel loop copy(A[0:N])                     // to + from
#pragma acc parallel loop copyin(A[0:N]) copyout(B[0:N])   // A in, B out
```

### 13.2 Data-region constructs [s67]

A **data region** defines a scope in which GPU arrays **stay resident and are shared across multiple compute regions** (multiple `kernels`/`parallel`/`loop` blocks). It is a *structured* construct: it must **start and end in the same function**.

```c
#pragma acc data [clause]      // C/C++
{ code region, including compute pragmas }

!$acc data                     // Fortran
  code region ...
!$acc end data
```

This is the key tool for fixing the "copy every iteration" disaster: wrap the whole `while` loop in one `data` region so the arrays live on the GPU for the entire solve.

### 13.3 The data clauses [s68]

These give the programmer control over *how and when* data is created on and copied to/from the device. OpenACC clause / OpenMP `map` equivalent:

| Clause | OpenMP map | Behaviour |
|---|---|---|
| `copyin(list)` | `map(to:)` | Allocate + copy **host→device** at region start; free at end **without** copying back |
| `copyout(list)` | `map(from:)` | Allocate on device (uninitialised); at end copy **device→host** and free |
| `copy(list)` | `map(tofrom:)` | Allocate + copy in at start, copy out at end, then free |
| `create(list)` | | Allocate on device, **no transfer** |
| `delete(list)` | | Free on device, **no transfer** |
| `present(list)` | | Assert data is already on the device |
| `present_or_copy`, `present_or_create`, `device_ptr` | | Conditional / raw-pointer variants |

Choosing the *minimal* clause (e.g. `copyin` for read-only inputs, `create` for scratch) avoids needless transfers.

### 13.4 Moving data host↔device, and array shaping [s69–70]

Data clauses can be attached to `data`, `kernels`, `parallel`, or `declare` constructs [s69]. Example:

```c
#pragma acc data copyin(a[0:N], b[0:N]) copyout(c[0:N])
{ #pragma acc parallel loop for(int i=0;i<N;i++) c[i]=a[i]+b[i]; }
```

**Array shaping** [s70] tells the compiler the *extent* of an array when it can't figure it out itself (common with raw pointers). You must specify the range explicitly, and **the notation differs by language**:

- **C/C++**: `start:count` → `a[0:N]` means N elements starting at index 0.
- **Fortran**: `start:end` → `a(1:N)`.

Example: `#pragma acc data copyin(a[0:N]) copyout(b[s/4:3*s/4])`. The shaped memory only exists within the data region and must be within a single function.

### 13.5 Encompassing multiple compute regions [s71–72]

The real value of a data region is sharing arrays across several kernels without shuttling them back and forth [s71]. The DAXPY example [s72] makes the two scopes visexplicit: an outer **data region** (`#pragma acc data create(D,Y) copyin(A) copyout(D)`) surrounds two inner **compute regions** (two `#pragma acc parallel loop`s that initialise then compute `D = A*X + Y`). Data crosses PCIe once, not twice per loop.

### 13.6 The checkpoint-2 progression — and the payoff [s73–79]

**Checkpoint 2** [s73–74]: parallelise Laplace *and* add data management. Step 1: add `parallel loop`. Step 2: add a **structured `data` directive** so `A` and `Anew` are handled properly; profile with `nsys profile -t nvtx,openacc --stats=true --force-overwrite true -o laplace ./laplace`.

**The still-wrong version ("Data C1")** [s75]. Putting the data clauses **on the parallel loops** looks reasonable and gives the arrays their "shape":

```c
#pragma acc parallel loop reduction(max:err) copyin(A[0:n*m]) copy(Anew[0:n*m])
for (...) { Anew[j][i] = 0.25*(...); err = max(err, fabs(Anew[j][i]-A[j][i])); }

#pragma acc parallel loop copyin(Anew[0:n*m]) copyout(A[0:n*m])
for (...) A[j][i] = Anew[j][i];
```

But it is **still slow** [s76] — "What went wrong?" again. Because the clauses are on the *inner* loops, the data is still copied **every `while` iteration**.

**Proof from the profiler** [s77]. The Nsight timeline of this version is damning: **98.9% Memory, only 1.1% Kernels**; the `while` loop takes 30.6 s, of which 60.3% is host-to-device and 39.7% device-to-host `memcpy`. The GPU spends virtually all its time copying, almost none computing.

**Why** [s78]. The copies happen on **every iteration of the `while` loop**. And note: with **two** `#pragma acc parallel` regions per iteration, you pay **four** copies per iteration (in+out for each).

**The fix and the payoff** [s79]. Wrap the entire `while` loop in a single structured `#pragma acc data` region so `A` and `Anew` stay resident on the GPU across all iterations, transferring only once at the start and once at the end. Result: the "Data Optimized" version jumps to **~68×** speedup — from *slower than serial* to 68× faster, purely by fixing data movement. **This is the module's headline result and its central lesson in one chart.**

---

## 14. Unstructured data lifetimes [s80–89]

*(Section: "Advanced OpenACC" [s80].)*

Structured `data` regions have a limit: they must begin and end in the **same function**. But real data lifetimes don't respect function boundaries — data is often allocated in one routine and freed in another [s81].

**The animated example** [s82–85] walks through a typical C pattern: an `allocate()` function `malloc`s and returns a pointer; `main` uses it in a loop; a `deallocate()` function `free`s it. The data is *created*, *used*, and *deallocated* in three different places.

**Unstructured data directives** [s86–87] solve this with a pair that need not be lexically nested:

```c
#pragma acc enter data <clauses>   // upload / create — no { } region opened
  ... sequential and/or parallel code, possibly across functions ...
#pragma acc exit data <clauses>    // download / delete
```

Applied to the alloc/free pattern [s87]:

```c
int* allocate(int size) {
  int *ptr = malloc(size*sizeof(int));
  #pragma acc enter data create(ptr[0:size])   // create on device here
  return ptr;
}
void deallocate(int *ptr) {
  #pragma acc exit data delete(ptr)            // delete on device here
  free(ptr);
}
```

Key properties: `enter data` **uploads/creates**, `exit data` **downloads/deletes**; they **can branch across multiple functions**; and they **do not open a data region** (no implicit scope).

**The clauses** [s88]:

- `copyin(list)` — allocate + copy host→device **on `enter data`**.
- `copyout(list)` — deallocate + copy device→host **on `exit data`**.
- `create(list)` — allocate on device, no transfer, on `enter data`.
- `delete(list)` — deallocate on device, no transfer, on `exit data`.

**Structured vs unstructured, side by side** [s89]:

| Unstructured (`enter/exit data`) | Structured (`data { }`) |
|---|---|
| Can have **multiple** start/end points | Must have **explicit** start/end |
| Can **branch across functions** | Must be within a **single function** |
| Memory lives until **explicitly deallocated** | Memory lives **only within the region** |

---

## 15. Loop optimisation clauses [s90–97]

*(Section 4: OpenACC Loop Optimization [s90].)* Once data is under control, you tune how loops map onto the hardware.

**`seq`** [s91–92] — "sequential." Tells the compiler to run a loop **sequentially** rather than parallelising it. Typical use: parallelise the outer loop across threads, but have each thread run the inner-most loop serially. The compiler may auto-apply `seq` to deeply nested loops. There is also a `routine seq` form [s92]: `!$acc routine seq` marks a function as callable **on the GPU** and executed **by a single device thread**; at the call site inside a `parallel loop`, each thread calls its own instance (Fortran `sqab` example).

**`private` and `firstprivate`** [s93] — `private(list)` gives **each thread its own copy** of the listed variables (uninitialised); `firstprivate(list)` is the same but each copy is **initialised to the host's value**. Example: a scratch `tmp[3]` array made private to each iteration of the outer loop, while the result array is shared.

**Scalars and the private clause** [s94] — by default, scalars are **`firstprivate` in a `parallel` region** and **`private` in a `kernels` region**. You usually needn't list them, except for global/module variables, scalars passed by reference to a device routine, or scalars used as an rvalue after the region ("live-out"). **Warning: forcing scalars into a `private` clause can actually hurt performance.**

**`collapse(N)`** [s95] — **combines the next N tightly-nested loops** into a single larger iteration space. Turns a multidimensional nest into one dimension, which increases memory locality and exposes more parallelism (a bigger pool of independent iterations for the GPU). `#pragma acc parallel loop collapse(2)`.

**`tile(x,y,…)`** [s96] — **breaks a multidimensional loop into blocks ("tiles")** that can execute simultaneously, improving data locality in some codes (e.g. `tile(32,32)`). Like `collapse`, the inner loops must **not** carry their own `loop` directive.

**`reduction(op:var)`** [s97] — takes many values and **reduces them to a single result**: each thread does a partial reduction over its iterations, and after the loop the compiler combines the partials with the operator. Supported operators: `+` (sum), `*` (product), `max`, `min`, `&` (bitwise AND), `|` (bitwise OR), `&&` (logical AND), `||` (logical OR). This is exactly what `error` needed (`reduction(max:error)`).

---

## 16. GPU hardware & the gang/worker/vector model [s98–100, s109–119, s121–122]

*(Section "4": GPU hardware hierarchy [s98].)* OpenACC exposes **three levels of parallelism** — **gang**, **worker**, **vector** — that map onto whatever hardware you target [s99]:

| Platform | Gang | Worker | Vector |
|---|---|---|---|
| Multicore CPU | Entire CPU (NUMA domain) | Core | SIMD vector |
| Manycore CPU (Xeon Phi) | NUMA domain (whole chip) | Core | SIMD vector |
| **NVIDIA GPU** | **Thread block** | **Warp** | **Thread** |
| AMD GPU | Workgroup | Wavefront | Thread |

The relationship: **N_threads = L_vector × N_workers** (threads per gang) [s99].

**The A100 in numbers** [s100]. Peak FLOPS = clock × cores × FLOP/cycle; with a boost clock of 1.41 GHz the A100 reaches **250 TFLOPS**. Hardware nests as Streaming Multiprocessors (SMs) → CUDA cores (each an FP + INT unit); software nests as Gang → Worker → Vector; and CUDA's own hierarchy is Thread ∈ Block ∈ Grid.

### The painter analogy [s109–114]

The lecture demystifies gang/worker/vector with a decorator painting a building:

- A **single worker** (painter) can only move so fast — his speed limits throughput [s109].
- Giving him a **bigger roller** (a wider *vector*) helps, but only up to a point — "we need more workers!" [s110].
- Adding **many workers** [s111] speeds things up, and organising them **into gangs** lets groups on different floors work **independently** — so you can use as many or as few gangs as you need [s112].

Mapping the metaphor [s113]: the **painter is a `worker`**; his **roller is a `vector`** (covers more wall at once); workers are organised into **`gangs`**; each gang has its own **cache**.

**The definitions** [s114]:

- **Gang** — multiple gangs are generated and loop iterations are spread across them; gangs are **independent** of each other; you can't know exactly how many run at a given time.
- **Worker** — allows **multiple vectors within a gang**; splits one large vector into smaller ones; an intermediate level, useful when inner parallel loops are small.
- **Vector** — the **lowest** level of parallelism; every gang has at least one vector; threads work in **lockstep** (SIMD/SIMT).

```c
#pragma acc loop gang
for (int i=0;i<N;i++)
  #pragma acc loop worker
  for (int j=0;j<N;j++)
    #pragma acc loop vector
    for (int k=0;k<N;k++) structured-block
```

### Controlling the sizes [s115]

The compiler picks the counts for you, but you can override with clauses on `#pragma acc parallel`:

- `num_gangs(N)` — number of gangs.
- `num_workers(M)` — workers per gang.
- `vector_length(P)` — vector length.

**Rule of 32** [s115]: on NVIDIA GPUs, **always make the vector length a multiple of 32** (32, 64, 96, 128, … 1024). This is because a warp is 32 threads.

### Sizing worked examples [s116–119]

Using a 4×8 array with `#pragma acc kernels loop gang worker(1)`:

- `vector(8)` on the inner (length-8) loop → 4 gangs, one worker each, vector exactly covers the inner loop [s116].
- `vector(4)` on the same length-8 loop → still 4 gangs, but each vector now does **two** iterations [s117].
- `vector(8)` on a length-**4** inner loop → the vector is **larger than the loop**; **half the vector is wasted**, so the code runs at half efficiency [s118].
- **The fix** [s119]: break the vector across **2 workers** (`worker(2) vector(4)`) so the smaller vector fits the loop exactly — no waste. **The lesson: always match gang/worker/vector dimensions to the actual loop sizes.**

### Final exercise and worker tuning [s120–122]

**Final exercise** [s120–121]: optimise `laplace2d_OpenACC` and **beat 0.60 s** (again profiling with `nsys`). Tuning the worker count [s122] on the optimised code shows a broad optimum: `Worker(2)` and `Worker(4)` reach the peak (~68×), while `Worker(16)` and `Worker(32)` drop back (~58–60×) — **more parallelism is not always better; oversubscribing hurts.**

---

## 17. OpenMP target offload & the OpenACC ↔ OpenMP comparison [s101–108, s123–128]

The final thread of the lecture asks: could you do all of this with **OpenMP offloading** instead? [s123, shown over a Leonardo image]. The answer is yes, with directly analogous directives — which is why the two are collectively "OpenX."

### 17.1 Offloading serial code, both ways [s34 revisited]

`#pragma acc parallel` ≡ `#pragma omp target teams` — both generate one or more gangs/teams executing the block redundantly until you distribute the loop.

### 17.2 `distribute`: spreading a loop over teams [s101–102]

OpenMP's `#pragma omp distribute` is the counterpart to `#pragma acc loop gang`:

```
#pragma omp target teams          #pragma acc parallel
{                                  {
  #pragma omp distribute            #pragma acc loop        // spread i over gangs
  for(i=0;i<n;i++)                  for(i=0;i<n;i++)
    for(j=0;j<m;j++)                 #pragma acc loop        // + do the right thing
      for(k=0;k<p;k++) ...            for(j=0;j<m;j++)
}                                       #pragma acc loop
                                         for(k=0;k<p;k++) ...
```

The contrast the slide draws [s101–102]: `omp target teams`/`distribute` explicitly says "generate teams" and "distribute `i` over them" but gives **no information about the `j` or `k` loops** (OpenMP is prescriptive — you must annotate each level). OpenACC's `acc parallel`/`loop` says the loops are independent and lets the compiler **"do the right thing"** (descriptive).

### 17.3 Synchronisation philosophy [s103–104]

- **OpenMP**: users may use **barriers, critical regions, and locks** to protect against data races; it is even possible to parallelise not-truly-parallel code with these [s103].
- **OpenACC**: users are **expected to refactor code to remove data races**; the code should be made **truly parallel and scalable** — there are no locks/critical sections [s103].

Example [s104]: where OpenMP wraps a racy update in `#pragma omp critical`, OpenACC expects you to have already written a race-free `parallelRand(A)` before the `#pragma acc parallel loop`.

### 17.4 The Laplace solver in OpenMP offload [s105–106]

For completeness [s105], the same solver as an OpenMP GPU offload:

```c
#pragma omp target data map(alloc:Anew) map(A)     // structured data region
while (error > tol && niter < niter_max) {
  error = 0.0;
  #pragma omp parallel for reduction(max:error)     // (this line runs on host)
  for (int j=1;j<n-1;++j) for (int i=1;i<m-1;++i) { ... }

  #pragma omp target teams distribute parallel for collapse(2) schedule(static,1)
  for (int j=1;j<n-1;++j) for (int i=1;i<m-1;++i) A[j][i] = Anew[j][i];
}
```

Note `map(alloc:Anew) map(A)` is the OpenMP equivalent of the outer `acc data` region, and `target teams distribute parallel for collapse(2)` is the full offload+distribute+worksharing chain.

### 17.5 The final performance picture [s107–108]

The summary chart [s107] lines up every version on the same axis (serial Ts reference): CPU threading (`OMP32`, `ACC32`) gives only a few ×; the naive GPU attempts (`ACCpar`, `ACCData C1`) are near-useless; and the properly data-managed versions — **`Data Opt`, `GWV` (gang/worker/vector tuned) ~68×, and `OMPGPU` ~48×** — dominate. OpenACC's best edges out OpenMP-offload here, but both are in the same league, and both are an order of magnitude beyond CPU threading.

**Closing thoughts** [s108]:

- **Accelerating C/C++/Fortran with OpenX is Simple, Powerful, Portable.**
- **Take a profile-driven approach** — you must *understand data dependencies and data movement* (the recurring lesson).
- **Optimise** by (a) making data movement coarse-grained with data regions and (b) tuning loops with `collapse`, `seq`, `independent`, etc.

### 17.6 The OpenACC ↔ OpenMP dictionaries [s124–128]

**Hardware mapping** [s124]:

| OpenACC | CUDA | NVIDIA GPU | OpenMP |
|---|---|---|---|
| Parallel / Kernel | Kernel | GPU | Parallel |
| Gang | Thread block | SMs | Team |
| Worker | Thread | SP / compute unit | Thread |
| Vector | Warp (32 threads) | 32-wide thread | SIMD |

**Directive mapping** [s125] (in theory the levels adapt to the hardware, but in practice some compilers struggle with certain levels):

| OpenACC | OpenMP |
|---|---|
| `acc parallel` | `omp target teams` |
| `acc loop gang` | `omp distribute` |
| `acc loop worker` | `omp parallel loop` |
| `acc loop vector` | `omp simd` |
| `acc declare` | `omp declare target` |
| `acc data` | `omp target data` |
| `acc update` | `omp target update` |
| `copy` / `copyin` / `copyout` | `map(tofrom / to / from : …)` |

**`parallel`: similar but different** [s126]:

| OMP `parallel` | ACC `parallel` |
|---|---|
| Creates a team of threads | Creates 1+ gangs of workers |
| **Well-defined** how thread count is chosen | Compiler **free to choose** gangs/workers/vector |
| **May** synchronise within the team | May **not** synchronise between gangs |
| Data races are the **user's responsibility** | Data races **not allowed** |

**`loop`: similar but different** [s127]:

| OMP `loop` (for/do) | ACC `loop` |
|---|---|
| **Worksharing**: splits iterations across the team; guarantees the user has handled data races | **Declares** iterations independent & race-free (`parallel`) or to-be-analysed (`kernels`) |
| Scheduling of iterations may **restrict** the compiler | User can declare independence **without** prescribing scheduling; compiler free to schedule with gang/worker/vector unless overridden |

**`distribute` vs `loop`** [s128]:

| OMP `distribute` | ACC `loop` |
|---|---|
| Must live in a **`teams`** region | (No such requirement) |
| Distributes iterations over 1+ teams | Declares iterations independent & race-free or to-be-analysed |
| Only the **master thread** of each team runs iterations until `parallel` is encountered | — |
| Iterations implicitly independent, but some compiler optimisations still restricted | Compiler free to schedule with gang/worker/vector unless overridden |

The through-line of all four tables: **OpenACC is descriptive (you state intent, the compiler schedules); OpenMP is prescriptive (you spell out the mapping).**

---

## Likely exam angles

- **`kernels` vs `parallel` [s44–45].** Be able to state the difference cleanly: `kernels` gives the compiler full leeway and **the compiler guarantees correctness** (won't parallelise what it can't prove safe); `parallel` puts the parallelism and **correctness on the programmer** (it trusts you and will parallelise even unsafe loops). `kernels` is OpenACC-only, can fuse loop nests, and has a barrier between loops; fully optimised, both perform similarly.
- **Why data movement dominates [s57–59, s76–79].** Expect a question on *why the naive GPU version is slower than serial*. Answer: implicit `copyin/copyout` on **every `while` iteration** over a 16 GB/s PCIe link vs. 900 GB/s on-device bandwidth; the profiler shows ~99% time in memory, ~1% in kernels. The fix — one enclosing `#pragma acc data` region so arrays stay resident — is what turns a slowdown into **~68×**. Know the `copy`/`copyin`/`copyout`/`create` semantics and the C `start:count` vs Fortran `start:end` array shaping.
- **The OpenACC vs CUDA trade-off [s13, s38].** Directives sit between libraries and CUDA: much less effort and portable single-source, reaching up to ~98% of CUDA speed. You give up a little peak performance and fine control for a large productivity gain.
- **Structured vs unstructured data [s67, s81–89].** Structured `data { }` = single function, memory scoped to the region; unstructured `enter data`/`exit data` = multiple entry/exit points, can cross functions, memory lives until explicitly deleted.
- **Gang/worker/vector [s99, s114–119].** Know the mapping to NVIDIA hardware (gang=thread block, worker=warp, vector=thread), `N_threads = L_vector × N_workers`, the **Rule of 32** for vector length, and the sizing pitfall (a vector larger than the inner loop wastes half the threads; fix by splitting across workers).
- **OpenACC ↔ OpenMP equivalence [s124–128].** Be ready to translate directives (`acc parallel`↔`omp target teams`, `acc loop gang`↔`omp distribute`, `copy`↔`map(tofrom)`) and to articulate the descriptive-vs-prescriptive distinction and the data-race stance (OpenACC forbids races and expects refactoring; OpenMP lets you use critical/locks).
- **The reduction [s7, s23, s97].** The `error` variable requires `reduction(max:error)` in every parallel version; know the available operators and that each thread computes a partial result combined after the loop.
