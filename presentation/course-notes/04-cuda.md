# Module 4 — GPU computing with CUDA

> Source deck: *Accelerating Scientific Applications on NVIDIA GPUs — Single GPU to Multi-GPU Scalability*, Nitin Shukla, HPC Department, CINECA, 2026 (160 slides). Slide references below, written `[sN]`, follow the PDF page order.
>
> This chapter is written for a student who never attended the lecture. It walks the deck's narrative in order: **why** GPUs look the way they do (the hardware), **how** you program them (the CUDA model), **what** it costs to move data, and **how** to make it fast (indexing, occupancy, coalescing, memory hierarchy, the roofline model, pinned memory, and streams). Everything here is drawn from the slides — no outside material is added.

---

## How the course is organised [s1–s2]

The deck frames the whole of GPU computing as eight conceptual modules [s2]: **Foundations** (latency vs. throughput, SM architecture, silicon economics), **Parallelism** (thread hierarchy, index math, kernel execution), **Memory** (hierarchy, pageable/pinned, shared memory), **Analysis** (Nsight Systems, Nsight Compute, roofline), **Errors** (device queries, runtime error handling), **Streams** (async execution, default vs. non-default streams), **Multi-GPU** (NVLink, NCCL, distributed parallelism), and **Future HW** (Blackwell, FP4, tensor cores, NVL72). In the actual slide flow these appear as four numbered lecture modules — CUDA programming model (Module 02), the impact of data movement (Module 03), the profiling tools and the memory-hierarchy deep dive (both labelled Module 04) — bracketed by a long hardware-foundations opening and a streams finale.

---

# Part 1 — Foundations: why GPUs exist and how they are built [s3–s27]

## The heterogeneous shift [s3]

The opening argument is that modern science *needs* heterogeneous computing (CPU + accelerator) for three reasons: **computational density** — a GPU tiles its die with thousands of smaller arithmetic units; **power efficiency** — it is optimised for throughput per watt; and **application need** — scientific data is growing faster than the serial speed of a single CPU can keep up with. The Leonardo supercomputer at CINECA is the running example of this design at scale.

## Latency-oriented vs. throughput-oriented design [s4–s10]

This is the single most important idea in the whole deck, and it is repeated from several angles.

**The chip-area argument [s4–s7].** A CPU die and a GPU die are drawn side by side. A real CPU die contains only *a couple* of large ALUs, surrounded by big Control blocks and a deep cache hierarchy (L1, L2, L3) feeding DRAM. A real GPU die is *tiled* with thousands of small ALUs (the green grid), a thin control strip, and a comparatively tiny cache (just L2) over DRAM. The same transistor budget is spent two opposite ways:

- **ALUs [s5].** The ALU does all arithmetic. A CPU puts a *few large* ALUs per core (roughly tens on a desktop chip). A GPU puts *thousands of smaller* ones down for parallel throughput.
- **Per-thread control [s6].** Control means fetch, decode, branch handling, scheduling. A CPU *replicates heavy control per core* so it runs branchy code fast. A GPU uses **SIMT** — one instruction drives many threads — so control logic stays thin.
- **Cache and memory [s7].** A CPU devotes huge area to L1/L2/L3 so that *one* thread can hide DRAM latency out of cache. A GPU keeps little on-chip cache and instead hides latency by **switching between warps**. The GPU's formula is therefore *simple control, lean cache, thousands of efficient cores*.

**The clock-speed paradox [s8–s9].** A CPU runs at 3.0–4.5 GHz, tuned for raw single-threaded speed. A GPU runs at only 1.3–1.5 GHz — the clock is deliberately kept low to manage heat and power so that thousands more cores fit on the same die. So the GPU runs at roughly **half** the clock of a CPU yet wins by doing thousands of things at once. Throughput is captured by a simple formula [s9]:

```
Throughput_GPU ≈ (Cores) × (Ops/Cycle) × GHz
```

Worked example: a typical 8-core CPU delivers ~64 GFLOPS (8 cores × 2 FLOPs × 4 GHz), while an NVIDIA A100 delivers ~19.5 TFLOPS (6912 cores × 2 FLOPs × 1.41 GHz) — roughly a **300× advantage for parallel tasks**. The metaphor: a sports car (one very fast core) vs. a bus (thousands of slower, efficient cores).

**The summary table [s10].** A CPU is scalar + SIMD, latency-optimised, with a hardware-managed cache hierarchy — ideal for operating systems, control-flow-heavy and latency-sensitive work. A GPU is **SIMT** parallel processing built from Streaming Multiprocessors (warp schedulers + register file + CUDA cores + L1), backed by L2 and HBM/GDDR — ideal for massively parallel workloads, AI training, and graphics/scientific computing.

## Anatomy of a modern GPU [s11–s13]

The GPU is a hierarchy of building blocks [s11]:

- **GPC (Graphics Processing Cluster)** — the top-level block of the die. Each GPC holds multiple SMs, a Raster Engine, and dedicated L2 cache slices. A high-end GPU like the H100 packs **7 GPCs**.
- **SM (Streaming Multiprocessor)** — the core compute unit where execution actually happens. Each SM manages its own register file (up to **256 KB**), its L1/shared memory (**192 KB on Hopper**), and schedules its warps independently.
- **GigaThread Engine** — the hardware orchestrator that dispatches thread blocks to whichever SMs are free, balancing the workload so no compute unit sits idle while work remains.

**Threads are grouped into warps [s12].** A thread block is broken into **warps of 32 threads**. The deck's phrasing is that "a warp is the vector element of the GPU" — threads 0–31 form one warp, 32–63 the next, and so on.

**Inside one SM [s13].** Zooming into a single SM shows an L1 instruction cache over four sub-partitions. Each sub-partition has its own L0 instruction cache, a warp scheduler, a dispatch unit, a **register file of 16,384 × 32-bit** entries, tensor memory, and execution units grouped by type (INT32, FP32, FP32, FP64, and a Tensor Core), plus 8 load/store units and an SFU (special function unit). The SM also carries a Tensor Memory Accelerator and **256 KB of L1 data cache / shared memory** shared across the sub-partitions.

## Warps and the scheduler — how latency is actually hidden [s14–s17]

**Hardware multithreading [s14].** The NVIDIA SM schedules threads in warps (groups of 32). A warp is simply a set of threads scheduled together to run the *same instruction in lockstep*. The key property: a warp's execution context **stays on chip**, so switching between warps has **zero overhead**. A Volta SM has **4 warp schedulers**, each feeding 32 CUDA cores, 8 load/store units, and 8 special function units.

**Scheduler policy: readiness, not fairness [s15].** Each cycle the scheduler *scans* the resident warps, *filters* down to the eligible ones (those whose next instruction's operands are ready), and *issues* an instruction. It does **not** track which warp has waited longest and does **not** enforce turn-taking — the policy prioritises **throughput over fairness**, so any eligible warp may be chosen.

**Why this hides latency [s16].** When a warp stalls — waiting on a memory load, or hitting a branch divergence — the scheduler instantly switches to another *ready* warp with zero overhead. This warp-switching is *the* mechanism by which GPUs hide latency. It only works if there are enough warps to switch to: **high occupancy** means many warps are resident on the SM, so the scheduler always has a ready warp to run. (Active warp → stalls on memory/divergent branch → instant switch → ready warp takes over.)

**A100 SM capacity at a glance [s17].** One A100 SM can hold **64 resident warps** simultaneously = **2048 active threads** (64 × 32) all resident on a single SM. It has **4 independent schedulers** issuing in parallel, and a typical upper bound of **16 warps eligible** in any given cycle.

## The host–device link and silicon economics [s18–s19]

**PCIe is the seam [s18].** The CPU die and GPU die are physically separate, connected by the **PCIe bus** — drawn as Gen4 ×16 at ~32 GB/s bidirectional, carrying host-to-device (H2D) and device-to-host (D2H) traffic. This link is a recurring performance theme for the rest of the course.

**Why HBM3 rules — the bandwidth wall [s19].** Modern AI models are **memory-bound, not compute-bound**. A single H100 can do 989 TFLOPS of FP8 math *only if data arrives fast enough*. DDR5 at ~100 GB/s is a severe bottleneck; HBM3 delivers **3.35 TB/s**. The bandwidth ladder: DDR5 ~100 GB/s, HBM2e ~800 GB/s, HBM3 3.35 TB/s — but HBM3 costs ~10× more per GB than DDR5, the price of breaking the wall. Alongside bandwidth, precision has evolved to raise throughput: FP32 is the baseline, TF32 gives 2× tensor throughput, FP8 gives 4× vs. FP32, and the Transformer Engine gives 6× (FP8 training vs. prior-gen FP16).

## Scaling beyond one chip: interconnects [s20–s22]

**No GPU is an island [s20].** Training frontier models needs dozens-to-hundreds of GPUs acting as one system, and interconnects are the nervous system. Three matter: **NVLink** (direct GPU-to-GPU at "memory speed" — NVLink 4.0 on Hopper is 900 GB/s bidirectional, 7× faster than PCIe Gen5); **NVSwitch** (a switching fabric enabling 256-GPU clusters like the DGX H100 SuperPOD, giving every GPU uniform low-latency access to every other); and **PCIe Gen5** (the traditional host link at 64 GB/s, which modern multi-GPU systems increasingly *bypass* with direct GPU-to-GPU networking).

**NVLink generations [s21].** 1.0/Pascal: 40 GB/s per link × 4 = 160 GB/s (P100). 2.0/Volta: 50 × 6 = 300 GB/s (V100). 3.0/Ampere: 50 × 12 = 600 GB/s (A100). 4.0/Hopper: 50 × 18 = 900 GB/s (H100). 5.0/Blackwell: 100 × 18 = 1800 GB/s (B200). In a DGX H100, eight H100s connect through NVSwitch as a fully non-blocking all-to-all mesh, each GPU getting 900 GB/s aggregate peer bandwidth — enabling full model-parallel training without PCIe bottlenecks.

**Interconnect comparison [s22].** PCIe 4.0 ×16: 32 GB/s, ~2 µs. PCIe5: 64 GB/s, ~1.5 µs. NVLink 4.0: 900 GB/s, ~300 ns, full mesh. NVLink-C2C: 900 GB/s, ~200 ns, chip-to-chip to the Grace CPU die. AMD Infinity Fabric 3.0: 348 GB/s, ~400 ns. CXL 3.0: 32 GB/s, ~1 µs, coherent. Bottom line: NVLink 4.0 delivers **14× the bandwidth of PCIe 4.0** at sub-microsecond latency — the differentiator for distributed training.

## Generations and the three data-center GPUs [s23–s25]

**Compute Capability (CC) [s23]** tracks architectural evolution: Pascal (2016, CC 6.x) — first HBM2, unified memory, NVLink 1.0; Volta (2017, CC 7.0) — Tensor Cores debut, independent thread scheduling per warp; Ampere (2020, CC 8.x) — refined Tensor Cores with TF32/FP64, Multi-Instance GPU (MIG), 3rd-gen NVLink at 600 GB/s; Hopper (2022, CC 9.0) — Tensor Memory Accelerator (TMA), Thread Block Clusters, native FP8, NVLink 4.0 at 900 GB/s.

**The 14-year arc [s24].** Fermi (2010, 40 nm) → Kepler (2012, 28 nm) → Maxwell (2014, 28 nm) → Pascal (2016, 16 nm) → Volta (2017, 12 nm) → Turing (2018, 12 nm) → Ampere (2020, 7 nm) → Ada (2022, 4 nm) → Hopper (2022, 4 nm) → Blackwell (2024, 4 nm). Process scaling from 40 nm to 4 nm gave ~10× transistor density; memory went from GDDR5 (~300 GB/s) to HBM3e (~8 TB/s), a **26× bandwidth increase**.

**Full spec table [s25].** Learn to read this — it recurs throughout the course:

| Spec | A100 (Ampere) | H100 (Hopper) | B200 (Blackwell) |
|---|---|---|---|
| Process | TSMC 7 nm | TSMC 4N | TSMC 4NP |
| CUDA cores | 6,912 | 14,592 | 20,736 |
| Tensor cores | 3rd gen (432) | 4th gen (456) | 5th gen (576) |
| FP16 tensor (dense) | 312 TFLOPS | 989 TFLOPS | 2,250 TFLOPS |
| FP8 tensor (sparse) | — | 1,979 TFLOPS | 4,500 TFLOPS |
| Memory | HBM2e, 80 GB | HBM3, 80 GB | HBM3e, 192 GB |
| Mem bandwidth | 2.0 TB/s | 3.35 TB/s | 8.0 TB/s |
| L2 cache | 40 MB | 50 MB | 192 MB |
| NVLink | 3.0, 600 GB/s | 4.0, 900 GB/s | 5.0, 1.8 TB/s |
| PCIe | 4.0 ×16 | 5.0 ×16 | 5.0 ×16 |
| Transistors | 54 B | 80 B | 208 B |
| TDP | 400 W | 700 W | 1,200 W |

## Ways to program a GPU, and the specialization payoff [s26–s27]

**Six programming routes [s26]:** CUDA C/C++ (low-level, maximum control, best for custom kernels/research); CUDA Fortran (full GPU access for Fortran scientific codes); cuBLAS/cuDNN (high-level optimised library APIs); OpenACC (directive-based `#pragma` on existing code); OpenMP GPU (standard parallel programming with target offloading); Python/CuPy (NumPy-compatible GPU arrays for prototyping and data science).

**Specialization is the multiplier [s27].** Over roughly a decade, general-purpose CUDA compute grew ~**10×**, but AI tensor compute grew ~**2380×** (P100 → Hopper, FP8). The architecture shifted from maximising FLOPS across *all* workloads to maximising *matrix-multiply* throughput for the specific tensor shapes that dominate transformer models.

---

# Part 2 — The CUDA programming model [s28–s57]

## The three pillars [s28–s30]

CUDA programming rests on three pillars [s29]: **thread hierarchy** (Grid → Block → Warp → Thread), **indexing math** (computing a global index), and **kernel execution** (the host launches, the device runs). The execution model [s30] is a **host (CPU) + device (GPU)** split: the program runs serial code on the host, launches a parallel *kernel* onto a grid of blocks on the device (`Kernel0<<<...>>>()`), returns to serial host code, then launches the next kernel — all under the **SIMT** (Single Instruction, Multiple Thread) model.

## Function specifiers: where code runs [s31]

CUDA adds function qualifiers that decide *where a function is callable from* and *where it executes*:

- `__global__` — a **kernel**: called from the host, executed on the device.
- `__device__` — a device function: called from the device, executed on the device.
- `__host__` — a host function (the default): called from the host, executed on the host.
- `__host__ __device__` — compiled for **both**, so the same function works on CPU and GPU.

## Launching arrays of threads [s32–s34]

**The launch [s32].** A CUDA kernel runs as a *grid* of threads. All threads in the grid run the same kernel code; each has a unique ID, `threadIdx.x`. The **execution configuration** `<<<numBlocks, numThreads>>>(arguments)` sets the thread hierarchy for the launch. A block has a fixed number of threads that are guaranteed to run simultaneously on the same SM. Crucially, **launching a kernel is asynchronous** with respect to the CPU — `cudaDeviceSynchronize()` blocks the host until the device finishes. The canonical body has each thread read its own element, compute, and write its own element:

```c
float x = input[threadIdx.x];
float y = fun(x);
output[threadIdx.x] = y;
```

**Minimal kernel [s33].** `__global__ void` marks a GPU function, invokable from CPU or GPU; it must be launched with an execution configuration `<<< >>>`; and because the launch is async, you call `cudaDeviceSynchronize()` to wait for it:

```c
__global__ void GPUFunction() {
    printf("function runs on the GPU.\n");
}

int main() {
    CPUFunction();
    GPUFunction<<<1, 1>>>();
    cudaDeviceSynchronize();
}
```

**Serial vs. parallel [s34].** `onGPU<<<1, 1>>>()` launches one block of one thread — it runs *once*, like a normal sequential call, with no parallelism. `onGPU<<<1, N>>>()` launches one block of N threads — all N run simultaneously in the same block. The key principle: the GPU runs the *same* kernel across thousands of threads, each computing one independent piece of the larger problem.

## Compilation: nvcc and the build pipeline [s35–s37]

**nvcc splits the work [s36].** When compiling a `.cu` file, nvcc runs **two parallel compilation paths**: the host C/C++ code is compiled and linked with the system compiler, while the device code is compiled for the GPU. A typical invocation:

```
nvcc -arch=sm_80 -o out CUDA.cu -run
```

`-arch=sm_80` names the target architecture (`sm_80` = Tesla A100); `-run` executes the compiled binary. You can inspect the device with `nvidia-smi` or the `deviceQuery` sample. The device path is **CUDA C++ → PTX → SASS** (via `ptxas`): the resulting object file holds host object code *plus* embedded GPU code (PTX and/or CUBIN), ready for linking.

**Recommended project structure [s37].** Keep application logic in `.cpp` files and GPU-specific implementations in `.cu` files, joined by lightweight wrapper headers — this improves organisation, gives better IDE support for host code, and speeds up incremental compilation. **Kernel definitions and launches must stay in `.cu` files**, but runtime API calls like `cudaMalloc` can appear in `.cpp` files via `<cuda_runtime.h>`. Watch for **register spilling** — the most common compiler-induced bottleneck; monitor per-kernel register usage with `nvcc --ptxas-options=-v`. The full pipeline is CUDA C++ → **PTX** (virtual, hardware-independent ISA) → **SASS** (native GPU machine code) → GPU execution.

## Exercise 1 — accelerating a loop [s38–s39]

The first exercise [s38] is to convert a serial CPU loop into a parallel CUDA kernel: replace `for (int i = 0; i < N; i++)` with parallel thread execution, let each thread handle one iteration using `threadIdx.x` as the index, launch with `<<<1, N>>>`, and add `cudaDeviceSynchronize()`. The starting point [s39] is a serial `loop(int N)` that prints each iteration index; the goal is N iterations → N threads, each printing its own index.

## Device functions and a worked kernel [s40–s41]

A `__device__` helper can be called from inside a kernel [s41]. The example computes Euclidean distance for a million random 2D point pairs:

```c
// Device helper — runs on the GPU, called from a kernel
__device__ float computeDistance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

// Kernel using the device function
__global__ void calculateDistances(float* x1, float* y1, float* x2, float* y2,
                                    float* distances, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        distances[idx] = computeDistance(x1[idx], y1[idx], x2[idx], y2[idx]);
    }
}

// Host+device function with the same signature usable in both worlds
__host__ __device__ float clamp(float value, float min_val, float max_val) {
    return fmaxf(min_val, fminf(max_val, value));
}
```

The lesson: `distance = sqrt((x2−x1)² + (y2−y1)²)`, computed for N = 1 million points, one thread per point, with an `if (idx < N)` bounds guard.

## The GPU is a co-processor [s42]

The GPU is not standalone — it is a co-processor to the host [s42]. The **thread hierarchy** runs Host → Device → Grid → Blocks → Threads. The **memory hierarchy** gives each block a shared memory, with all threads also reaching global memory; the host talks to device memory through `cudaMalloc`, `cudaMemcpy`, `cudaMemset`, and `cudaFree`.

## SIMT vs. SIMD [s43–s44]

Both SIMD and SIMT get parallelism by broadcasting one instruction to many execution units, but they differ [s43–s44]. Consider `for (int I = 0; I < N; I++) A[I] = B[I] + C[I]`:

- **SIMD — "one instruction, one data chunk" [s43].** A single instruction operates on several packed data elements at once, using *wide vector units* (e.g., 4 or 8 lanes; 128- or 256-bit registers holding 2–16+ values of one type). All lanes must follow the *same* control flow — **no divergence allowed**. Vectorisation matters because it produces good memory-access patterns that maximise bandwidth. (Scalar: 32 loads, 16 adds, 16 stores; SIMD: 8 loads, 4 adds, 4 stores.)
- **SIMT — "one instruction, many threads" [s44].** Uses *scalar* execution units, not rigid wide vectors. 32 threads in a warp share one instruction fetch, executed over several cycles (e.g., 4 cycles on 8 CUDA cores). It gives flexible thread-level parallelism *with* divergence support — a single instruction can take multiple flow paths, so **`if` statements are allowed**. SIMT lets a CUDA GPU do "vector" computation on scalar cores, and is much easier than getting a CPU compiler to auto-vectorise.

## Thread hierarchy and the case for many warps [s45–s49]

**The hierarchy [s45].** A GPU holds hundreds of thousands of grids; multiprocessors number in the tens of thousands; a block of 1024 threads is split into warps of 32, so 1024/32 = **32 warps**. Formally, `Thread ∈ Block ∈ Grid`.

**Why so many warps per SM [s46].** Three reasons: **latency hiding** (switch to a ready warp while one waits on memory or a pipeline dependency), **resource utilisation** (more warps keep the ALUs and memory bandwidth busy and balance load), and **parallelism** (more concurrent threads → better performance on data-intensive work).

**Two ways to feed the scheduler [s47].** Code must be organised to give the scheduler *enough independent operations*. The more warps available, the more context-switching can hide latency. There are two paradigms, and they can be combined:

- **Thread-Level Parallelism (TLP) [s48].** Strive for **high SM occupancy** — provide as many threads to the SM as possible, so a free scheduler always finds a warp to run. Best when each kernel has a *low* number of independent operations. (Illustrated as four threads each doing `x=x+c; x=x+b; x=x+a`, giving independent work across threads.)
- **Instruction-Level Parallelism (ILP) [s49].** Put a *high* number of independent operations *inside* one kernel (each kernel acting on a lot of data). This lets the scheduler stay on the same warp and fully load each hardware pipeline. Note: the scheduler will not pick a new warp until there are eligible instructions ready on the current one.

## From thread index to global index [s50–s57]

**The built-in variables [s50–s52].** With a launch like `performWork<<<2, 4>>>()`:
- `gridDim.x` — number of blocks in the grid (here, 2) [s50].
- `blockIdx.x` — index of a block within the grid (0 or 1); `blockDim.x` — threads per block (here, 4) [s51].
- `threadIdx.x` — index of a thread within its block (0…3) [s52].

**The global index formula [s53].** A block holds at most 1024 threads, so to process more data you coordinate work *across* blocks by mapping data elements to threads with:

```
globalIndex = threadIdx.x + blockIdx.x * blockDim.x
```

With `blockDim.x = 8`: block 0 covers elements {0…7}, block 1 covers {8…15}, block 2 {16…23}, block 3 {24…31} — each block handles a contiguous, non-overlapping range.

**Bounds checking [s54–s55].** Because you usually launch *more* threads than data elements, the kernel **must** check that its global index is `< N` before touching memory [s54]; threads whose index runs off the end simply do nothing. The pattern [s55]: deliberately over-provision threads, pass `N` (the dataset size) into the kernel, compute the global index, and guard the work with `if (idx < N)`. On the A100 the hard limits are: **1024 threads per block**, block dimensions up to 1024 × 1024 × 64, and grid dimensions up to 65535 × 65535 × 65535 (~280,000 billion blocks).

```c
__global__ void vectorSum(int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) { /* only do work if in range */ }
}
```

**Choosing the block size [s56].** Best performance comes from blocks whose thread count is a **multiple of 32**, because of the warp hardware. The standard "round-up" launch computes just enough blocks to cover N:

```c
int N = 100000;
size_t threads_per_block = 256;
size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
kernel<<<number_of_blocks, threads_per_block>>>(N);
```

The `+ (threads_per_block − 1)` term rounds the integer division up, so blocks always cover all elements even when N is not divisible by the block size.

**The one line that matters [s57].** The whole mapping reduces to `int index = threadIdx.x + (blockIdx.x * blockDim.x)` — every thread runs the same program and distinguishes itself only through this index.

---

# Part 3 — The impact of data movement [s58–s72]

## The transfer bottleneck [s58–s62]

**Compute is cheap, moving data is not [s59].** The classic figures: a host (Ivy Bridge EX) at ~670 GFLOPs with 42 GB/s to 32 GB of DDR3, versus a GPU (Tesla K40) at ~4 TFLOPS with 288 GB/s to 12 GB of GDDR5 — but the **PCI Express link between them is only ~8 GB/s**. The GPU is an order of magnitude faster at compute, but the pipe feeding it is far narrower than either side's local memory.

**The five-step processing pattern [s60–s62].** Because CPU and GPU have physically distinct memories joined by PCIe, every GPU program follows the same skeleton: **(1)** allocate GPU memory; **(2)** copy data from CPU to GPU; **(3)** load and run the GPU program, caching data on-chip for performance; **(4)** copy results back from GPU to CPU; **(5)** free GPU memory.

## The GPU memory hierarchy [s63]

Five levels, from smallest/fastest/most expensive down to largest/slowest/cheapest:

| Level | Size | Bandwidth | Latency |
|---|---|---|---|
| Registers | 256 KB / SM | ~50 TB/s | < 1 cycle |
| L1 / Shared memory | 128–228 KB / SM | ~20 TB/s | ~20 cycles |
| L2 cache | 50 MB (H100) | ~10 TB/s | ~200 cycles |
| HBM3 / GDDR6X (VRAM) | 80 GB (H100) | 3.35 TB/s | ~500 cycles |
| NVLink / PCIe (GPU-to-GPU) | multi-GPU | 900 GB/s (NVL) | µs scale |

The take-away is the enormous latency and bandwidth gap between on-chip storage (registers, shared memory) and off-chip VRAM — the gap that all later optimisation aims to exploit.

## Unified (managed) memory [s64–s70]

**The idea [s64].** Unified Virtual Memory (UVM) gives a **single allocation, a single pointer, accessible everywhere** — eliminating explicit copies and greatly simplifying code porting, and letting CPU and GPU share memory to cut total usage. The trade-offs: it can *increase* memory latency, and it gives the programmer **limited control over placement**, since UVM manages placement automatically and not always optimally.

**How it behaves over time [s65–s69].** You allocate with `cudaMallocManaged()`. When the memory is allocated it may not be resident on *either* CPU or GPU yet. When some code (say a CPU `init()`) first touches it, a **page fault** occurs; the fault triggers **migration** of just the demanded pages to where they're needed. Later, when a kernel `work<<<>>>()` on the GPU touches the same data, it faults and migrates again. This fault-and-migrate cycle repeats *any time* the memory is accessed somewhere it is not currently resident — which, done naively, means many small transfers.

**Simplified allocation [s70].** The code change from CPU to managed memory is minimal:

```c
// CPU code                          // CUDA with Unified Memory
int N = 10000;                       int N = 10000;
size_t size = N * sizeof(int);       size_t size = N * sizeof(int);
int *a;                              int *a;
a = (int*) malloc(size);             cudaMallocManaged(&a, size);
free(a);                             cudaFree(a);
```

## Exercise 2 — vector sum with managed memory [s71–s72]

Take a CPU vector-addition function and accelerate it [s72]: **T1** turn `vecSum` into a CUDA kernel; **T2** choose a working execution configuration; **T3** use `cudaMallocManaged` to handle the data transfer.

---

# Part 4 — Mapping software to hardware, and scaling [s73–s79]

**The direct mapping [s73].** Each software abstraction maps onto a hardware unit: **Kernel → GPU** (the full device, e.g. 132 SMs); **Grid → Chip** (all 132 SMs); **Block → SM** (one streaming multiprocessor, a cooperative thread group); **Warp → Warp Scheduler** (32 threads in lockstep on a dispatch unit); **Thread → CUDA Core** (one ALU / FP32 unit, one instruction stream).

**A kernel is not a loop [s74].** In `vector_add.cu`, `__global__ void add(float* A, float* B, float* C) { int i = threadIdx.x; C[i] = A[i] + B[i]; }`, the index `i = threadIdx.x` is unique per thread, system-assigned, and automatic. The kernel is *not* a loop — it is a function executed once per thread, and each thread uses its unique ID to work on one specific element (the slide shows 16 of 65,536 threads all running the same code).

**Transparent scalability [s75].** The same grid of 8 blocks runs correctly on any GPU: a 2-SM GPU runs ~2 blocks at a time (four waves); a 4-SM GPU runs ~4 at a time (two waves). This works precisely because **each block may execute in any order relative to other blocks** — the programmer never assumes an ordering, so the same binary scales up or down with the hardware.

**Multiple blocks per SM [s76–s79].** With `Kernel<<<24, 4>>>()`, blocks are scheduled onto SMs, and — depending on the number of SMs and each block's resource needs — **more than one block can be resident on a single SM** at once. Making the grid dimensions divisible by the SM count helps achieve full SM utilisation; otherwise some SMs sit idle (unused) while others finish the remainder.

---

# Part 5 — Device queries and error handling [s80–s84]

**Don't hard-code the SM count [s81].** The number of SMs varies by GPU, so query it at runtime rather than baking it in: `cudaGetDevice(&deviceId)` returns the active device id, and `cudaGetDeviceProperties(&props, deviceId)` fills a `cudaDeviceProp` struct with the device's properties, including its SM count.

**Enumerating devices [s82].** The standard pattern loops over all devices and prints key properties, including a peak-bandwidth calculation:

```c
#include <stdio.h>
int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
}
```

**Validate against the CPU [s83].** Since GPU results need not be bit-identical, correctness is checked with a tolerance: `validateResults` loops over elements and flags a mismatch when `fabs(hostRef[i] − gpuRef[i]) > 1e-5`, printing the offending index and both values, otherwise reporting "Results match!".

**Wrap every CUDA call [s84].** A small macro checks the `cudaError_t` returned by any runtime call, converts codes to text with `cudaGetErrorString`, and asserts success:

```c
#include <stdio.h>
#include <assert.h>
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}
int main() {
    checkCuda( cudaDeviceSynchronize() );
}
```

---

# Part 6 — Grid-stride loops: handling data larger than the grid [s85–s99]

**The problem [s86–s89].** When the dataset is larger than the total number of launched threads, each thread must handle *more than one* element. The work is assigned with a **grid-stride loop**: a thread's *first* element is its global index `threadIdx.x + blockIdx.x * blockDim.x`, and each subsequent element is reached by adding the total number of threads in the grid:

```
stride = blockDim.x * gridDim.x
```

**The pattern [s90, s92].** The loop lives inside the kernel:

```c
__global__ void kernel(int *a, int N) {
    int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
    int gridStride = blockDim.x * gridDim.x;
    for (int i = indexWithinTheGrid; i < N; i += gridStride) {
        // do work on a[i];
    }
}
```

**Exercise 3 [s91].** Extend the vector-sum kernel to use a grid-stride loop, keep managed memory, and compare performance across problem sizes N = 1<<16, 1<<20, 1<<22, 1<<24.

**A concrete 1-million-element vector add [s93, s96–s97].** For two arrays of 1M elements, one straightforward mapping is `vecAdd<<<1024, 1024>>>(...)` — 1024 blocks of 1024 threads = 1,048,576 threads — with `int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx < n) c[idx] = a[idx] + b[idx];` [s93, s96]. An alternative processes **4 elements per thread**: `idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4`, which needs a bounds check (because 1,048,576 does not divide evenly by the ~253,952 threads available) and lets you cut the block count from 1024 down to 256 [s97].

**How the GPU actually runs it [s94–s95].** Work is broken into blocks, each holding a subset of the threads. On a 124-SM GPU with 2048 threads per SM that is 253,952 threads total [s94]. The GPU works by **oversubscription** [s95]: you queue far more blocks than can run at once; blocks stream onto SMs as they free up, completed blocks exit and free SMs, and this deep queue ensures the GPU is *never idle*. Increasing SM count → more concurrency → higher performance.

**Why the stride loop matters — threads vs. data [s98].** If you launch too few threads (`blockDim.x * gridDim.x < n`) and give each thread only *one* element, only a small fraction of the data (e.g. ~0.1%) is processed and the rest is left untouched → **incorrect results**. The measured cases make it concrete: a small vector (n = 1<<16) passes either way; a large vector with enough threads (n = 1<<24) passes either way; but a large vector with *limited* threads and one-element-per-thread **FAILS** (mismatch at index 16384), while the grid-stride version **PASSES**. The grid-stride loop guarantees the whole range is covered without oversized grids, keeps memory access efficient, and smooths intra-block scheduling jitter.

**Which is better? [s99].** *No stride* (one element per thread): straightforward, efficient for small-to-medium vectors, and gives **coalesced memory access** (good for bandwidth) — but wastes utilisation if n greatly exceeds the launched threads. *With stride*: scalable to very large vectors with a limited thread count, good when `n >> total threads` — but can hurt cache performance (strided accesses have less spatial locality) and, depending on stride size, reduce coalescing.

---

# Part 7 — Profiling tools and zero-copy memory [s100–s105]

**The Nsight workflow (introduced here, detailed later) [s100].** Module 04 is NVIDIA's profiling suite: **Nsight Systems** for the system-wide timeline and **Nsight Compute** for per-kernel analysis.

**Zero-copy memory [s101].** Zero-copy lets GPU threads directly access **pinned host memory** without an explicit transfer, cutting overhead for certain workloads and simplifying code. It is best for small/medium data or frequently updated data where transfer overhead dominates, and it *excels on integrated GPUs* where CPU and GPU share physical memory (eliminating PCIe transfers entirely). The API contrast:

```c
// Traditional
cudaMalloc(&d_data, size);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
kernel<<<>>>(d_data);
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

// Zero-copy — no explicit copy
cudaHostAlloc(&h_data, size, cudaHostAllocMapped);
cudaHostGetDevicePointer(&d_data, h_data, 0);
kernel<<<>>>(d_data);
```

**Zero-copy vs. device memory [s102–s104].** The worked "Vector Sum" allocates host memory with `cudaHostAlloc(..., cudaHostAllocMapped)`, obtains device pointers with `cudaHostGetDevicePointer`, and times the kernel. The comparison table (sizes from 1 KB to 256 MB) shows zero-copy is generally faster for small transfers (e.g. at 1 KB it is markedly quicker), with the advantage shrinking as size grows.

**Choosing a transfer strategy [s105].** A decision table: **large one-time transfer** → pinned + synchronous `cudaMemcpy` (max throughput, simple); **overlapping compute + transfer** → pinned + async + streams (hides transfer latency); **frequent small updates** → zero-copy or UVM (avoids repeated malloc/free); **integrated GPU (laptop)** → UVM or zero-copy (shared physical memory); **multi-GPU** → NVLink P2P `cudaMemcpyPeer` (bypasses the CPU entirely).

---

# Part 8 — Higher-dimensional grids [s106–s115]

**Multidimensional launches [s107–s108].** The host specifies a 3D grid-block-thread configuration at runtime. All threads from one launch form a **grid** sharing one global memory space; a grid is many blocks. CUDA provides built-in 3D types — `dim3` for dimensions, `uint3` for indices — accessed as `threadIdx.{x,y,z}`, `blockIdx.{x,y,z}`, `blockDim.{x,y,z}`. Example: a 3×2 grid = 6 blocks, each 5×3 = 15 threads, totalling 90 threads.

```c
dim3 gridDim(3, 2, 1);    // 6 blocks
dim3 blockDim(5, 3, 1);   // 15 threads/block
kernel<<<gridDim, blockDim>>>(args);   // total: 6 × 15 = 90 threads
```

**Locating a thread in a 2D grid [s109–s111].** With 8×4 = 32 threads (one warp) per block, the global coordinates are `x = blockIdx.x * blockDim.x + threadIdx.x` and `y = blockIdx.y * blockDim.y + threadIdx.y`. Worked flat indices: block (0,0)/thread (0,0) → 0; block (1,0)/thread (0,0) → x = 8, flat 8; block (1,1)/thread (4,1) → x = 12, y = 5, flat 92.

**Warp-level view [s112–s113].** A launch of 4 blocks × 32 threads = 128 threads = 4 warps (1 warp/block); scaling to 16 blocks gives 16 warps and 512 threads. Each 32-thread warp executes one instruction per clock in lockstep.

**The general indexing recipe [s114].** 1D: `idx = blockIdx.x*blockDim.x + threadIdx.x`. 2D: compute `row` and `col`, then `idx_2d = row * (gridDim.x*blockDim.x) + col`. 3D: compute x, y, z, then `idx_3d = z*(gridDim.x*blockDim.x)*(gridDim.y*blockDim.y) + y*(gridDim.x*blockDim.x) + x`.

**Reference table [s115].** `dim3 gridDim` = grid dimensions (x,y,z); `uint3 blockIdx` = block index within grid; `dim3 blockDim` = block dimensions; `uint3 threadIdx` = thread index within block. Linear-ID formulas: 1D → `x`; 2D → `x + y·Dx`; 3D → `x + y·Dx + z·Dx·Dy`.

---

# Part 9 — Matrix multiplication [s116–s118]

**Exercise 3b [s116].** Two-matrix multiplication: create a `dim3` configuration with x and y dimensions > 1, define two indices (one per axis), and use pageable and pinned memory (optionally managed memory).

**The math and the GPU strategy [s117].** Standard matrix multiply is `P_ij = Σ_{k=0..n−1} M_ik · N_kj`. The GPU strategy is **one thread per output element** `P[i][j]`, using a 2D grid: `row = blockIdx.y*blockDim.y + threadIdx.y`, `col = blockIdx.x*blockDim.x + threadIdx.x`.

**Why it needs the GPU [s118].** The host version is three nested loops — `O(N³)` operations:

```c
void matrixMultOnHost(float* M, float* N, float* P, int Width) {
    for (int row = 0; row < Width; ++row)
        for (int col = 0; col < Width; ++col) {
            float pval = 0;
            for (int k = 0; k < Width; ++k)
                pval += M[row*Width + k] * N[k*Width + col];
            P[row*Width + col] = pval;
        }
}
```

A 4096×4096 multiply needs 4096³ ≈ **68 billion FLOPs**, so GPU parallelism is essential. This sets up the shared-memory tiling optimisation covered in Part 11.

---

# Part 10 — Memory-hierarchy deep dive: the roofline model [s119–s134]

**Hardware is no longer transparent [s119].** To extract real TFLOPs you must consciously orchestrate *execution* and *data movement* — you cannot treat the hardware as an abstraction.

**The throughput wall [s120].** GPUs can do billions of FLOPS yet most kernels run far below peak, and the culprit is almost always **memory**: global-memory access is orders of magnitude slower than on-chip compute, so cores sit idle waiting for data. Two remedies preview the rest of the module — **tiled matmul** (stage small tiles in fast shared memory to cut redundant global reads and enable coalescing) and fixing **the sequential bottleneck** (in a naive pipeline, transfers and kernel execution happen one after another, leaving cores idle during PCIe transfers).

**The Nsight tool flow [s121].** **Nsight Systems** gives the workload-level timeline (GPU kernels track, memory-transfer track, CPU/GPU overlap, CUDA API calls); **Nsight Compute** gives per-kernel detail (SM throughput, memory bandwidth, roofline chart, warp efficiency, instruction mix). The method: *start* with Nsight Systems for the big picture, *dive* into the hottest CUDA kernels with Nsight Compute, and iterate until performance is satisfactory.

**A profiled vector add [s122–s124].** The kernel uses a grid-stride loop and managed memory, and the launch shows the **four-parameter** triple-chevron `sumArraysOnGPU<<<1, 1, 0, stream>>>(...)` (blocks, threads, shared-memory bytes, stream). Profiling a 50-million-element add (190.73 MB/vector) [s123] shows the parallelism payoff: single thread = 15 MB/s (1×); single block of 256 threads = 1.95 GB/s (134.2×); multiple blocks = 1151 GB/s (**79,036×**). But the **Unified Memory limitation** [s124] appears in the `nsys` stats: 8192 host-to-device and 1142 device-to-host "unified" memcpy operations occur *with no explicit `cudaMemcpy` calls* — because arrays are initialised on the CPU, their pages are CPU-resident, and each GPU access during the kernel triggers an on-demand page fault and migration. Those many small migrations stall GPU threads and erase much of the speedup.

**Prefetching fixes it [s125–s127].** `cudaMemPrefetchAsync()` preloads data to the GPU *before* the kernel runs. The `nsys` "pattern to look at" [s126]: prefetching collapses 8192 H2D transfers down to **192**, and 1142 D2H down to **96** — far fewer, larger, more efficient transfers instead of a storm of page faults. (The comparison also cites pinned transfers at ~5.8 GB/s vs. pageable at ~2.3 GB/s on the sample device.)

**GPU vs. CPU memory trade-offs [s129].** Within an SM: **registers** are fastest and smallest; **L1 cache** is fast, small, on-chip; **shared memory** is medium speed and shared among threads; **global memory** is slowest, largest, off-chip. Relative to global memory, L1 has ~13× the bandwidth and 1/15 the latency, and shared memory ~3× the bandwidth and ~1/3 the latency — the numbers that make on-chip data reuse worthwhile.

**The memory bandwidth wall, stated plainly [s130].** *"A modern GPU can execute trillions of floating-point operations per second — yet most kernels run far below peak capacity. The culprit is rarely compute; it is memory."* Memory bandwidth — the rate bytes move from on-card memory to the compute units — is the true bottleneck in most real HPC and AI workloads.

**Arithmetic intensity [s131–s132].** The key metric is **FLOPs ÷ Bytes** — math operations per byte transferred. It decides whether a kernel sits on the roofline's flat roof (compute-bound) or its sloped ramp (bandwidth-bound). **Low intensity** (fewer than ~1–4 FLOPs/byte) is bandwidth-bound — memory is the limiter; **high intensity** is compute-bound — the GPU is saturated with work. The **simple-addition trap** [s132] makes it vivid: an elementwise `a + b` loads two FP32 values (8 B) and writes one (4 B) = **12 bytes per operation**, giving arithmetic intensity 1 ÷ 12 ≈ **0.08** and only **~5% GPU utilisation**. Low-intensity kernels leave the vast majority of compute power idle.

**The roofline model [s133–s134].** Plot a kernel's arithmetic intensity on the X-axis; its achievable performance is the **lower** of two ceilings [s133]. *Left of the ridge point* is **bandwidth-bound** — performance grows with intensity, data movement is the bottleneck. *Right of the ridge* is **compute-bound** — performance plateaus at peak FLOPs/s, math throughput is the bottleneck. The core formula [s134]:

```
Performance = min( Maximum Compute Throughput,  Memory Bandwidth × Arithmetic Intensity )
```

where compute throughput is the flat ceiling, memory bandwidth is the slope of the ramp, and arithmetic intensity (FLOPs per byte) pushes a kernel rightward toward the compute ceiling.

## Pageable vs. pinned memory [s135–s140]

**Virtual memory recap [s136].** A memory **page** is a fixed-length contiguous block of virtual memory (typically 4 KB) — the fundamental unit of memory management. Each program has its own logical memory broken into pages, mapped to physical memory via a **page table**. The paging process: memory request → page-table lookup (MMU translates virtual → physical) → page-fault check (if not in RAM, load from disk) → data access.

**Why pinned memory is faster [s137].** With **pageable** (normal) host memory, the OS can swap the pages, so a transfer takes a *staging* path: copy into pageable RAM, copy again into a temporary pinned staging buffer the OS allocates, then DMA-copy (blocking) to the GPU — synchronous, high-overhead, low utilisation. With **pinned (page-locked)** memory, the buffer is locked in RAM and cannot be swapped, so the GPU's DMA engine transfers *directly* — enabling faster CPU→GPU transfers, asynchronous copies, and overlap of transfer with compute.

**The code difference [s138].** Pinned memory just swaps the host allocator and deallocator:

```c
// Pageable                                    // Pinned
h_a = (float*) malloc(nbytes);                 cudaMallocHost(&h_a, nbytes);
cudaMalloc(&d_a, nbytes);                      cudaMalloc(&d_a, nbytes);
cudaMemcpy(d_a, h_a, nbytes, H2D);             cudaMemcpy(d_a, h_a, nbytes, H2D);
kernelGPU<<<>>>(..., d_a, ...);                kernelGPU<<<>>>(..., d_a, ...);
cudaMemcpy(h_a, d_a, nbytes, D2H);             cudaMemcpy(h_a, d_a, nbytes, D2H);
cudaFree(d_a); free(h_a);                      cudaFree(d_a); cudaFreeHost(h_a);
```

**Benchmark [s139–s140].** On an A100-SXM-64GB with 16 MB (4,194,304 elements), pageable transfers reach ~8.05 GB/s (H2D) / 9.96 GB/s (D2H) while pinned reaches ~24.28 / 24.03 GB/s; the pinned vector add runs in 2.2 ms vs. 5.4 ms — pinned memory is **~2.48× faster**. The Nsight timeline [s140] confirms it: for pageable the run is dominated by memcpy (60.5% HtoD, 39.5% DtoH; VectorAdd_Pageable 4.902 ms), while the pinned version finishes the same work in 2.202 ms.

---

# Part 11 — Streams: asynchronous concurrency [s141–s159]

**What a stream is [s141].** A **CUDA stream** is an ordered **FIFO queue** of operations — kernel launches, memory copies, synchronization events — submitted to the GPU. Operations *within one stream* execute strictly in order, but **different streams can execute concurrently**. Streams are the fundamental unit of GPU concurrency control. The paradigm shift: synchronous execution forces the host to block and wait for every GPU op; asynchronous streams let the CPU enqueue work and immediately continue, overlap memory transfers with kernel execution, run multiple independent work queues in parallel, and use memory bandwidth and compute units simultaneously — the goal being to **hide memory latency behind computation**.

**The rules [s142–s144].** Multiple (non-default) streams can be created; kernels within any single stream must execute in order; kernels in *different* non-default streams may run concurrently with no fixed ordering [s142]. By default, kernels go into the **default stream (stream 0)**, where they run strictly in issue order [s143]. The payoff is overlap [s144]: instead of serial `H2D → kernel → D2H`, you pipeline three chunks so that while chunk 1's kernel runs, chunk 2's H2D copy and chunk 0's D2H copy proceed in parallel — a clear performance improvement.

**Implementation [s145].** Three steps: **(1)** create a stream — `cudaStream_t stream; cudaStreamCreate(&stream)`; **(2)** launch into it — `kernel<<<blocks, threads, smem, stream>>>(args)`; **(3)** copy asynchronously — `cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream)`. `cudaMemcpyAsync` with **pinned host memory** frees the CPU to continue while data moves.

**Pinned memory is mandatory here [s146].** Standard heap memory is pageable and the OS can relocate it, breaking DMA. Pinned (page-locked) memory guarantees a fixed physical address so the DMA engine can transfer directly. Allocate with `cudaMallocHost` or `cudaHostAlloc`; it is **mandatory for `cudaMemcpyAsync`**, and lets you stream data in chunks that bypass device-memory limits.

**Worked examples [s147–s148].** The two-stream version [s147] splits the data (first half → stream1, second half → stream2), issues two async copies, two kernels, and two copies back, then `cudaStreamSynchronize`s both. The four-stream version [s148] splits 1M elements across four streams in a loop — each iteration does an async H2D copy at its offset, a `processChunk` kernel, and an async D2H copy — then waits on and destroys every stream. In Nsight Systems [s149], a matmul across streams shows overlapping activity on several streams (36.3% default stream 7, plus streams 16/17/20), with "Overlapped Chunked Execution" and "Pipelined Execution" regions confirming the concurrency.

**Exercise 4 [s150].** Refactor `Vector_Add_streams.cu` to use non-default streams: it currently launches an initialisation kernel three times (one per vector), so launch each initialisation in its *own* non-default stream and confirm the success message prints.

## Streams applied to tiled matrix multiplication [s151–s158]

**The hardware engine model [s152].** A GPU is not one processor but several **engines** running simultaneously: an **execution engine** (kernels/compute) and dedicated **copy engines** (H2D/D2H DMA). Because **PCIe is full-duplex**, you can upload data and download results at the same time — which is exactly what streams exploit.

**Matmul context [s153].** Matrix multiply (`C_ij = Σ_{k} A_ik · B_kj`) is the workhorse of deep learning, and its challenge is that **global-memory bandwidth is the bottleneck** — you must minimise access to slow VRAM. The efficiency metrics to watch: arithmetic intensity (ops/byte), occupancy (active vs. max warps), and transfer overhead (idle GPU time).

**Memory reuse via tiling [s154–s155].** Global memory (VRAM) is far — ~400–800 cycles of latency — while shared memory (SRAM) is near — ~20–40 cycles. The goal for matmul is **memory reuse**: if you read each element from global memory N times, you are bandwidth-limited rather than compute-limited [s154]. The **tiling pattern** [s155] fixes this: instead of reading whole matrices, threads cooperatively load small sub-matrices (tiles) into shared memory, reuse each tile many times from SRAM, and thereby **reduce global-memory traffic by a factor of the tile width**.

**The combined payoff [s156–s157].** Stacking optimisations on matmul [s156]: a single serial stream reaches 450 GFLOPS; shared-memory tiling (no streams) reaches 820 GFLOPS; shared memory **plus** streams reaches 1.2 TFLOPS — streams add 30–50% effective throughput on memory-bound tasks. The global-vs-shared comparison [s157] shows shared-memory tiling pulling further ahead as the problem grows: at 4096³ the shared version hits a 496.68× speedup over CPU vs. 293.56× for the global-memory version.

**When to use streams [s158].** Three situations: **latency hiding** (when transfer time is comparable to kernel time), **task parallelism** (many independent small kernels that don't individually fill the GPU), and **multi-GPU orchestration** (needed for peer-to-peer copies and multi-device pipelines).

## Common pitfalls and close [s159–s160]

Six recurring mistakes and their fixes [s159]: **race conditions** → proper synchronization; **memory leaks** → RAII patterns; **uncoalesced memory access** → optimise access patterns; **low occupancy** → balance resources and threads; **excessive host-device transfers** → use streams and batching; **ignoring error checking** → always check CUDA API return codes. The deck closes on a questions slide [s160].

---

# Likely exam angles

These are the recurring, testable ideas the deck keeps returning to — the places an exam question is most likely to land.

**Occupancy and latency hiding.** Be ready to explain *why* GPUs need many resident warps: the SM hides memory and pipeline latency by switching to another **ready** warp with **zero overhead** [s14–s16]. "High occupancy" means many warps are resident so the scheduler always has a ready warp. Know the A100 numbers: 64 resident warps, 2048 threads, 4 schedulers, ~16 eligible warps/cycle per SM [s17]. Understand the scheduler picks on **readiness, not fairness** [s15], and that latency can be hidden via **TLP** (more threads/occupancy) or **ILP** (more independent instructions per thread), which can be combined [s47–s49].

**Coalescing.** Coalesced (contiguous, aligned) global-memory access maximises effective bandwidth; it is the main advantage of the one-element-per-thread pattern [s99]. Strided grid-stride access can *reduce* coalescing and spatial locality — a classic trade-off question [s98–s99]. Uncoalesced access is listed explicitly as a pitfall [s159].

**Warp divergence.** Because a warp executes one instruction in lockstep across 32 threads, SIMT *allows* `if` statements but divergent branches serialise the paths — the divergence that makes a warp stall and forces a scheduler switch [s16, s44]. Contrast with SIMD, which forbids divergence entirely [s43].

**When shared memory helps.** Shared memory (SRAM, ~20–40 cycles, ~3× global bandwidth) pays off when data is **reused** many times — the tiled matrix-multiply case, where staging tiles in shared memory cuts global traffic by a factor of the tile width and turns a bandwidth-bound kernel toward compute-bound [s120, s154–s157]. If data is read once, shared memory buys little.

**The roofline / arithmetic intensity.** Be able to compute arithmetic intensity = FLOPs ÷ bytes, classify a kernel as bandwidth- vs. compute-bound, and apply `Performance = min(peak compute, bandwidth × intensity)` [s131–s134]. The canonical worked case: elementwise `a+b` moves 12 bytes per FLOP → intensity ≈ 0.08 → ~5% utilisation, so it is hopelessly bandwidth-bound [s132].

**Indexing and launch configuration.** Reproduce `globalIndex = threadIdx.x + blockIdx.x*blockDim.x`, the round-up block count `(N + tpb − 1)/tpb`, why blocks should be a multiple of 32, the `if (idx < N)` bounds guard, and the grid-stride loop with `stride = blockDim.x * gridDim.x` [s53–s57, s90]. Extend to 2D/3D indexing [s114–s115].

**Data movement.** Know the five-step host/device pattern [s60–s62], pageable vs. pinned (pinned enables direct DMA and async copies, ~2.48× faster in the benchmark) [s137–s140], unified memory and its page-fault migration cost, fixed by `cudaMemPrefetchAsync` [s64–s70, s124–s127], and streams for overlapping transfer with compute on full-duplex PCIe [s141–s145, s152].

**nvcc and errors.** The `-arch=sm_80` flag targets the A100; the device path is CUDA C++ → PTX → SASS; kernels/launches stay in `.cu` files; wrap runtime calls in a `checkCuda`/`cudaError_t` handler and validate GPU output against the CPU within a tolerance [s36–s37, s83–s84].

---

*Gaps: none. All 160 slides were read (the deck ends at slide 160, "Questions?"); a few slides are near-duplicate section headers or repeated code frames and were folded into the adjacent explanation rather than listed separately.*
