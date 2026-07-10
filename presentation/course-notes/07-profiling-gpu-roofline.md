# Module 7 — Profiling II: GPU tools (nsys/ncu) & the roofline model

This module covers two lecture decks:

1. **`gpu_profiling.pdf`** — *"GPUs: bottlenecks in accelerated codes"* (L. Bellentani, CINECA; 77 slides). This is the practical, tool-driven deck: how a GPU program actually spends time, and how to *see* that time with **NVIDIA Nsight Systems (`nsys`)**, plus **Score-P** and **Vampir** for MPI+GPU codes.
2. **`GPURooflineModelForHPC.pdf`** — *"GPU Roofline Model"* (CINECA for Politecnico di Milano; 18 slides). This is the analytical deck: the **roofline model**, **arithmetic intensity**, and how to diagnose whether a kernel is memory- or compute-bound with **NVIDIA Nsight Compute (`ncu`)**.

> Slide references are written `[s N]` (or `[s A–B]` for ranges) and are **scoped to the deck named in the section heading** you are reading. Section 1 = deck 1 (`gpu_profiling.pdf`); Section 2 = deck 2 (`GPURooflineModelForHPC.pdf`).

The two decks fit together: deck 1 teaches you to *find* the bottleneck on a timeline (which operations dominate, whether data movement is drowning your kernels, whether you overlap work); deck 2 teaches you to *classify* a single kernel's bottleneck (memory-bound vs compute-bound) and decide which optimization will actually help.

---

## Section 1 — GPU tracing with Nsight Systems (`gpu_profiling.pdf`)

### 1.1 Why GPUs at all: the paradigm shift `[s1–5]`

The deck opens with a short history of parallel-computing scale `[s2–5]`. In the **1990s** the advent of **MPI** brought *distributed programming* across many CPUs `[s2]`. For years, machines grew by *increasing the density and frequency of conventional CPU architecture* `[s3]`. The **petascale** era arrived in **2008** with the **Roadrunner** supercomputer at Los Alamos, reaching **10¹⁵ FLOP/s** `[s4]`. The **exascale** era arrived in **2022** with **Frontier** at Oak Ridge, reaching **10¹⁸ FLOP/s** — and it got there with *new hardware: Graphics Processing Units (GPUs)* `[s5]`. The takeaway: modern peak performance comes from GPUs, so understanding GPU behavior is essential.

### 1.2 GPU vs CPU architecture `[s6–8]`

A **CPU** has *several strong cores* built for *specialized processing*, backed by a deep cache hierarchy — per-core L1, then L2, then a shared L3, then DRAM `[s6]`. A **GPU** instead has *thousands of (weaker) cores* designed for *embarrassingly parallel processing*, with a much shallower hierarchy — a large array of cores over a single L2 cache and DRAM `[s7]`.

The programming consequence is shown with a trivial loop `[s8]`:

```c
for (int i = 0; i < N; i++) {
    a[i] = 2 * a[i];
}
```

Each iteration is independent, so element `a[0]` goes to thread `th0`, `a[1]` to `th1`, and so on up to `a[N]`. The GPU *supports a very large amount of parallelism* precisely because there are thousands of cores to map those independent iterations onto `[s8]`. **Lesson:** GPUs win only when you can expose enough independent work to feed all those cores.

### 1.3 Heterogeneous programming and the anatomy of an offloaded region `[s9–20]`

A GPU program is **heterogeneous**: the application is mostly sequential CPU code, and you offload the *compute-intensive functions* — a *small percentage of the code* that accounts for a *large percentage of the runtime* — to the GPU `[s9]`. The critical constraint: **GPU and CPU have separate memories**, so the data must be physically present in GPU memory before a kernel can use it `[s10]`.

Slides `[s11–20]` build up, piece by piece, the **trace of a single offloaded region** — this is the mental model you will later recognize on a real Nsight Systems timeline. Two horizontal timelines run in parallel, one for the **CPU** and one for the **GPU**. Inside the CPU's "offloaded region" the following happen in sequence:

1. **DATA IN** `[s13–14]` — the input data is copied from CPU memory to GPU memory (host-to-device).
2. **L (launch)** `[s15]` — a short launch step where the CPU tells the GPU to start; this launch takes time (**latency**).
3. **GPU KERNEL** `[s16]` — the kernel actually runs on the GPU (the cyan bar on the GPU line).
4. **WAIT** `[s17–18]` — meanwhile the CPU sits idle *waiting* for the kernel to finish.
5. **DATA OUT** `[s19]` — the results are copied back from GPU to CPU memory (device-to-host).

The finished picture `[s20]` shows the GPU line as `DATA IN → GPU KERNEL → DATA OUT` and the CPU line as one long `OFFLOADED REGION` decomposed into `DATA IN`, `L`, `WAIT`, `DATA OUT`. **This is the vocabulary of every GPU timeline: kernels, memory copies, launch latency, and host wait time.**

### 1.4 The cost of data movement `[s21–27]`

Slides `[s21–27]` re-draw the hardware to hammer home the single most common GPU bottleneck. The diagram shows the CPU's *High Capacity Memory* and the GPU's *High Bandwidth Memory (HBM)* connected by an **IO bus** (this is PCIe/NVLink) `[s21]`, with a red "YOU ARE HERE" dot on the CPU because *the program starts on the CPU* `[s21]`.

The chain of facts `[s22–24]`:

- GPUs and CPUs have **separated memories** `[s22]`.
- GPUs **need the data in their own memory** `[s23]`.
- The **IO bus is slow compared to GPU bandwidth** `[s24]` — moving data across the bus is far slower than the GPU reading its own HBM.

The conclusion is stated in two boxes: **"MOVING DATA IS EXPENSIVE"** `[s25–26]` and therefore **"IMPROVE DATA LOCALITY"** `[s27]`. The accompanying timeline shows that when you move data repeatedly, the CPU line fills with red (data) and yellow (wait) boxes and the GPU does little useful work `[s26]`. **Lesson:** the first thing to check on any GPU profile is whether host↔device copies are dominating.

### 1.5 Offload options and kernel-launch latency `[s28–34]`

There are *many ways to offload kernels to GPUs* `[s28–29]`, arranged by *increasing programming effort* `[s28]`:

- **Libraries** — *"drop-in" acceleration* (least effort; e.g. cuBLAS).
- **OpenACC Directives** — *easily accelerate applications* (pragmas).
- **Programming Languages** — *maximum flexibility* (e.g. CUDA C/C++/Fortran).

Whichever route you take, **launching a kernel is not free**: *there is a time needed to launch the kernels ("latency")* `[s31]`. On the timeline the CPU issues a **LAUNCH** and then a **WAIT**, and the gap before the kernel begins on the GPU is the launch latency `[s32]`. When a program launches *many small kernels back-to-back* `[s33]`, this per-launch overhead accumulates. Hence the two summary boxes: **"KERNEL LAUNCH ADDS OVERHEAD"** and **"EXPOSE AS MUCH PARALLELISM AS POSSIBLE"** `[s34]` — fewer, bigger kernels that do more work per launch amortize the overhead better than many tiny ones.

### 1.6 Nsight Systems: the tool taxonomy `[s35–36]`

The tooling section opens with the **Nsight Systems** title slide — *"Tracing GPUs and CPUs"* `[s35]`. Nsight Systems (`nsys`) is a **system-wide, timeline-oriented** profiler.

The measurement taxonomy `[s36]` classifies profiling along three axes:

- **Sampling vs Instrumentation** — *sampling* periodically interrupts and records where the program is (statistical, low overhead); *instrumentation* inserts probes that record every occurrence of an event exactly.
- **GPU vs CPU** — the tool measures both sides of the heterogeneous program.
- **Traces vs Summaries** — a *trace* is the full time-ordered timeline; a *summary* is aggregated statistics.

What `nsys` can capture `[s36]`:
- **Instrument** CUDA APIs, OpenACC regions, MPI and OpenMP (CPU side), and GPU kernels / GPU metrics.
- **Sample** the CPU (user routines, libraries).
- Collect **network metrics** (host-to-device *H2D*, device-to-device *D2D*, and NIC metrics).

### 1.7 The `nsys` command line `[s37–38]`

The general form `[s37]`:

```
nsys [command_switch] [optional command_switch_options] [application] [optional application_options]
```

**Command switches** `[s37]`:
- **`profile`** — the all-in-one command; *needed for concurrent profiling sessions*. This is what you use most of the time.
- `start` · `launch` · `cancel` · `shutdown` · `stop` — *interactive mode* (drive a session step by step).
- `export` · `stats` · `analyze` — *post-processing* to export data or produce textual summaries.

**Command switch options** `[s37]`:
- **`--trace=[nvtx,cuda,osrt],mpi,openmp,openacc,cublas,cusolver,...`** — which classes of events to trace. (`osrt` = OS runtime.)
- `--cuda-memory-usage` — collect GPU memory usage.
- `--mpi-impl` — select the MPI implementation (`openmpi` or `mpich`).
- `--nic-metrics` — network bandwidth.

**CINECA cluster gotcha** `[s38]`: to avoid saturating `/tmp`, wrap the `nsys` call so its scratch goes to your job's temp directory:

```bash
rm -rf /tmp/nvidia
ln -s $TMPDIR /tmp/nvidia
nsys <...>
rm -rf /tmp/nvidia
```

### 1.8 Flat profiles from CPU sampling `[s39]`

A **flat profile** answers "which function used the most CPU time" by *sampling* the CPU `[s39]`:

```bash
nsys profile --sample=cpu --backtrace=fp
```

`--backtrace=fp` uses frame-pointer backtraces to attribute each sample to a call stack. **Critical caveat from the slide:** *debug symbols are needed for backtraces* `[s39]` — without them the profile shows "[Broken backtraces]" and you cannot tell which function a sample belongs to. The "Flat View" lists each symbol with its self-time percentage.

### 1.9 Analysis summaries `[s40]`

Nsight Systems produces an **Analysis Summary** describing the run itself `[s40]`: the profiling session duration (e.g. `01:57.999`), the **Target** (local/UTC time, platform, OS, CPU description, whether GPU context switch is supported), a **Process summary** and **Module summary** (CPU time per module, both overall and per process), a **Thread summary** (per-thread CPU utilization), and the captured **Environment Variables**. This is the context you record so a profile is reproducible.

### 1.10 Top-down / bottom-up profiles `[s41]`

Sampling data can be viewed as a **call tree**. The goal `[s41]` is to *identify the most time-consuming calling path*. A **Bottom-Up View** starts at the leaf functions (where time is actually spent) and walks up to their callers, answering the two questions posed on the slide: *"Which is the most time-consuming path to offload?"* and *"Which are the parent routines potentially involved?"* — i.e. it tells you which region of code is worth porting to the GPU.

### 1.11 The timeline view `[s42]`

The **timeline** is the heart of Nsight Systems `[s42]`:

```bash
nsys profile --trace=osrt,cuda --sample=cpu --backtrace=fp
```

Rows are organized by **location**, which can be `[s42]`:
- **architecture** (a CPU core or the GPU processing unit),
- **stream** (a GPU stream / queue),
- **thread**,
- **library** (CUDA, CUPTI),
- **runtime** (OpenACC).

In the example the CUDA hardware row is a Tesla V100, and the GPU work is split across streams: *50.1% on the default stream 7* (itself *78.2% Kernels, 21.8% Memory*), plus *35.5% on stream 37* and *14.4% on stream 38* `[s42]`. Hovering a sample shows the call stack that produced it. **This is where you visually confirm kernel/memcpy overlap, gaps (idle GPU), and which stream does what.**

### 1.12 NVTX: annotating your own code `[s43–48]`

A raw timeline is hard to read because *source code is not automatically instrumented* `[s44]`. **NVTX** (the NVIDIA Tools Extension, header `nvToolsExt.h`) is a *C-based API to annotate events and code ranges* so they appear as labeled, colored bars on the Nsight Systems timeline `[s44]`. It adds *limited overhead when the tool is not attached* `[s44]`, so you can leave the annotations in. On the slide, adding NVTX ranges turns an unreadable timeline into clearly labeled regions like `PHONON`, `phqscf`, and `solve_linter` `[s43]`.

Two functionalities `[s44]`:
- **NVTX Markers** — annotate an **event occurring at a specific instant**.
- **NVTX Ranges** — annotate the **timespan of a code region** (ranges nest on a CPU thread).

Each call comes in three **variants** depending on the payload `[s44–45]`:
- **`A`** — message is ASCII.
- **`W`** — message is Unicode (wide).
- **`Ex`** — takes a full attribute *structure* (lets you set color, message type, etc.).

The API `[s46]`:

```c
// Markers (events at a specific time)
nvtxMarkA(__FUNCTION__ ":nvtxMarkA");
nvtxMarkW(__FUNCTIONW__ L":nvtxMarkW");
nvtxMarkEx(&eventAttrib);

// Ranges (nested time ranges on a CPU thread)
nvtxRangePushA(__FUNCTION__ ":nvtxRangePushA");  // message-only
    // ... code here ...
nvtxRangePop();

nvtxRangePushEx(&eventAttrib);                    // structured event
    // ... code here ...
nvtxRangePop();
```

The **event attribute structure** for the `Ex` variants is filled in like this `[s44]`:

```c
nvtxEventAttributes_t eventAttrib = {0};              // set to default
eventAttrib.version = NVTX_VERSION;                   // version and size
eventAttrib.size    = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
eventAttrib.messageType   = NVTX_MESSAGE_TYPE_ASCII;  // message type & message
eventAttrib.message.ascii = __FUNCTION__ ":ascii";
eventAttrib.colorType = NVTX_COLOR_ARGB;              // color type and color
eventAttrib.color     = COLOR_YELLOW;
```

**Fortran** users use a wrapper module `[s48]` (available at `https://github.com/maxcuda/NVTX_example`) that: defines a derived type for the event, fills color and message attributes according to whether an optional ID is present, and resolves to the `*Ex` or `*A` API depending on the input.

### 1.13 Textual summaries: `nsys stats` `[s49]`

Beyond the GUI timeline, you can aggregate a report on the command line `[s49]`:

```bash
nsys stats report.nsys-rep
nsys stats -r openaccsum report_acc_data1.nsys-rep
```

`-r` selects the **report type**. Different kinds of metrics can be summarized `[s49]`: **`nvtxsum`** (NVTX ranges), **`cudaapisum`** (CUDA API calls), **`gpukernsum`** (GPU kernels), **`gpumemtimesum`** (GPU memory-transfer time), and `openaccsum` (OpenACC). The OpenACC summary table columns are `Time(%)`, `Total Time (ns)`, `Num Calls`, `Avg`, `Med`, `Min`, `Max`, `StdDev`, and `Name` `[s49]`. In the example, `Exit Data` (27.9%) and `Enter Data` (26.4% / 19.8%) dominate — again pointing at data movement rather than compute.

### 1.14 Worked example — CFD Jacobi solver `[s50–53]`

The example is a Fortran **CFD** code whose hotspot is `jacobistep_acc`, an OpenACC Jacobi stencil update `psinew(i,j) = 0.25 * (psi(i+1,j)+psi(i-1,j)+psi(i,j+1)+psi(i,j-1))`, plus a `deltasq` reduction for the error `[s50]`.

**Step 1 — annotate and filter with NVTX** `[s51].** The main phases are wrapped in NVTX ranges (`main_loop`, `write_out`), and profiling is restricted to just the loop:

```bash
nsys profile --trace=nvtx --capture-range=nvtx --nvtx-capture='main_loop' \
             --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0
```

`--capture-range=nvtx` with `--nvtx-capture='main_loop'` tells `nsys` to only record while inside the `main_loop` range, keeping the report small and focused.

**Step 2 — check for expensive data movements** `[s52]`. The timeline breakdown is damning: **only 1.9% of GPU time is Kernels** while **98.1% is Memory** — split roughly **50.3% HtoD memcpy / 49.7% DtoH memcpy** `[s52]`. Within the tiny kernel slice, work splits `50.4% jacobistep_acc / 49.6% cfd_144_gpu`. This is the textbook memory-bound symptom the earlier slides warned about.

**Step 3 — analyze how kernels map to GPU hardware** `[s53]`. Hovering a kernel exposes its launch configuration and occupancy. For `jacobi_deltasq_35_gpu` the tooltip reads:

| Property | Value |
|---|---|
| grid | `<<<8525,1,1>>>` |
| block | `<<<128,1,1>>>` |
| Launch Type | Regular |
| Static Shared Memory | 0 bytes |
| Dynamic Shared Memory | 2,048 bytes |
| Registers Per Thread | 38 |
| Local Memory Per Thread | 0 bytes |
| Local Memory Total | 146,276,352 bytes |
| Shared Memory executed | 102,400 bytes |
| Shared Memory Bank Size | 4 B |
| **Theoretical occupancy** | **75%** |
| Latency | 5.085 µs |
| Stream | 16 |

These are the per-kernel knobs that matter for tuning: the **grid/block** dimensions (how many threads), **registers per thread** (register pressure — too many registers limits how many warps fit), **shared memory** used, and the resulting **theoretical occupancy** (here 75%), i.e. how full the SM's warp slots are. Occupancy is the bridge to deck 2's discussion of hiding memory latency with enough active warps.

### 1.15 Asynchronous operations and streams `[s54–64]`

By default a GPU operation is **synchronous (SYNC): the CPU waits for the operation on the GPU to be over** before continuing `[s54]`. In **asynchronous (ASYNC)** mode, **the CPU does not wait** — it launches the GPU operation and immediately continues `[s54]`.

Why this matters is shown with three arrays `[s55]`:

```c
for (i=0; i<n; i++) a[i] = 1;         // populate A
for (i=0; i<n; i++) b[i] = 1;         // populate B
for (i=0; i<n; i++) c[i] = a[i]+b[i]; // C = A + B
```

Populating A and populating B are independent. **SYNC** runs them in order: `POPULATE A → POPULATE B → CALCULATE A+B` `[s56]`. **ASYNC** runs `POPULATE A` and `POPULATE B` concurrently and only then `CALCULATE A+B`, which depends on both `[s57]`.

Concurrency is expressed with **streams** (also called "queues") `[s58]`. A GPU can process multiple independent queues at once; each stream runs its operations in order, but different streams run in parallel. In OpenACC you place a kernel on a stream with the `async(n)` clause `[s59–63]`:

```c
#pragma acc parallel loop async(1)      // KERNEL 1 → stream 1
for (i=0; i<n; i++) a[i] = 1;

#pragma acc parallel loop async(2)      // KERNEL 2 → stream 2
for (i=0; i<n; i++) b[i] = 1;

#pragma acc wait(1) async(2)            // stream 2 waits for stream 1 to finish
#pragma acc loop async(2)               // KERNEL 3 → stream 2 (needs a and b)
for (i=0; i<n; i++) c[i] = a[i] + b[i];

#pragma acc update self[c[0:N]] async(2)  // copy c back to host on stream 2
#pragma acc wait                          // host waits for everything
```

The slides narrate this build: kernel 1 goes to queue 1 `[s59]`, kernel 2 to queue 2 `[s60]`; `wait(1) async(2)` makes *"this kernel wait for the operation in [stream] 1 to be completed and then be put in the queue of stream 2"* `[s61]`; kernel 3 (the `C = A + B` sum) then runs on queue 2 `[s62]`; and the final `update self` copies the result back — *"when stream 2 has completed the operation, the host is updated"* `[s63]`.

The payoff `[s64]`: without streams, two independent `H2D → kernel → D2H` sequences run **serialized** (one after the other). With streams they **overlap copying and computation** — while stream 1's kernel runs, stream 2 can already be doing its H2D copy, shortening total time. (The slide notes realistically: *"in real applications, your boxes will not be so evenly sized."*)

### 1.16 Worked example — Mandelbrot with streams `[s65–70]`

The Mandelbrot image is computed block-by-block in OpenACC Fortran; *each block is a kernel launched on the GPU* and *each unit of work executes an independent "mandelbrot"* `[s65]`.

- **Version 1 — copy the whole image** `[s65–66]`: `!$acc data copy(image(1:HEIGHT,1:WIDTH))`. The timeline shows **31.6% Kernels, 68.4% Memory** (≈50/50 HtoD/DtoH) — memory-dominated `[s66]`.
- **Version 2 — copy only what's needed with `update`** `[s67–68]`: switch to `!$acc data create(image(...))` and, per block, `!$acc update self(image(:,starty:endy))` so only the computed strip is copied back. Now **34.0% Kernels / 66.0% Memory**, taking *about 22 ms for 8 blocks* `[s68]`.
- **Version 3 — overlap with async streams** `[s69–70]`: put each block on an alternating stream and update asynchronously:

```fortran
!$acc parallel loop async(mod(block,2))
    ...
!$acc update self(image(:,starty:endy)) async(mod(block,2))
...
!$acc wait
```

Because the update for one block runs on a different stream from the computation of the next block, *the data update can overlap with computation of the next block* `[s69]`. Result: *about 14 ms for 8 blocks on 2 streams* `[s70]` — a ~1.6× speedup over version 2, purely from overlapping copies with compute (34.6% now attributed to stream 18).

### 1.17 Multi-Process Service (MPS) `[s71–72]`

**MPS** *enables cooperative multi-process CUDA applications, typically MPI jobs*, and *uses Hyper-Q to process CUDA kernels from different processes concurrently on the same GPU* `[s71]`. Without MPS, multiple MPI ranks sharing one GPU serialize their kernels; with MPS they can run together and fill the GPU.

A SLURM launch `[s71]`:

```bash
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=8
#SBATCH --gres=gpu:1

# This must be different for each node
export CUDA_MPS_LOG_DIRECTORY=./pipe0
export CUDA_MPS_PIPE_DIRECTORY=./pipe0

nvidia-cuda-mps-control -d                                   # start the MPS daemon
mpirun -np 8 --map-by node:PE=1 --rank-by core pw.x
echo quit | nvidia-cuda-mps-control                          # stop the daemon
```

On the timeline (a run labeled `Quartz nat = 9` across 8× A100-SXM-64GB) MPS *automatically enables multi-stream* and lets the ranks *overlap computation and data movements* `[s72]`.

### 1.18 Score-P and Vampir for heterogeneous programs `[s73–77]`

For MPI + OpenACC + CUDA codes, the deck introduces **Score-P** (instrumentation) + **Vampir** (visualization). The motivating case `[s73]` is a **PHonon** simulation from **QuantumESPRESSO** distributed by "images": MPI + OpenACC + CUDA, one MPI rank per GPU, one MPI task per image, with images doing *fairly independent computations*. Scaling efficiency is **0.88 on 2 GPUs but only 0.68 on 4 GPUs** — the slide asks *"why is efficiency low at 4 GPUs?"*, motivating a deeper profile.

**Score-P is source-code instrumentation** `[s74]`. You build through its wrapper and enable the GPU backends at runtime:

```bash
# build
SCOREP_WRAPPER=off cmake <cmake-options> <source-path>
make SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--mpp=mpi --openacc --cuda" <executable>

# measure at runtime (environment variables configure it)
export SCOREP_CUDA_ENABLE=yes
export SCOREP_OPENACC_ENABLE=yes
srun -n <task-number> ./myapp.exe    # generates a scorep_ folder with profiling data
```

**Score-P workflows** `[s75]` break down the **time spent in each OpenACC region** into: *launching kernels*, *waiting for GPU compute*, and *implicit and explicit data movement*. The goals are to **identify expensive kernel launches** and **identify expensive data movements**. From the **GPU's perspective** it reports *time spent in GPU kernels* vs *time spent IDLE* — in the example, `43.14659` units of **COMPUTE IDLE**, a large idle fraction that explains the poor 4-GPU efficiency.

**Vampir** visualizes the Score-P trace `[s76–77]`:

```bash
vampir scorep_12x1_trace/traces.otf2
```

The recommended reading order `[s76]`: **(1)** identify the overall pattern — *initialization, iteration, finalization*; **(2)** zoom into a single *iteration* to study it; then `[s77]` **(3)** visualize *communication patterns, durations, data movement (CPU–GPU), and CUDA kernels on the event timeline*. The function summary ranks MPI and CUDA calls (e.g. `MPI_Bcast`, `cudaMemcpyAsync`, `cudaDeviceSynchronize`, `cudaLaunchKernel`) by accumulated time.

---

## Section 2 — The GPU roofline model (`GPURooflineModelForHPC.pdf`)

This deck's subtitle sets the arc `[s1]`: *diagnosing execution bottlenecks, from naive CUDA implementations to Tensor Cores, and Nsight Compute profiling workflows.*

### 2.1 Why roofline `[s1–2]`

GPUs advertise enormous *theoretical peak throughput*, but real applications reach it only if they *expose massive parallelism* **and** *optimize data reuse* `[s2]`. The roofline model exists to answer, for a given kernel, **which of the two is holding you back**. The deck frames three ideas `[s2]`:

- **Roofline Modeling** — identify whether a bottleneck is caused by **memory throughput** or by **compute power**.
- **Hierarchical Modeling** — real architectures require optimizing across several memory levels: **HBM**, **L2 cache**, and **Shared Memory**.
- **The Occupancy Factor** — maximizing the number of **active warps per Streaming Multiprocessor (SM)** is critical to **hide memory latency** and keep the hardware fully utilized. (This is the same "theoretical occupancy" you saw in the kernel tooltip in deck 1.)

**The bound formulation** `[s2]` — the attainable performance `P` of a kernel is capped by:

$$P = \min(P_{\max},\; I \times B)$$

where **`P_max`** is the peak compute throughput (FLOP/s), **`B`** is the memory bandwidth (bytes/s), and **`I`** is the arithmetic intensity (FLOP/byte). You are limited either by raw compute (`P_max`) or by how fast memory can feed you (`I × B`), whichever is **smaller**.

### 2.2 Arithmetic (operational) intensity `[s3]`

To keep thousands of arithmetic pipelines busy, *data loaded from HBM must be heavily reused on-chip*; *without data locality, processing elements sit idle waiting for memory transactions* `[s3]`. **Arithmetic intensity** quantifies that reuse `[s3]`:

$$I = \frac{\text{Total Floating-Point Operations (FLOPs)}}{\text{Total Data Exchanged with Memory (Bytes)}} \quad [\text{FLOP/Byte}]$$

This single number splits all kernels into two domains `[s3]`:

- **Memory-Bound domain (low `I`)** — operations like *streaming vectors, updates, reductions, and transformations* have **zero temporal reuse**; each byte is used for only one or two FLOPs, so execution speed is **hard-capped by the maximum throughput of the memory interface**.
- **Compute-Bound domain (high `I`)** — algorithms like *GEMM (matrix multiply), high-degree stencils, and convolutions* exhibit **high temporal data reuse**; by recycling data in shared memory or registers they shift the bottleneck **from the memory bus to the arithmetic units**.

### 2.3 The GPU roofline chart and its roofs `[s4]`

A **roofline chart** plots **performance (TFLOP/s, log scale)** on the y-axis against **arithmetic intensity (FLOP/Byte, log scale)** on the x-axis `[s4]`. It exposes which hardware limit dominates a kernel. The chart has multiple **roofs** — ceilings a kernel cannot exceed:

- **HBM bandwidth roof** — the sloped line on the left; the global-memory traffic limit that caps **low-reuse (low `I`)** kernels. A memory roof is a diagonal because attainable performance = `I × B` grows with `I`.
- **L2 / shared-memory bandwidth roof** — a **higher** sloped roof that applies when data is reused from on-chip or cache-resident storage (on-chip bandwidth is higher than HBM, so the ceiling is higher).
- **CUDA-core roof** — the flat ceiling of standard **FP32/FP64** arithmetic throughput.
- **Tensor-core roof** — for supported precisions, the **Tensor Core** Matrix-Multiply-Accumulate (MMA) ceiling, the highest flat roof on modern GPUs.

The two flat roofs (CUDA-core, tensor-core) are the compute limits `P_max`; the two sloped roofs (HBM, L2/shared) are the memory limits `I × B`. Where the sloped memory roof meets a flat compute roof is the **ridge point** — the arithmetic intensity at which a kernel transitions from memory-bound (left of the ridge) to compute-bound (right of the ridge). Anything plotted **above** the roofs is the **forbidden region** — *performance above the hardware roofs is not physically attainable* `[s4]`.

### 2.4 Navigating the GPU memory hierarchy `[s5]`

*Performance engineering on GPUs revolves around moving data up to faster memory closer to the registers* `[s5]`. The three levels:

- **Global Memory (HBM)** — massive capacity but long latency; access *must be tightly coalesced into uniform transactions* to be efficient.
- **L2 Cache** — a **hardware-controlled** buffer between global memory and the processors; it *automatically saves frequently used data so the GPU doesn't keep re-fetching from HBM*.
- **Shared Memory** — a **programmer-controlled** high-speed workspace on each SM; it is *the fastest memory available* and is used to *manually cache data for complex tasks*.

Crucially, the slide connects optimizations to **movements on the roofline chart** `[s5]`:

- **Memory Coalescing** improves transaction efficiency, driving performance **vertically upward** toward the memory-bandwidth slope (you use the bandwidth you already have more efficiently — same `I`, higher achieved performance).
- **Shared-Memory Tiling** decreases the data payload fetched per operation, **increasing effective arithmetic intensity** and shifting the coordinate **horizontally to the right** (same FLOPs, fewer bytes → higher `I`).

Memorize this: **coalescing moves you up, tiling moves you right.**

### 2.5 Case study — the naive GEMM kernel `[s6]`

The running example is dense matrix multiply (GEMM). In the *naive* approach, **each independent CUDA thread computes a single output element of `C`** `[s6]`:

```cpp
__global__ void gemm_naive(int M, int N, int K,
                           const float* A, const float* B, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}
```

Why it is slow `[s6]`: every thread *loops through a full row and column, fetching elements sequentially from global memory*; **redundant global-memory requests escalate**, because neighboring threads (and neighboring blocks) re-fetch the same rows and columns. This *forces severe HBM congestion, pinning performance in the memory-bound domain* — low effective `I`, stuck on the HBM roof.

### 2.6 Algorithmic vs. effective intensity `[s7]`

This distinction is the key insight of the deck `[s7]`. For square `N×N` matrices, matmul performs **~2N³ floating-point operations**. In theory that implies *very high arithmetic intensity* — but real behavior depends on *how data is actually moved and reused*.

- **Algorithmic Intensity** — the *idealized* metric from the mathematical formulation, assuming **minimal data movement and perfect reuse**. It reflects the intrinsic properties of the algorithm (its best-case ceiling).
- **Effective Intensity** — the *observed* metric based on **actual memory traffic** on the GPU (HBM, caches, shared memory). **This is what profiling tools measure, and it is what determines your position on the roofline.**

The naive kernel has high *algorithmic* intensity but low *effective* intensity because it re-reads operands from HBM. Hence the **performance strategy** `[s7]`: **increase effective intensity toward the algorithmic limit.**

### 2.7 Shared-memory tiling `[s8–9]`

**Tiling** is how you raise effective intensity `[s8]`. It *partitions execution into cooperative thread neighborhoods*: the threads of a block **jointly load a sub-matrix tile into fast shared memory**, then repeatedly read operands **from shared memory instead of re-streaming them from high-latency HBM**. Reducing global-memory access *minimizes latency stalls*, letting the GPU spend clock cycles on arithmetic rather than waiting for data.

**Architectural impact** `[s8]`: swapping global-memory access for on-chip buffers means the **bytes moved across the main bus drop sharply while the FLOP count stays identical** — exactly what raises `I`. This *slides the kernel's coordinate to the right* on the roofline, freeing it to scale up toward the compute roof.

The tiled kernel `[s9]`:

```cpp
template<int TILE>
__global__ void gemm_tiled(int M, int N, int K,
                           const float* A, const float* B, float* C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // cooperatively stage one tile of A and one of B into shared memory
        if (row < M && (t * TILE + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < N && (t * TILE + threadIdx.y) < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();                              // wait until the tile is fully loaded
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();                              // wait before overwriting the tile
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
```

The two `__syncthreads()` barriers are essential: the first ensures the whole tile is loaded before any thread reads it; the second ensures every thread has finished using the tile before the next iteration overwrites it.

### 2.8 Mapping the optimization path on the roofline `[s10]`

Each refinement moves the kernel's coordinate on the chart `[s10]`:

- **Naive** — stuck at **low intensity, bound by memory bandwidth** (on the HBM roof, far left).
- **Coalesced** — moves **vertically upward** along the memory line by using memory more efficiently (better use of the same bandwidth).
- **Tiled (SRAM)** — cuts physical data transfers, pushing the coordinate **horizontally to the right into the compute-bound region**.
- **Tensor / cuBLAS** — transforms the operation to use hardware matrix pipelines, lifting performance **up to the high specialized ceilings** (the tensor-core roof).

This picture is the whole method in one image: profile → see where you land → apply the optimization that moves you toward the roof you're not yet touching.

### 2.9 Nsight Compute (`ncu`) `[s11]`

Where Nsight Systems traces the whole timeline, **Nsight Compute** zooms into a *single kernel*. It *uses hardware performance counters to collect data and automatically generate hierarchical roofline analyses* `[s11]`.

```bash
ncu --set full --import-source yes -o profile_output_%p \
    --section SpeedOfLight_HierarchicalTensorRooflineChart \
    ./matrix_compute.x
```

Flags taught on the slide `[s11]`:
- **`--set full`** — collect the full set of metrics (thorough but slower).
- **`--section SpeedOfLight_HierarchicalTensorRooflineChart`** — the section that produces the **hierarchical tensor roofline** (the multi-roof chart with HBM, L2/shared, CUDA-core, and tensor-core ceilings). "Speed of Light" is Nsight Compute's term for how close you are to the hardware limits.
- **`--import-source yes`** and compiling with **`--generate-line-info`** — correlate bottlenecks with the **specific lines of your CUDA source**.
- **`--kernel-name`** — focus the analysis on a specific kernel (a full-metric run on every kernel is expensive; filter to the one you care about).
- **Run profiling jobs on dedicated compute nodes** (never the login node).

> *Faithfulness note:* the deck names the `SpeedOfLight` roofline section and the `--set full` flag but does not enumerate individual `ncu` sub-metrics (e.g. per-metric SM-throughput %, DRAM-throughput %, or warp-stall reasons). The occupancy / registers-per-thread / shared-memory figures taught in this course come from the Nsight Systems kernel tooltip in deck 1 `[s53]` and the occupancy discussion in deck 2 `[s2]`.

### 2.10 Scaling performance with Tensor Cores `[s12]`

**Tensor Cores** are *specialized matrix-multiply-accumulate units* that compute, in one operation `[s12]`:

$$D = A \times B + C$$

- **Hardware model** — they accelerate small **matrix tiles (MMA operations)** at *much higher throughput than standard CUDA cores*.
- **Programming requirements** — efficient use needs *compatible data layouts* and *supported precision formats: FP16, BF16, or TF32*.
- **Software stack** — usually reached through **cuBLAS, CUTLASS, cuBLASLt, or cuDNN** rather than hand-written.
- **Performance effect** — they *move the kernel to a higher compute ceiling* on the roofline.

**Roofline interpretation** `[s12]`: `P_tensor >> P_cuda-core`. Tensor Core execution introduces a **higher compute roof**, so you must *always compare achieved performance against the correct precision ceiling* (e.g. the TF32/FP16 tensor peak, not the FP64 peak). Achieved performance depends on precision, problem size, and how well the computation maps to MMA operations.

### 2.11 cuBLAS and CuPy — the Python path `[s13]`

You rarely beat the vendor library. **cuBLAS** *combines assembly micro-kernels, fine-grained register blocking, and optimized warp scheduling to saturate the execution pipelines* `[s13]`. **CuPy** is the *dispatcher* that *exposes these optimized CUDA routines through Python/NumPy-style syntax*, routing array products straight into the hardware-tuned library entry points `[s13]`.

**HPC benchmark fact to remember** `[s13]`: on **Leonardo's NVIDIA A100, cuBLAS DGEMM reaches ≈ 19.5 TFLOP/s** — use this as a practical real-world peak reference.

Correct GPU timing in CuPy uses **CUDA events**, not wall-clock, and must **synchronize** because GPU calls are asynchronous `[s13]`:

```python
import cupy as cp

a_gpu = cp.asarray(matrix_a_cpu)
b_gpu = cp.asarray(matrix_b_cpu)
cp.cuda.Stream.null.synchronize()          # make sure prior work is done
start_event = cp.cuda.Event()
end_event   = cp.cuda.Event()
start_event.record()
for _ in range(trials):
    c_gpu = cp.matmul(a_gpu, b_gpu)        # dispatched to cuBLAS
end_event.record()
end_event.synchronize()                    # wait for the GPU to finish
total_time_sec = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0  # ms → s
runtime = total_time_sec / trials
```

### 2.12 Common roofline pitfalls `[s14]`

Four mistakes that produce misleading roofline plots `[s14]`:

- **Mismatched byte counting** — confusing *algorithmic* predictions with *actual* memory traffic. Caching behavior, memory-coalescing failures, and write-back traffic can all inflate the real bytes moved, so counting bytes from the formula gives the wrong `I`.
- **Evaluating trivial workloads** — testing tiny matrices. Small workloads *fail to saturate the GPU*, leaving kernels **latency-bound** and plotting **underneath all the roofline boundaries** (so the chart tells you nothing about the roofs).
- **Ignoring numerical precision** — plotting **single-precision** measurements against **FP64** ceilings (or vice versa). Always calibrate the roofs to the **active precision mode**.
- **Inaccurate profiling windows** — including **host-to-device memcpy (PCIe transfers)** or **library initialization** time inside the kernel's roofline metrics. **Isolate the kernel execution window** to measure pure kernel performance. (This echoes deck 1's "benchmark without I/O" discipline.)

### 2.13 GPU performance checklist and strategy `[s15]`

The consolidated optimization checklist `[s15]`:

1. Enforce strict **coalesced access** patterns across the memory buses.
2. Stage **high-reuse arrays in shared memory and registers** to raise arithmetic intensity.
3. **Monitor occupancy** — ensure enough active warps to hide pipeline latencies.
4. **Eliminate branch divergence** inside warps to keep the SIMT pipelines active.
5. Use **optimized vendor libraries** for standard computation patterns instead of rolling your own.
6. **Re-run the profiler after every change** to map your optimization trajectory.

**The decision rule** `[s15]` (and the heart of the whole module):

> - If a kernel is pinned to a **memory roof** → focus on **data locality and cache reuse** (coalescing, tiling).
> - If a kernel reaches a **compute roof** → look for a **higher-throughput execution path** (e.g. Tensor Cores / better precision / vendor library).

### 2.14 Hands-on lab and takeaways `[s16–17]`

The lab is the three optimization stages made concrete `[s16]`:

- **Phase 1 — Baseline:** implement the naive matmul, profile with Nsight Compute to establish the baseline roofline coordinate.
- **Phase 2 — SRAM Tiling:** add shared-memory tiling; measure how the **effective arithmetic intensity shifts to the right**.
- **Phase 3 — Production:** deploy a cuBLAS solution; analyze proximity to the **Tensor Core roof** and quantify the remaining performance gap.

For each phase you **measure and identify**: `I_effective`, achieved **GFLOP/s**, and the **primary bottleneck** `[s16]`.

The closing takeaways `[s17]`:
- **Diagnostic clarity** — the roofline model draws clear limits, distinguishing *fundamental hardware constraints* from *code inefficiencies*.
- **Intensity drives performance** — maximizing data locality (shared memory, registers) shifts workloads out of the memory-bound region into high-throughput regimes.
- **Informed engineering** — use **empirical profiler data** to target specific hardware bottlenecks rather than tuning blindly.

The one-line method for the whole module `[s17]`:

> **Profile → Identify Bottleneck → Target Optimization → Validate with Roofline** — repeat.

---

## Likely exam angles

**1. Is a kernel memory-bound or compute-bound, and how do you decide?**
Compute the **arithmetic intensity** `I = FLOPs / Bytes` `[deck2 s3]` and compare against the **ridge point** where the sloped memory roof meets the flat compute roof `[deck2 s4]`. If the kernel sits **left of the ridge / on a sloped roof**, it is **memory-bound** (`I × B < P_max`), so optimize data movement: coalescing to move **up**, tiling to move **right** `[deck2 s5, s10]`. If it sits **right of the ridge / near a flat roof**, it is **compute-bound**, so change the execution path (Tensor Cores, precision, vendor library) `[deck2 s12, s15]`. Governing equation: `P = min(P_max, I × B)` `[deck2 s2]`. Remember **algorithmic vs. effective** intensity — profilers measure effective, and the strategy is to raise effective toward algorithmic `[deck2 s7]`.

**2. What does occupancy mean, and why does it matter?**
Occupancy is the number of **active warps per SM** relative to the maximum the SM can host `[deck2 s2]`; the Nsight Systems kernel tooltip reports it as **theoretical occupancy** (e.g. 75%) `[deck1 s53]`. High occupancy lets the scheduler **hide memory latency** by switching to another ready warp while one warp waits on data. It is limited by per-thread resources — **registers per thread** (register pressure) and **shared memory per block** `[deck1 s53]`. More occupancy is not automatically faster, but too little occupancy leaves memory latency exposed and the SM stalling.

**3. When do streams (asynchronous execution) help?**
Streams help when you have **independent work** whose **data movement can overlap with computation** `[deck1 s54–64]`. Serialized `H2D → kernel → D2H` sequences become overlapped when placed on different streams, so one stream's copy runs while another's kernel computes `[deck1 s64]`. The Mandelbrot example makes it quantitative: **~22 ms on one stream vs ~14 ms on two streams** for 8 blocks `[deck1 s68, s70]`. Streams also let **MPS** run kernels from multiple MPI ranks concurrently on one GPU `[deck1 s71–72]`. They do **not** help if the work is inherently serial or if a single kernel already saturates the GPU.

**4. Why is data movement so often the bottleneck, and how do you see it?**
The **IO/PCIe bus is slow compared to GPU bandwidth**, so **moving data is expensive** and you must **improve data locality** `[deck1 s24–27]`. On a timeline you diagnose it by the **Kernels vs Memory split** — the CFD example was **1.9% Kernels / 98.1% Memory** (≈50/50 HtoD/DtoH), the classic memory-bound signature `[deck1 s52]`. Benchmark **without I/O** and **isolate the kernel window** — including PCIe copies or library init in kernel metrics is a listed roofline pitfall `[deck2 s14]`.

**5. Which tool for which question — nsys vs ncu (vs Score-P/Vampir)?**
**Nsight Systems (`nsys`)** = system-wide **timeline/trace**: kernel vs memcpy overlap, stream usage, launch latency, NVTX-annotated regions, CPU sampling `[deck1 s35–49]`. **Nsight Compute (`ncu`)** = deep dive on **one kernel** with hardware counters and the **hierarchical roofline** (`--set full`, `--section SpeedOfLight_HierarchicalTensorRooflineChart`, `--kernel-name`, `--generate-line-info`) `[deck2 s11]`. **Score-P + Vampir** = MPI + OpenACC + CUDA codes at scale, decomposing OpenACC-region time into launch / GPU-compute-wait / data-movement and exposing **COMPUTE IDLE** to explain poor multi-GPU efficiency `[deck1 s73–77]`.

**6. Kernel-launch overhead — what and why.**
Launching a kernel costs **latency** `[deck1 s31–32]`; many small back-to-back kernels make this overhead accumulate, so **"kernel launch adds overhead → expose as much parallelism as possible"** (fewer, larger kernels) `[deck1 s34]`.

**7. Reproduce the key commands.**
Timeline trace: `nsys profile --trace=osrt,cuda --sample=cpu --backtrace=fp` `[deck1 s42]` (debug symbols required for backtraces `[deck1 s39]`). NVTX-filtered capture: `nsys profile --trace=nvtx --capture-range=nvtx --nvtx-capture='main_loop' --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0` `[deck1 s51]`. Text summaries: `nsys stats -r {nvtxsum,cudaapisum,gpukernsum,gpumemtimesum} report.nsys-rep` `[deck1 s49]`. Kernel roofline: `ncu --set full --section SpeedOfLight_HierarchicalTensorRooflineChart ...` `[deck2 s11]`. Reference peak: cuBLAS DGEMM ≈ **19.5 TFLOP/s** on Leonardo A100 `[deck2 s13]`.
