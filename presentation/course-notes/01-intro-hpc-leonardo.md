# Module 1 — Introduction to HPC & the Leonardo supercomputer

*Source material: two CINECA lecture decks by Alberto Guarnieri (HPC Application Engineer) — `IntroToHPC.pdf` (41 slides) and `IntroToLeonardo.pdf` (24 slides). This chapter reconstructs both lectures for a student who never attended, following each deck's order. Slide references appear as `[s3]`, `[s5-9]`, etc.*

---

## Part A — Introduction to High Performance Computing

The first lecture builds up, in four movements, the answer to one question: *why is modern scientific computing forced to be parallel, and what does that force look like in hardware, in software, and in the numbers we use to judge it?* The deck's own roadmap `[s2]` is: (1) why we care about parallelism, (2) elements of HPC architecture, (3) an introduction to parallel programming, and (4) measuring performance.

### A.1 Why parallelism? Motivation and the physical wall

**What supercomputers are for `[s3]`.** Computational science uses supercomputers to study physical systems across every scale that is otherwise out of reach: the **very large** (meteo-climatology, cosmology, oil reservoirs), the **very small** (drug discovery, silicon chip design, molecular biology), the **very complex** (fundamental physics, fluid dynamics, solid mechanics), and things **too dangerous or too expensive** to do for real (fault simulation, nuclear tests, crash analysis). The slide anchors these with real projects — MISTRAL (meteorology), LIGATE (drug discovery), and SPACE (cosmology). The teaching point: HPC is a *scientific instrument*, like a telescope or an accelerator, for problems you cannot observe directly.

**A newer driver: AI & Big Data `[s4]`.** Beyond classical simulation, newer fields lean on massively parallel systems for three reasons: they need **optimized data transfers**, **great volumes of data storage**, and **processing power for structured dataflows**. The slide names the IT4LIA AI Factory and EUHubs4Data as European initiatives in this space. The message is that the same machines built for simulation are now central to data- and AI-driven science.

**Serial computing and why it wastes hardware `[s5]`.** A **serial algorithm** executes its instructions *sequentially* — one at a time, in order — on a **single thread**. On modern hardware this **underutilizes compute resources**: the chip is capable of much more than one instruction stream can keep busy. This is the baseline the rest of the lecture argues against.

**The need for parallelism — the end of "free" speed `[s6]`.** **Moore's law** observed that the **density of transistors on a chip doubles roughly every two years**, and this held for more than 40 years. But it is now bounded by physical limits:
- **Minimum transistor size** — transistors cannot be smaller than single atoms.
- **Quantum tunnelling** — at very small scales, quantum effects cause current leakage.
- **Heat dissipation and power consumption** — you cannot get the heat out fast enough.

The crucial distinction the slide draws: **more transistors does not mean higher CPU frequency.** Around **2006** a **power wall** was hit, marking **the end of Dennard scaling** (the historical trend where shrinking transistors let you raise clock frequency at constant power). The companion graph tracks five quantities from 1970 to 2020 — transistor count (still rising), single-thread performance (flattening), clock frequency in MHz (flat after ~2006), typical power in watts (flat), and number of logical cores (rising only after the frequency plateau). Read together: since we can no longer make one core faster, we make *many* cores — hence parallelism is a necessity, not a luxury.

**The other bottleneck: the processor–memory gap `[s7]`.** Regardless of how powerful a processor is, the real limitation in HPC is the **growing performance gap between processors and memory** — the cost of getting data *to and from* the CPU. Both hardware and software design must **minimize data-transfer time**. Two quantities define a data channel:
- **Bandwidth** — how much data can be transferred over a channel per unit time.
- **Latency** — the minimum time needed to *initiate* a data transfer.

Keep these two separate in your head: bandwidth is throughput once data is flowing; latency is the fixed start-up delay before it flows.

**Memory hierarchy `[s8]`.** Because CPUs are much faster than the devices that supply data, machines interpose **cache memory** — small but fast memory sitting between the processor and main memory. The slide gives an access-time pyramid (fastest/most-costly at the top):

| Level | Access time |
|---|---|
| Registers | 1 ns → 2 ns |
| L1 cache | 3 ns → 10 ns |
| L2 cache | 25 ns → 50 ns |
| Main memory (DRAM) | 30 ns → 90 ns |
| Hard drive | 5 ms → 20 ms |
| CD/DVD-ROM/RW | 100 ms → 5 s |
| Tape backup | 10 s → 3 min |

The mechanics: on a memory look-up the request propagates down through the cache levels until it reaches main memory. When data is **not** in cache, the machine loads it *together with adjacent data* from main memory, betting that the next access will fall in that neighborhood — a **cache hit** (found in cache) versus a **cache miss** (had to go to main memory). This is why data locality matters so much for performance.

**Cache hierarchy and "memory bound" `[s9]`.** Caches are classified by **level**, i.e. their closeness to the microprocessor:
- **L1** — extremely fast but small (e.g. 32 KB), usually embedded in the CPU.
- **L2** — bigger (e.g. 2 MB) but slower, possibly on a separate chip. Each core may have its own dedicated L1 and L2.
- **L3** — often shared amongst cores.

A program that is constantly stalled waiting for data from memory is called **memory bound** — its speed is limited by memory, not by arithmetic. (Contrast with compute-bound, limited by the processor itself.)

**Parallel computing, defined `[s10]`.** **Parallelization** means converting a serial algorithm into one that can perform multiple operations simultaneously. **Parallel computing** is the *simultaneous use of multiple compute resources* to solve a computational problem that has been broken into discrete parts solvable **concurrently**. This is the pivot from the "why" to the "how."

### A.2 Elements of HPC architecture

This section `[s11]` covers the data execution model, memory-access differences, node interconnection, and accelerators.

**Anatomy of a parallel computer `[s12]`.** A **parallel computer** is a collection of processors and memory banks joined by an **interconnection network** so processors can work collectively in concert. The major architectural design issues are: **processor coordination**, **memory organization**, **address space**, **memory access**, **granularity**, **scalability**, and the **interconnection network**. The slide's legend introduces the vocabulary used throughout: *memory*, *accelerator* (e.g. GPU), *processor*, *core*, *vector unit*, and the shared *parallel filesystem* — a single node is processors+cores+vector-units+memory with an attached accelerator, and many such nodes hang off a high-performance network and a parallel filesystem.

**Flynn's taxonomy `[s13]`.** The classic classification of parallel machines is by the **number of instruction streams** and **data streams**:
- **SISD** — Single Instruction, Single Data: traditional serial execution.
- **SIMD** — Single (global) Instruction, Multiple Data: special-purpose vector / data-parallel instructions, present in many modern CPUs. This is parallelism *within a single core*.
- **MISD** — Multiple Instruction, Single Data: not particularly useful, except when interpreted as a form of **pipelining**.
- **MIMD** — Multiple Instruction, Multiple Data: general-purpose parallel computers (multicore or multinode).

**Memory classification and NUMA `[s14]`.** How memory is accessed depends on hardware and programming model:
- **Shared memory** — memory is shared among processors *within a node*.
- **Distributed memory** — memory is partitioned among processors and reached through a common *network*.

Crucially, the time each processor needs to reach memory is **not uniform**. This is **NUMA (Non-Uniform Memory Access)**: a processor gets *faster* access to its **local** memory and *slower* access when it must cross the network to reach memory attached to another processor.

**NUMA nodes, concretely `[s15]`.** The diagram shows a two-socket node: **CPU Socket 1** and **CPU Socket 2**, each socket holding CPUs with private **L1/L2** caches, a shared **L3**, and its own **memory controller** driving local memory banks. The two sockets are joined by a **Quick Path Interconnect**, and an **IO controller** sits on top. Each socket + its local memory is one **NUMA node**; reaching the *other* socket's memory crosses the interconnect and costs more — the physical reason NUMA exists.

**Network topology `[s16]`.** Networks linking nodes come in several technologies chosen by price, performance, and vendor — **Ethernet, Gigabit, InfiniBand, Omnipath**. They are wired in particular **topologies**, either **direct** or **indirect**. The slide illustrates a **TORUS** (1-D, 2-D, 3-D), a **FAT TREE**, and hypercubes of increasing dimension. Terminology to remember: **if switches are used, the network is called a fabric.**

**Leonardo's network topology `[s17]`.** All Leonardo nodes are interconnected through an **NVIDIA Mellanox** network using **Dragonfly+**, capable of a maximum bandwidth of **200 Gbit/s** between each pair of nodes. Dragonfly+ is a relatively new topology for InfiniBand-based networks: it interconnects a very large number of nodes while keeping the count of switches and cables — and the network **diameter** — small. Compared with a **non-blocking fat tree**, cost is reduced and **scaling-out to more nodes becomes feasible**; compared with a **2:1 blocking fat tree**, it achieves close to **100% network throughput** for arbitrary traffic. Structurally, Leonardo's Dragonfly+ uses a **fat-tree intra-group** interconnection (two layers of switches) and an **all-to-all inter-group** interconnection.

**Parallel filesystem `[s18]`.** The **filesystem** governs how files are stored on disk and retrieved or written. In a parallel architecture with many simultaneous disk accesses, you need a **parallel filesystem technology** such as **GPFS, LUSTRE, or BeeGFS**. The diagram shows the Lustre-style split into management targets (MGT), metadata targets (MDT), and object storage targets (OSTs) with their servers, serving many thousands of clients.

**Graphics Processing Units `[s19]`.** GPUs, conceived as graphics cards, have been adopted as powerful HPC **accelerators**. Their defining trade-off: **low single-thread performance compared to CPUs, but thousands of threads per GPU.** A larger portion of the transistor budget goes to **data processing** rather than **data caching and flow control**, which makes them **perfect for linear-algebra operations**. Their programming model is similar to SIMD but GPU-specific: **SIMT — Single Instruction, Multiple Threads.** The node diagram quotes the relevant link bandwidths: **16× PCI Gen4.0 at 32 GB/s** (CPU↔GPU), **NVLink 3.0 at 200 GB/s** (GPU↔GPU), and **InfiniBand HDR100 at 12.5 GB/s** (to the NICs/network).

**CPU vs GPU, side by side `[s20]`.** The contrast diagram: a **CPU** has a few **cores**, each with a large **Control** unit, an **ALU**, and **L1 cache**, backed by shared **L2 and L3 cache** and **DRAM**. A **GPU** is built from **streaming multiprocessors**, each with a tiny control unit (Ctl), small L1, and *many* **ALUs**, backed by a shared **L2 cache** and **DRAM**. The two are joined over **PCI-Express**. Visually this is the whole GPU story: spend silicon on arithmetic units, not on control and caching.

### A.3 Introduction to parallel programming

This section `[s21]` covers different programming models, load balancing, and problem partitioning.

**Single Program, Multiple Data (SPMD) `[s22]`.** Typical numerical applications exploit **MIMD** hardware *not* by hand-writing different instruction streams, but by running **the same program on different portions of the data**. Because the instruction stream is often **data-dependent** (it contains branches based on the state of the data), the effective execution is still MIMD. This **Single Program, Multiple Data (SPMD)** model is the widely prevalent, principal way parallel algorithms are specified. The flow diagram makes the pattern concrete: a single *Program start* fans out into several parallel `Print` tasks, all of which meet at a **Barrier** (a synchronization point where everyone waits for everyone) before proceeding to *Program End*.

**Task (thread) parallelism `[s23]`.** **Task parallelism**, also called thread parallelism, partitions the *algorithm* across multiple threads. If an algorithm is a series of **independent operations**, those operations can be spread across processors — task 1 → cpu 1, task 2 → cpu 2, and so on — realizing parallelization by splitting the *work*.

**Data parallelism `[s24]`.** **Data parallelism** instead spreads the *data* across processors. The processors execute *merely the same operations*, but on **diverse data sets** — most often by distributing the elements of an array across the computing units. (Task parallelism divides the code; data parallelism divides the data.)

**Threads, processes, tasks, and multi-core `[s25]`.** Multiple **threads** can exist within the same **process** and share resources such as memory, whereas different **processes** do **not** share those resources. **Multi-core processors** can run more than one process or thread at the same time. A **task** is a set of program instructions loaded in memory; threads can split themselves into two or more simultaneously running tasks. The slide notes that in practice **tasks and processes are often used synonymously.**

**All the levels of parallelism in a cluster `[s26]`.** There are multiple, *stackable* ways to extract parallelism from an HPC cluster:
- **Instruction level** — e.g. **FMA**, a fused multiply-and-add executed as one operation.
- **SIMD / vector processing** — data parallelism within a core (the slide contrasts *scalar* processing, `c[i]=a[i]+b[i]` one element at a time, with *vector* processing, `c[1:n]=a[1:n]+b[1:n]` on whole arrays).
- **Hyperthreading** — multiple hardware threads per core (e.g. **4 hardware threads/core on Intel KNL, 8 on PowerPC**).
- **Cores per processor** — e.g. **18 for Intel Broadwell**.
- **Processors (sockets) per node**.
- **Processors + accelerators** — e.g. CPU + GPU.
- **Nodes in a system**.

The unifying lesson: **to reach the maximum (peak) performance of a parallel computer, all levels of parallelism must be exploited.** Leaving any level idle leaves performance on the table.

**Parallel programming models — shared vs distributed `[s27]`.** Shared- and distributed-memory organizations map to different **programming models**, and the performance of a multi-threaded program depends primarily on **load balance** and **data movement**. On a **shared-memory** system the hardware caching protocols *implicitly* orchestrate data movement between local caches and main memory (this is the world of **OpenMP**); these instructions coordinate their accesses via **locks** and **atomic operations**. By contrast, on **distributed-memory** systems data movement is done *explicitly* by **passing messages** (the data) between processors or nodes (this is the world of **MPI**). Many real simulations combine both in a **hybrid** approach: **OpenMP for intra-node** communication, **MPI for inter-node**.

**MPI, OpenMP, hybrid `[s28]`.**
- **Message Passing Interface (MPI)** — lets parallel processes communicate by sending "messages" (data). It is the most standard way to communicate **between nodes**, but can also be used within a node.
- **OpenMP** — lets parallel processes communicate via **shared memory within a node**. It **cannot** be used *between* shared-memory nodes.
- **Hybrid MPI + OpenMP** — combines the two, e.g. OpenMP inside a shared-memory node and MPI between nodes.

**Shared vs distributed — the trade-off table `[s29]`.** The two organizations have complementary strengths:

| Task | Distributed memory | Shared memory |
|---|---|---|
| Scalability | easier | harder |
| Data mapping | harder | easier |
| Data integrity | easier | harder |
| Performance optimization | easier | harder |
| Incremental parallelization | harder | easier |
| Automatic parallelization | harder | easier |

The warning attached: **programmers must be careful not to spend too much time communicating instead of computing** — most of the time applications show massive bottlenecks *because of communication*.

**Parallel programming on GPUs `[s30]`.** The main options for programming GPUs:
- **CUDA** — NVIDIA's extension to C/C++ for GPU programming.
- **HIP** — AMD's extension to C/C++, similar to CUDA.
- **OpenACC** — similar to OpenMP but used for GPU programming (directive-based).
- **OpenCL** — an open-standard alternative for programming GPUs.
- **SYCL** — a C++ library for programming any device, based on OpenCL.

The slide's SAXPY example shows the same `y[i] = a*x[i] + y[i]` loop written three ways: plain C, C-with-CUDA (a `__global__` kernel indexed by `blockIdx.x*blockDim.x + threadIdx.x`, launched as `saxpy<<<4096,256>>>(...)` with explicit `cudaMemcpy` host↔device transfers), and OpenACC (`#pragma acc kernels loop gang/vector` directives added to ordinary loops).

**Load balancing — the master-worker model `[s31]`.** Several HPC models distribute the workload among processors. A common one is the **master-worker model** (also **fork-join**): a single control process — the **master** — creates and manages all the **worker** processes. Depending on the language, communication can be **one-sided**, meaning the programmer explicitly manages only one process in a two-process operation. The diagram shows the master writing tasks into a shared *Space* (pool of tasks), workers collecting tasks, executing them, and returning results into the space, which the master then collects.

**Static vs dynamic scheduling `[s32]`.** How is the job actually distributed among workers?
- **Static scheduling** — the order in which threads execute is controlled in advance, **evenly dividing the work among all available threads**. Useful when the workload is **known a priori** before execution. (The illustration hands thread 0 iterations 0,4,8,12; thread 1 gets 1,5,9,13; etc. — a fixed, even split.)
- **Dynamic scheduling** — thread scheduling is done by the operating system (or the library in use) according to some scheduling algorithm, so the execution order depends on that algorithm unless you impose control. Useful when the exact workload of subtasks is **unknown** before execution (the illustration shows uneven chunk sizes handed out as threads become free).

**Domain decomposition `[s33]`.** **Domain decomposition** is the subdivision of the geometric domain into **subdomains**. It decomposes a problem into **fine-grain tasks**, maximizing the number of tasks that can run concurrently. It is usually combined with **divide-and-conquer** (recursively subdividing the problem into a tree-like hierarchy of subproblems), **array parallelism**, and **pipelining** (subdividing sequences of tasks performed on each chunk of data). The general goal is to **maximize the potential for concurrent execution while maintaining load balance**. Ideally you would have an **embarrassingly parallel** problem — many similar but *independent* tasks solved simultaneously — though this is generally **not achievable**. Finally, **mapping** assigns the coarse-grain tasks to processors, subject to **trade-offs between communication costs and concurrency**. (The workflow shown: *Partition → Communicate → Agglomerate → Map*.)

### A.4 Measuring performance

The final section `[s34]` covers strong scaling, weak scaling, and the limits.

**Scalability `[s35]`.** **Scalability** (or **scaling**) is the ability to handle more work as the size of the computer or application grows — the ability of hardware and software to deliver greater computational power when more resources are added. For **HPC clusters**, being scalable means system capacity can be increased *proportionally* by adding hardware. For **HPC software**, scalability is sometimes called **parallelization efficiency** — the **ratio between the actual speedup and the ideal speedup** obtained with a given number of processors. A **scalability test** measures an application's performance across varying problem sizes and processor counts; it does **not** test general functionality or correctness. Scalability tests split into two kinds: **strong scaling** and **weak scaling**.

**Strong scaling `[s36]`.** In **strong scaling**, the **number of processors is increased while the problem size stays constant** — so the **workload per processor decreases**. It is mostly used for **long-running, CPU-bound** applications, to find a configuration giving reasonable runtime at moderate resource cost; the per-processor workload must stay **high enough to keep all processors fully occupied**. The quantity plotted is the **speed-up**:

> **S(Nₚ) = tₛ / t(Nₚ)**

where `tₛ` is the serial time and `t(Nₚ)` the time on `Nₚ` processors. In practice the speedup achieved by adding processes **decreases continuously**, and the real-speedup curve peels away from — and eventually falls below — the ideal straight line (the example curve peaks around 128 processors, then drops).

**Strong scaling — efficiency `[s37]`.** The other key parameter is **efficiency**:

> **E_ss(Nₚ) = (tₛ · Nₛ) / (t(Nₚ) · Nₚ) = S(Nₚ) / Nₚ**

where `tₛ` is the serial task time, `t(Nₚ)` the time to complete the same unit of work with `Nₚ` processing elements, and `Nₛ` the number of processors needed to complete the serial task (typically 1, but sometimes a chosen basic number of cores). **Ideally** we want a **linear speedup equal to the number of processors** — every processor contributing 100% of its power — but this is a very challenging goal for real-world applications, and parallel efficiency falls off as cores increase.

**Amdahl's law `[s38]`.** Execution time does **not** decrease proportionally as you add cores. Compute time cannot be reduced indefinitely because the problem cannot be subdivided forever, and real applications suffer **nonzero latency and nonzero bandwidth** (they do not communicate infinitely fast). This is captured by **Amdahl's Law**:

> **S(Nₚ) = 1 / [ (1 − P) + P/Nₚ ]**

where **P** is the fraction of the code that can be **parallelized**, **(1 − P)** is the **serial** fraction, and **P/Nₚ** is the parallel part spread over `Nₚ` processors. The decisive consequence: **for a fixed problem, the upper limit of speedup is determined by the serial fraction of the code.** Even with infinite processors, the serial part `(1 − P)` caps your speedup — which is exactly why the plotted speedup-versus-parallel-fraction curve shoots up only as the parallelizable fraction `F` approaches 1.

**Weak scaling `[s39]`.** In **weak scaling**, **both the number of processors and the problem size are increased**, keeping the **workload per processor constant**. The metric is **weak scaling efficiency**:

> **E_ws = tₛ / t(Nₚ)**

Weak scaling is mostly used for **large, memory-bound applications** whose required memory cannot be satisfied by a single node. Such applications usually **scale well to higher core counts**, because memory-access strategies often focus on the nearest neighboring nodes while ignoring those further away (so they scale well by construction); the upscaling is then limited only by available resources or the maximum problem size. For an application that scales *perfectly* weakly, the work done per node stays the same as the machine grows — you solve progressively **larger** problems in the *same* time it takes to solve smaller ones on a smaller machine.

**Scalability limits `[s40]`.** What ultimately caps scaling:
- **Hardware** — memory-CPU bandwidths and network communications.
- **Algorithm** — including the quality of the **domain decomposition**.
- **Parallel overhead** — the time required to *coordinate* parallel tasks, which includes **task start-up time, synchronizations, data communications, software overhead** (imposed by parallel compilers, libraries, tools, the operating system), and **task termination time**.

The barrier diagram makes the overhead visible: five tasks reach a **barrier** at *different* times, and everyone is stalled until the slowest arrives before any task passes the continuation point — synchronization turns load imbalance directly into wasted time.

**Recap of the HPC lecture `[s41]`.** Four takeaways: (1) **consider all the parallelization levels supported by the hardware**; (2) **intra-node → shared-memory access, inter-node → distributed-memory access**; (3) **HPC applications are typically memory-bound — check with a weak-scaling test**; (4) **GPUs are meant for data-parallel workloads, using the SIMT programming model.**

---

## Part B — Introduction to the Leonardo supercomputer

The second lecture is the practical companion: what Leonardo *is*, how to get on it, how to set up your environment with **modules**, and how to run work through the **SLURM** scheduler `[s2]`. Everything here operationalizes Part A on one specific machine.

### B.1 What Leonardo is

**Global ranking — the TOP500 `[s3]`.** The **TOP500** is the reference ranking of HPC systems (by LINPACK performance). Leonardo debuted **4th in November 2022** and stood **10th in November 2025**. The November-2025 list, top to bottom, is **El Capitan (#1), Frontier, Aurora, JUPITER Booster, Eagle, HPC6, Fugaku, Alps, LUMI, and Leonardo (#10)**. Two rows anchor the scale:

| | Cores | Rmax (PFlop/s) | Rpeak (PFlop/s) | Power (kW) |
|---|---|---|---|---|
| **#1 El Capitan** (AMD EPYC + Instinct MI300A, US) | 11,340,000 | 1,809.00 | 2,821.10 | 29,685 |
| **#10 Leonardo** (Xeon Platinum 8358 + NVIDIA A100 64 GB, quad-rail HDR100 IB, EViDEN, EuroHPC/CINECA, Italy) | 1,824,768 | 241.20 | 306.31 | 7,494 |

Note that **HPC6 (#6, Eni, Italy)** means Italy holds two systems in the top ten. Two vocabulary items used later: **Rmax** = maximal LINPACK performance actually achieved; **Rpeak** = theoretical peak.

**Leonardo infrastructure — the big picture `[s4]`.** Leonardo is built from two compute modules plus front-end and storage:
- **Booster Module** — **3456 nodes, 240 PFLOPS HPL** (GPU-accelerated).
- **Data-Centric Module (DCGP)** — **1536 nodes, 9 PFLOPS HPL** (CPU-only).
- **Low-latency interconnect at 200 Gb/s** and an **Ethernet interconnect at 100 Gb/s**, with an **InfiniBand/Ethernet gateway** rated **2.5 TB/s**.
- **Storage Fast Tier: 5.4 PB at 1.4 TB/s**; **Storage Capacity Tier: 106 PB at 620 GB/s**.
- A **front-end & service partition** (login nodes, visualization nodes, service & management) that connects out to facility routing and the internet/GÉANT.

**Login nodes `[s5]`.** The front-end you actually land on:
- **2× Intel Xeon Platinum 8358** processors (Intel **Ice Lake**, 32 cores, 3.4 GHz with Turbo).
- **Hyper-threading (×2) enabled**.
- **RAM: 512 GiB DDR4 3200 MHz**.
- **14 TiB disk in RAID1**.
- **No GPUs**, and they are **open to the outside network**.
- A **serial partition** runs on two login nodes; a **datamover service** runs on three login nodes.

The absence of GPUs and openness to the internet are exactly why login nodes are for editing, compiling, and transfers — not for running real jobs.

**Compute nodes — the Booster (GPU) module `[s6]`.** Built from the **Atos BullSequana X2135 "Da Vinci" blade**:
- **3456 nodes**, named **`lrdn[0001-3456]`**.
- **1× Intel Xeon Platinum 8358** — 32 cores.
- **RAM: 512 (8 × 64) GB DDR4 3200 MHz**.
- **Accelerators: 4× NVIDIA A100 64 GB "custom"** — a **15% performance improvement over the standard A100**.
- **Internal network: NVIDIA Mellanox HDR DragonFly+ at 200 Gb/s**.
- **Diskless.**

**Booster module — GPU detail `[s7]`.** Per-GPU peak performance is **11.2 TFlops FP64** and **22.4 TFlops FP32**. Intra-node connections use **NVLink, PCIe, and GPU-direct**, giving **200 GB/s between the GPU pairs**; **each GPU has a direct 100 Gb/s connection to the InfiniBand network**; PCIe is **Gen4 at 31.5 GB/s**; and the **GPU memory bandwidth is 6.5 TB/s** (this is the CX200 board carrying 4× A100 and 4× HDR100 links to IB).

**Compute nodes — the DCGP (CPU-only) module `[s8]`.** Built from the **BullSequana X2140 three-node CPU blade**:
- **1536 nodes**, named **`lrdn[3457-4992]`**.
- **2× Intel Xeon Platinum 8480+** — **56 cores per CPU**.
- **RAM: 512 GB DDR5 4800 MHz**.
- **InfiniBand: 1× NVIDIA HDR card, 100 Gb/s, via PCIe Gen 5**.
- **Disk: 1× M.2 SSD 3.84 TB**.
- Performance: **Rmax per node ≈ 8.5 TFLOPS**, **Rmax ≈ 13 PFLOPS** overall.

**Network topology, on Leonardo `[s9]`.** The network is **shared between the Booster and DCGP modules**. It is a **Dragonfly+ topology based on NVIDIA Mellanox InfiniBand HDR**, with **bidirectional bandwidth of 200 Gb/s**. All nodes are divided into **cells**; within a cell there is a **non-blocking two-layer Fat Tree**, and there is an **all-to-all connection between cells**. Two operational features: **the SLURM scheduler is aware of the network topology and tunes node allocations accordingly**, and an **adaptive routing algorithm alleviates network congestion**. Cells come in flavors: **Booster cells, DCGP cells, a Hybrid cell (Booster + DCGP), and a Service cell**.

**Storage tiers `[s10]`.** Storage is **shared between the Booster and DCGP modules**:
- **Fast Tier — 5.4 PB at 1.4 TB/s**, on **SSD (NVMe)** disks, holding **home + public + fast scratch**.
- **Capacity Tier — 106 PB at read 744 GB/s / write 620 GB/s**, on **HDD** disks, holding **work + large scratch + DRES**.

Both tiers are **connected via the InfiniBand network** and **managed by the Lustre parallel filesystem**.

**Filesystem areas `[s11]`.** The user-facing storage areas, each with its own policy:

| Area | Quota | Scope | Persistence | Backup |
|---|---|---|---|---|
| **$HOME** | 50 GB per user | user-specific | permanent | daily backup (soon) |
| **$PUBLIC** | 50 GB per user | user-specific (permissions **755**) | permanent | **no** backup |
| **$SCRATCH** | no quota | user-specific | temporary — **data removed after 40 days** | **no** backup |
| **$WORK** | per account (**default 1 TB**) | account-specific | permanent | **no** backup |
| **$FAST** | like $WORK | — | — | **fast I/O** |
| **$TMPDIR** | — | **local on nodes**, job-specific | — | — |

All filesystems are **based on Lustre**. Check your areas, disk usage, and quota with:

```bash
$ cindata
$ cinQuota
```

Reference: `https://docs.hpc.cineca.it/hpc/hpc_data_storage.html`.

### B.2 Accessing the cluster

**Access to Leonardo `[s12]`.** You should have received an **email with a username and password**; **no 2FA** is required for these accounts. **Be careful — accounts are easily banned if you enter the wrong password multiple times.** Access is over **SSH**:

```bash
$ ssh <username>@login.leonardo.cineca.it
```

**Large data transfers** should go through the **Leonardo Datamover** host instead of the login node:

```bash
$ scp ./file <username>@data.leonardo.cineca.it:/destination/path/
```

On login you are greeted by a banner summarizing the cluster (RHEL 8.7, the Booster and DCGP specs, the InfiniBand Dragonfly+ network, SLURM as workload manager, and support at `superc@cineca.it`). Reference: `https://docs.hpc.cineca.it/general/access.html#access-via-secure-shell-ssh`.

### B.3 Environment setup with modules

**The module environment `[s13]`.** Software is exposed through **environment modules**, grouped into **profiles**. Loading a profile **adds** its modules to what you can see (profiles are additive):

```bash
$ module load profile/astro
$ module av                      # list available modules
```

The available profiles include `profile/archive`, `profile/base`, `profile/astro`, `profile/candidate`, `profile/chem-phys`, `profile/geo-inquire`, `profile/deeplrn`, `profile/lifesc`, `profile/meteo`, `profile/quantum`, `profile/spoke7`, `profile/statistics`. Two inspection commands:

```bash
$ module show <module_name>/<version>   # dependencies, paths
$ module help <module_name>/<version>   # help text, brief description, usage examples
```

**More module commands `[s14]`.**

```bash
$ modmap -m <module_name>               # detect all profiles, categories, modules (e.g. different releases)
$ module load <profile>
$ module load <module_name>/<version>   # all dependencies loaded automatically
$ module list                           # list profiles and modules loaded so far
```

**Critical architecture rule:** you will find modules compiled to support **GPUs** and modules suitable only for **CPUs**. You can tell from the compiler embedded in the module's full name (e.g. `gromacs/2022.3--intel-oneapi-mpi--2021.10.0--oneapi--2023.2.0`). Modules compiled with **gcc, nvhpc, or cuda should be used only on the Booster partition**, while modules compiled with **intel oneapi are suitable for the DCGP partition**. Reference: `https://docs.hpc.cineca.it/hpc/hpc_enviroment.html`.

**Installing your own software `[s15]`.** If a package is not available, you can install it yourself:
- **Install without sudo permissions.**
- **Install with conda/pip inside a conda/virtual env** — note that the *official* conda repository is **no longer reachable** from CINECA clusters, so install **Miniconda** and rely on the **conda-forge** repository (`https://docs.hpc.cineca.it/services/miniconda.html`).
- **Install with Spack.**

Write to **`superc@cineca.it`** for help with an installation or to request a new module.

**Programming environment `[s16]`.** Compilers and MPI libraries are available as modules in **`profile/base`** — and you must **use the ones suitable for the architecture**. Compilers:
- **GCC** — GNU compilers: `gcc`, `g++`, `gfortran`.
- **NVHPC** — formerly hpc-sdk, formerly PGI + CUDA → NVIDIA compilers `nvc`, `nvc++`, `nvcc`, `nvfortran`.
- **CUDA**.
- **INTEL ONEAPI** — Intel compilers `icc`, `icpc`, `ifort`; oneAPI compilers `icx`, `icpx`, `ifx` → **no NVIDIA GPU support**.

MPI libraries:
- **OpenMPI** — for GNU/NVHPC compilers.
- **Intel oneAPI MPI** — for Intel compilers → **not CUDA-aware**.

Discover what's available with `modmap -m`, `module av`, `module show`, `module help`, and `man`.

### B.4 Running jobs with SLURM

**The golden rule `[s17]`.** CINECA HPC clusters are **shared among many users**, so **responsible use is crucial**.
- **Login nodes:** interactive runs are **strongly discouraged** and should be limited to short test runs → a **10-minute CPU-time limit**; avoid running large or parallel applications; **no GPUs** on login nodes.
- **Compute nodes:** long production jobs must be submitted to compute nodes through the **SLURM scheduler**. Jobs are submitted in two main ways — **batch mode** and **interactive mode**. Nodes are **shared**, but the resources allocated to you (**cores, GPUs, RAM, $TMPDIR**) are assigned in an **exclusive** way.

**Project accounts `[s18]`.** You have a limited number of **core-hours** to spend. These are **not** assigned to *user* accounts but to **project accounts**, shared among everyone on the same project (your research partners). **Mind the different project-account names** (e.g. a trailing `_0`): accounts defined on **Booster** can only be used on the **Booster partition** (`boost_usr_prod`), and accounts defined on **DCGP** only on the **DCGP partition** (`dcgp_usr_prod`). For this course:
- **DCGP module: `tra26_polimi`**
- **Booster module: `tra26_polimi_0`**

Check the projects linked to your account with:

```bash
$ saldo -b            # Booster accounts
$ saldo -b --dcgp     # DCGP accounts
```

**Batch mode `[s19]`.** Write a **batch script**, request resources with **`#SBATCH`** directives, then submit with `sbatch`; the job is queued and scheduled:

```bash
$ sbatch jobscript.sh
```

A representative script:

```bash
#!/bin/bash

#SBATCH --nodes=1                     # nodes
#SBATCH --ntasks-per-node=4           # tasks per node
#SBATCH --cpus-per-task=8             # cores per task
#SBATCH --gres=gpu:4                  # GPUs per node
#SBATCH --mem=494000                  # mem per node (MB)
#SBATCH --time=00:30:00               # time limit (d-hh:mm:ss)
#SBATCH --account=tra26_polimi_0      # project account
#SBATCH --partition=boost_usr_prod    # partition name
#SBATCH --qos=boost_qos_dbg           # quality of service
#SBATCH --exclusive                   # node in exclusive way
#SBATCH --out=%j.out                  # output file
#SBATCH --err=%j.err                  # error file

# Export environment variables
# Load modules, activate virtual/conda environment
# Launch your application with srun / mpirun / python ...
```

Gotcha called out on the slide: **`boost_qos_dbg` is fast to allocate but cannot be requested together with `--exclusive`.**

**Interactive mode `[s20]`.** Request resources with the same SLURM directives, but via **`srun`** or **`salloc`**; the job is queued and scheduled, and you launch your application from the resulting prompt with stdin/stdout/stderr connected to your terminal. Two cases:

*Non-MPI programs* (one process using one or more GPUs) — use `srun ... --pty /bin/bash`; the session starts **on the compute node** (`[username@lrdn0053 ~]$`):

```bash
$ srun -N 1 --ntasks-per-node=8 --cpus-per-task=4 \
       --gres=gpu:4 -t 00:30:00 -p boost_usr_prod \
       -q boost_qos_dbg -A tra26_polimi_0 --pty /bin/bash
```

*MPI programs* (using one or more GPUs) — use `salloc`; a new session starts **on the login node** (`[username@login14 ~]$`), from which you launch across the allocation:

```bash
$ salloc -N 1 --ntasks-per-node=8 --cpus-per-task=4 \
         --gres=gpu:4 -t 00:30:00 -p boost_usr_prod \
         -q boost_qos_dbg -A tra26_polimi_0
```

Exit the interactive session with `exit` (or **Ctrl+D**).

**Partitions and QoS `[s21]`.** The **`--partition`** (`-p`) flag names the **partition** — the specific set of nodes among which your job searches for resources:

```bash
#SBATCH --partition=boost_usr_prod    # or  -p boost_usr_prod
```

The **`--qos`** (`-q`) flag sets the **Quality of Service**, used to modify a partition's limits and priority, or to access selected partitions; if unspecified, the default QoS **`normal`** applies:

```bash
#SBATCH --qos=boost_qos_dbg           # or  -q boost_qos_dbg
```

The limits table for the debug QoS:

| Partition | QoS | TRES limits per job | Walltime | MaxTRES per user | Priority |
|---|---|---|---|---|---|
| `boost_usr_prod` | `boost_qos_dbg` | Max = 8 nodes | 00:30:00 | 8 nodes, 256 cores, 32 GPUs | 80 |

Reference: `https://docs.hpc.cineca.it/hpc/leonardo.html#job-managing-and-slurm-partitions`.

**Reservations `[s22]`.** A **reservation** gives you **priority over a pool of resources**. It is associated with a project account and a partition, both of which must be specified:

```bash
#SBATCH --reservation=s_tra_poli1 --account=tra26_poli --partition=dcgp_usr_prod
```

This course has **6 reservations**, `s_tra_poli1` through `s_tra_poli6`, one per session:
- **June 8th:** `--reservation=s_tra_poli1 -A tra26_polimi -p dcgp_usr_prod`
- **June 9th morning:** `--reservation=s_tra_poli2 -A tra26_polimi -p dcgp_usr_prod`
- **June 9th afternoon:** `--reservation=s_tra_poli3 -A tra26_polimi_0 -p boost_usr_prod`
- **June 10th:** `--reservation=s_tra_poli4 -A tra26_polimi_0 -p boost_usr_prod`
- **June 11th:** `--reservation=s_tra_poli5 -A tra26_polimi_0 -p boost_usr_prod`
- **June 12th:** `--reservation=s_tra_poli6 -A tra26_polimi_0 -p boost_usr_prod`

(Note the pattern: DCGP sessions use `tra26_polimi` + `dcgp_usr_prod`; Booster sessions use `tra26_polimi_0` + `boost_usr_prod`.)

**Monitoring `[s23]`.** The commands for watching and managing jobs:

```bash
$ squeue -u <username>          # or: squeue --me
```
Shows all your scheduled jobs with their status (pending, running, closing, …) and the **jobID** needed by other SLURM commands.

```bash
$ scontrol show job <job_id>
```
Prints a long list of information about the job; if it is not running yet, it tells you **why**, and if it is scheduled with top priority, gives an **estimated start time**.

```bash
$ scancel <job_id>
```
Removes the job (queued or running) by killing it.

```bash
$ sinfo                          # e.g.  sinfo -o "%10D %a %20F %P"
```
Provides information about SLURM nodes and partitions.

```bash
$ sacct <options> <job_id>       # e.g.  sacct -Bj <job_id>
```
Displays accounting data for all jobs and job steps in the SLURM accounting log or database.

**Recap of the Leonardo lecture `[s24]`.** Four operating principles: (1) **use login nodes only for internet access, installations, and small tests**; (2) **Leonardo Booster is the GPU partition, Leonardo DCGP is the CPU-only partition**; (3) **rely on the already-available software stack, optimized for the cluster architecture**; (4) **submit jobs with SLURM and be considerate of other users on the same project account.**

---

## Likely exam angles

Definitions, comparisons, formulas, and concrete numbers the two decks emphasize most:

- **Moore's law vs Dennard scaling** — Moore = transistor density doubles ~every 2 years; Dennard scaling (frequency scaling at constant power) *ended* around 2006 at the "power wall." Be ready to explain why more transistors ≠ more frequency, and name the three limits: minimum transistor size, quantum tunnelling, heat/power `[s6]`.
- **Bandwidth vs latency** — throughput per unit time vs the minimum time to *initiate* a transfer `[s7]`.
- **Cache hit vs cache miss; memory-bound** — and the cache-level hierarchy (L1 ≈ 32 KB embedded; L2 ≈ 2 MB, per-core; L3 shared) `[s8-9]`.
- **Flynn's taxonomy** — define SISD / SIMD / MISD / MIMD and give the typical use of each; know that SIMD is intra-core and MIMD is general-purpose multicore/multinode `[s13]`.
- **Shared vs distributed memory; NUMA** — definitions plus the trade-off table (distributed easier for scalability/data-integrity/optimization; shared easier for data-mapping/incremental/automatic parallelization) `[s14, s29]`.
- **SIMD vs SIMT** — CPU vector parallelism vs the GPU's "Single Instruction, Multiple Threads" model; and *why* GPUs suit linear algebra (more transistors for compute than for cache/control) `[s19-20]`.
- **Task parallelism vs data parallelism** — divide the code vs divide the data `[s23-24]`.
- **MPI vs OpenMP vs hybrid** — message passing (inter-node) vs shared memory (intra-node) vs both; know locks/atomics belong to shared memory `[s27-28]`.
- **The parallelism levels** — instruction (FMA), SIMD/vector, hyperthreading, cores/socket, sockets/node, +accelerators, nodes; "exploit all levels to reach peak" `[s26]`.
- **Static vs dynamic scheduling** — workload known a priori vs unknown `[s32]`.
- **Domain decomposition & embarrassingly parallel** — Partition→Communicate→Agglomerate→Map; embarrassingly parallel = many independent tasks (usually unattainable) `[s33]`.
- **Strong vs weak scaling** — strong: fixed problem, more processors (CPU-bound); weak: problem *and* processors grow together, constant work per processor (memory-bound). Know **which** is recommended to test HPC apps (weak scaling) `[s36, s39, s41]`.
- **Formulas to reproduce** — speed-up `S(Nₚ)=tₛ/t(Nₚ)`; strong-scaling efficiency `E_ss=S(Nₚ)/Nₚ`; weak-scaling efficiency `E_ws=tₛ/t(Nₚ)`; **Amdahl's law** `S(Nₚ)=1/[(1−P)+P/Nₚ]`, and its punchline that the serial fraction caps speedup for a fixed problem `[s36-39]`.
- **Parallel overhead / scalability limits** — hardware bandwidth, algorithm/decomposition, and coordination overhead (start-up, synchronization, communication, software, termination); the barrier picture of load imbalance `[s40]`.
- **Rmax vs Rpeak; TOP500** — achieved vs theoretical peak LINPACK; Leonardo #10 (Nov 2025), Italy also has HPC6 at #6 `[s3, s8]`.
- **Booster vs DCGP** — Booster: 3456 nodes, 4× A100 64 GB, DDR4, diskless, `boost_usr_prod`; DCGP: 1536 nodes, 2× Xeon 8480+ (56 c/CPU), DDR5, CPU-only, `dcgp_usr_prod`. Node names `lrdn[0001-3456]` / `lrdn[3457-4992]`. Know which compilers map to which partition (gcc/nvhpc/cuda → Booster; intel oneapi → DCGP) `[s6, s8, s14]`.
- **Storage areas** — $HOME (50 GB, backup soon), $PUBLIC (50 GB, 755, no backup), $SCRATCH (no quota, purged after 40 days), $WORK (1 TB/account), $FAST (fast I/O), $TMPDIR (node-local, job-specific); all on **Lustre** `[s10-11]`.
- **Login-node rules** — no GPUs, 10-minute CPU-time limit, edit/compile/transfer only; run via SLURM on compute nodes `[s5, s17]`.
- **SLURM essentials** — batch (`sbatch`) vs interactive (`srun --pty` starts *on the compute node*; `salloc` starts *on the login node*); key `#SBATCH` flags (`--nodes`, `--ntasks-per-node`, `--cpus-per-task`, `--gres=gpu:N`, `--mem`, `--time`, `--account`, `--partition`, `--qos`, `--exclusive`); the `boost_qos_dbg` limits (≤ 8 nodes, 30 min) and its incompatibility with `--exclusive`; monitoring with `squeue`, `scontrol show job`, `scancel`, `sinfo`, `sacct`; `saldo -b` / `saldo -b --dcgp` for budget; project accounts are shared and partition-specific (`tra26_polimi` vs `tra26_polimi_0`) `[s18-23]`.
