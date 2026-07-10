# Module 6 — Profiling I: methodology, CPU tools, MPI profiling

*Source material: two CINECA lecture decks by Laura Bellentani — `introduction.pdf`
("Profiling techniques and tools", 69 slides) and `mpi.pdf` ("MPI and OpenMP: common
bottlenecks and how to spot them", 57 slides). Slide references `[s.N]` are the PDF page
number inside the relevant deck. Everything below is taken only from the slides.*

This module answers two questions. **Deck 1** teaches *how measurement works* — what
profiling is, what it costs, which techniques exist, and the CPU-side tools (timers,
gprof, PAPI, roofline, PMPI). **Deck 2** teaches *what goes wrong in parallel programs*
— the MPI/OpenMP bottleneck taxonomy, the POP efficiency model that quantifies each
bottleneck, and the tool workflows (Scalasca/Score-P, ITAC, VTune) that expose them. The
overarching lesson the course repeats: profiling tells you *where* to look; tracing tells
you *why* it is slow; and at scale the answer is almost always communication and
dependencies, not computation.

---

## Section 1 — Profiling techniques and tools (`introduction.pdf`)

### 1.1 Why profile? [s2-10]

The motivation is framed with a slow program: a run prints `After 100 iterations, error
is 0.7069E-02; Time for 100 iterations was 15.26 seconds; Each individual iteration took
0.1526 seconds` [s2]. It is slow (the snail), and we want it fast.

The deck then shows *why speed is not free* with a strong-scaling table [s4-7]. The same
job is run on more MPI ranks, and **efficiency falls as ranks rise**:

| procs | 2 ranks | 4 ranks | 16 ranks | 32 ranks |
|-------|---------|---------|----------|----------|
| time (s) | 7.8 | 4.76 | 1.58 | 1.00 |
| efficiency | 97% | 80% | 60% | 47% |

Adding processors keeps reducing time, but each added processor buys less and less: at 32
ranks you are wasting more than half your cores. Profiling is how you find out *why*. The
deck flashes the tool landscape you will meet — Vampir, TAU, NVIDIA Nsight, Score-P,
Intel VTune [s9-10] — before setting the agenda.

**Two goals for the course** [s12]: (1) understand the idea of *measuring* an
application, and (2) identify a *suitable measurement tool* for your program. Two external
training resources are named: POP online training (`pop-coe.eu`) and Vi-HPS training
(`vi-hps.org`).

### 1.2 The glossary: profiling vs tracing [s13-16]

- **Profiling** = "the act of measuring the properties of an application by
  **summarizing** a set of events **over the execution interval**" [s13]. You get
  aggregated numbers (totals, averages), not the play-by-play.
- **Tracing** = "recording the **stream of the events**, to provide additional
  information on **where** and **when** each takes place during the execution" [s13]. You
  get a time-ordered log, so you can reconstruct interactions.

Two consequences the deck stresses:

- **Results depend on the test case chosen** [s14]. A profile is only as representative
  as the input you ran.
- **The amount of events monitored drives report size** [s15-16]. Profiles are compact
  (**a few MBs**); traces can be huge (**hundreds of GBs**) because they keep every event
  with its timestamp.

### 1.3 The cost of measurement [s17-20]

**Measurement affects performance** — you cannot observe for free [s17]:

- **Overhead** — measurement takes time and memory and uses CPU, lowering performance; it
  *increases with the amount of data collected*.
- **Perturbation** — measurement might *hinder optimization strategies* (the instrumented
  build is not the build you would ship).
- **Accuracy** — limited by the resolution of the timer/counter, and *more detrimental on
  short routines* (a routine shorter than the timer tick is measured badly).

The mitigations, introduced as a three-step recipe [s18-20]:

1. **Focus your measurement on a set of events** — don't measure everything.
2. **Identify a suitable test case for this set** — a run that exercises those events.
3. **Apply filters** — exclude the events you don't care about to cut overhead and report
   size.

### 1.4 What limits performance, and what to measure [s21-27]

**Performance factors** are split into two columns [s21-24]:

- **Sequential**: vectorization; L*X* cache misses / hints; instructions-per-cycle
  (`#instruction / #cycles`); I/O volume.
- **Parallel**: workload decomposition; communications; amount of serial work;
  synchronization.

**Main metrics**, again split by technique [s25-27]:

- **Profiling metrics**: time; counts (visits); bytes transferred; hardware counters.
- **Tracing metrics**: timestamp; location; event-type; communicator. (These extra fields
  are exactly what lets a trace place an event in space and time.)

### 1.5 Reports: profiles and traces [s28-31]

**Profiles** summarize performance metrics over the whole run [s28-29]. Two report shapes:

- **Flat profile** — a table of functions ranked by time (the deck shows a gprof-style
  table with columns `% time / cumulative seconds / self seconds / calls / self ms/call /
  total ms/call / name`). The key distinction raised here is **exclusive vs inclusive
  time**: *exclusive* (self) = time in the function's own body; *inclusive* (total) =
  the function plus everything it calls.
- **Call graph** — the same data resolved along the *calling path*, so you see which
  parent called which child and how time flows down the tree.

**Traces** show "the variation of the performance metric and the distribution of the event
over time" [s30-31]. The example is an Nsight Systems timeline: the vertical axis is
**process / thread / CUDA stream**, the horizontal axis is **time**, and each bar is an
event (kernels, CUDA API calls, NVTX ranges). This is the view that reveals *interaction*
between execution units — impossible to see in a flat profile.

### 1.6 Measurement techniques: sampling vs instrumentation [s32-37]

**Sampling** [s32]:
- The OS interrupts the CPU at regular intervals to record the currently-executing
  instruction.
- The profiler correlates each record with the corresponding routine / source line.
- The frequency of a routine (or line) is therefore *estimated statistically* — you infer
  "this function took 30% of time" from the fraction of samples that landed in it.

**Instrumentation** [s33-34]:
- Special code is added at the **beginning and end of functions** to measure an event.
- Two kinds:
  - **Source-code modifiers** — intrusive, need to recompile the code. Tagged **+EFFORT,
    +INFO** (more work, but richer/more precise data).
  - **Binary profilers** — work at runtime, inserting instrumentation at the *first
    assembly instruction of each routine*. Tagged **-EFFORT, -INFO** (less work, less
    detail).

**Pros and cons** [s35-37]:

| | Sampling | Instrumentation |
|---|---|---|
| ✔ | negligible overhead | suitable for irregular codes |
| ✔ | no modification of source code | provides the *exact* frequency |
| ✔ | suitable for frequent small routines | |
| ✘ | requires long runs and regular codes | large overhead on short routines |

The practical note: for sampling, **compile with debug symbols** so the profiler can map
addresses back to source lines [s37]. The two techniques answer the earlier accuracy
warning: sampling is cheap but statistical (needs long, regular runs); instrumentation is
exact but expensive on short/frequent routines.

### 1.7 Real-life example: profiling alone is not enough [s38-40]

The instructor's own case. An application distributes 216 *independent* `solve_linter`
calculations across MPI ranks, each rank offloading to one GPU — so in principle it should
scale to 216 GPUs with little communication [s38]. But the measured efficiency collapses:

| GPUs | 1 | 24 | 108 | 216 |
|------|---|----|----|-----|
| efficiency | 1.0 | 0.9 | 0.78 | 0.56 |

The lesson stated on the slide: **"Most of the time profiling an application is not enough
to identify the bottleneck."** Profiling is a *starting point* to know which events to
focus on; **traces are fundamental to understand how processes, streams and threads
interact.** The Vampir trace views at 108 and 216 GPUs [s39-40] ask "Does it *sound*
well?" — you look at the timeline for gaps, serialization and `MPI_Barrier` regions that a
number alone would hide.

### 1.8 External timers [s41-43]

The cheapest measurement: no recompilation, negligible overhead, and it draws the
**wall/elapsed time vs CPU usage** distinction [s41].

- **`time`** (shell keyword, works for pipelines): `time ./matrixmul.exe` prints
  - `real 0m7.357s` → wall-clock time
  - `user 0m7.335s` → CPU busy in user space
  - `sys 0m0.020s` → CPU busy in system calls
- **`/usr/bin/time`** (the executable, `man time`) adds resource data:
  `7.30user 0.00system 0:07.31elapsed 99%CPU (…22912maxresident)k …pagefaults`. Running
  `/usr/bin/time sleep 10` shows `0:10.00elapsed 0%CPU` — 10 s of wall time but ~0% CPU,
  because `sleep` does no work.
- **`date`** for scripts: capture `start_time=$(date +"%s")`, run, `end_time=$(date
  +"%s")`, subtract. **Wall time** is defined as "the difference from the start to the end
  of a program execution, **for all threads or tasks**."

**Wall vs CPU time relationship** [s42]:
- **CPU time < wall time** when the program waits: downloading data from the network,
  contended resources, disk I/O.
- **CPU time > wall time** on multicore (e.g. OpenMP), because CPU time sums across cores.
  The ideal is `CPU time = walltime × cores` (perfect utilization of every core).

**`top`** — the table of processes [s43] — shows per-process wall time, memory and CPU
usage, including MPI processes. With `mpirun -np 4 ./mpimatrix.exe` you see four
`matrixmul_mpi` processes each near 100% CPU; with `export OMP_NUM_THREADS=8` a single
OpenMP process shows **`%CPU` of 797.4 — CPU > 100% because OpenMP threads run on many
cores at once**.

### 1.9 Instrumentation with clocks [s44-47]

Wrap a region of source with timing calls — `CALL start_clock()` / `[code] `/ `CALL
stop_clock()` — to find the most time-consuming portion [s44]. It is intrusive and adds
overhead. Timer functions:

- **Serial** [s44]: Fortran 77 `etime()`, `dtime()`; Fortran 90 `cpu_time()`,
  `system_clock()`, `date_and_time()`; C/C++ `clock()`.
- The critical distinction [s45-46]: **`SYSTEM_CLOCK` measures wall time**, **`CPU_TIME`
  measures CPU time**, and **`cpu_time` is sensitive to the number of threads**. On a
  matrix multiply with `OMP_NUM_THREADS=4`, `system_clock` reports walltime **3.839 s**
  while `cpu_time` reports **14.85 s** — the CPU time is ~4× the wall time because it sums
  the four threads' work. (The `real/user/sys` from `time` confirm it: `user` ≈ 15 s.)
- **Parallel** [s47]: OpenMP `omp_get_wtime()`, MPI `MPI_Wtime()` — both return wall time
  as `DOUBLE PRECISION`; you bracket the region and subtract (`t2 - t1`).

### 1.10 Hardware performance counters and PAPI [s48-50]

**Hardware counters** are "special-purpose registers built into modern microprocessors to
store the counts of hardware-related activities" [s48]. They are used to compute *derived*
metrics such as instructions-per-cycle and load imbalance in MPI, and they are **more
accurate with low overhead** than software timing.

**PAPI** is a "high-level portable interface providing access to hardware counters"
[s49]. Event families:
- Memory-hierarchy access events — `PAPI_LX_DCM` (data cache misses at level X = 1,2,3…).
- Cycle / instruction counts — **`PAPI_TOT_CYC`**, **`PAPI_TOT_INS`**, `PAPI_VEC_INS`.
- Pipeline status — `PAPI_MEM_SCY`.

PAPI's APIs are callable from C/Fortran to start/stop/read counters, and PAPI is
integrated into VTune, Scalasca and TAU. The command **`papi_avail`** lists the presets
plus hardware info [s50] — e.g. PAPI version 6.0.0.1, 128 total cores, 4 SMT threads/core,
16 cores/socket, 2 sockets, 5 hardware counters, 384 max multiplex counters — followed by
the preset table (`PAPI_L1_DCM`, `PAPI_L1_ICM`, `PAPI_L2_DCM`, `PAPI_L3_DCM`, …).

### 1.11 The roofline model [s51-54]

A **visual representation of application performance** [s51]. Procedure:

1. **Extract parameters**: **WORK** = number of operations performed by a kernel; **MEMORY
   TRAFFIC** = number of bytes moved by a kernel.
2. **Compute metrics**:
   - **Operational (arithmetic) intensity** = `WORK [FLOPs] / MEMORY TRAFFIC [BYTES]`.
   - **Performance** = `WORK [FLOPs] / TIME [s]`.
3. **Plot** the kernel as a point `K = (x = operational intensity, y = performance)` on a
   log-log grid [s52].
4. **Compare** with the machine's *peak performance* (GFLOPs/s) and *peak bandwidth*
   (GB/s) [s51].

The roof has two pieces [s53]: a **slanted line `y = BW [GB/s] × x`** (the
bandwidth-limited region) and a **horizontal line at peak performance** (the
compute-limited region). Their crossing is the *ridge point*. A kernel to the left of the
ridge is **memory-bound** (limited by `O1`); to the right it is **compute-bound** (limited
by `O2`). The vertical gap between a point and the roof above it is the headroom you can
still recover. The real Intel Advisor chart [s54] stacks many roofs — `L1 Bandwidth 603.17
GB/s`, `L2 179.35`, `L3 38.41`, `DRAM 15.95 GB/s`, and compute peaks from `Scalar Add Peak
7.58 GFLOPS` up to `Int8 Tile Dot Product (AMX) 7168.06 GINTOPS` — so you can see which
cache level and which instruction set each kernel is bound by.

### 1.12 PMPI: profiling by interposition [s55-56]

The MPI standard defines every function under **two names**: the weak symbol `MPI_Send()`
and the strong symbol `PMPI_Send()` [s55]. This lets you **interpose** your own `MPI_`
routine: it does the real communication by calling `PMPI_`, and adds extra bookkeeping
around it. The worked wrapper:

```c
static int    totalbytes = 0;
static double totalTime  = 0.0;
int MPI_Send(const void *start, int count, MPI_Datatype datatype,
             int dest, int tag, MPI_Comm comm) {
    double t_start = MPI_Wtime();
    int size;
    int result = PMPI_Send(start, count, datatype, dest, tag, comm);
    totalTime  += MPI_Wtime() - t_start;   // cumulates time
    MPI_Type_size(datatype, &size);
    totalbytes += count * size;            // cumulates message size
    return result;
}
```

**Library interposition** is exactly how VTune, Scalasca and TAU intercept MPI calls
[s56]: the application calls `MPI_Send`, the interpositioned library records data and
forwards to the real `PMPI_` in the MPI library. The TAU profile shown lists per-call time
for `MPI_Init_thread`, `MPI_Alltoall`, `MPI_Recv`, `MPI_Bcast`, `MPI_Barrier`,
`MPI_Allreduce`, `MPI_Comm_split`, plus user events for message sizes of all-reduce /
all-to-all / broadcast / reduce.

### 1.13 CUPTI [s57]

The **CUDA Profiling Tools Interface** enables building profiling/tracing tools that
target CUDA applications [s57]: CUDA-API instrumentation, OpenACC support, NVLink metrics,
and kernel performance counters. Its API groups are **Activity** (async record of CUDA
activities), **Callback** (notify on CUDA events), **Event** and **Metric** (kernel
counters/metrics), **Profiling** (metrics over a range), **PC Sampling**, and
**Checkpoint**. (This is the foundation the Nsight tools sit on — covered next week.)

### 1.14 Profiling software [s58-61]

Handmade profiling is tedious and adds overhead; dedicated tools are designed for **reduced
overhead** [s58]. They typically report: time in subroutines/functions (MPI, CUDA or
code-specific); number of calls; memory usage / allocations / deallocations / leaks; load
balancing and thread usage; I/O performance (bandwidth, bytes read/written); and
communication pattern (peculiar to traces).

The tool table [s59]:

| Tool | Scope | Availability |
|---|---|---|
| **Scalasca–Score-P** | profiling & tracing of MPI + multithreaded (OpenACC, CUDA, GPU kernels) | installed on Marconi, G100, Leonardo |
| **ITAC** (Intel Trace Analyzer & Collector) | quick tracing of Intel-compiled apps | Intel-licensed (Marconi, G100) |
| **Intel VTune Amplifier** | detailed profiling for Intel apps | Intel-licensed (Marconi, G100, Leonardo) |
| Extrae/Paraver | general-purpose tracing | installable on CINECA machines |
| Valgrind | memory debugging | installed on CINECA machines |
| **TAU** | profiling & tracing, PAPI | installable on CINECA machines |
| Vampir | tracing | under license |
| Darshan | I/O | installable on CINECA machines |
| **Nsight Systems** | traces of GPU-accelerated apps | CLI on Leonardo, G100 |

Crucial gotcha [s60]: these tools **usually require to be installed with the same compiler
and MPI implementation as the program to be profiled.** A tool-coverage map [s61] places
each tool against the programming model: ITAC / APS / Scalasca for MPI+OpenMP; VTune
(Advisor) also for OpenMP-target offload; Score-P / TAU / Vampir and NVHPC Nsight for
OpenACC/CUDA/GPU offload.

### 1.15 The performance-engineering workflow [s62-66]

A four-phase loop (Vi-HPS) [s62-66]:

1. **Preparation** — select a relevant test case; select the events to observe; prepare
   the app with debug symbols; insert extra code.
2. **Measurement** — collect performance data; aggregate the collected data.
3. **Analysis** — calculate derived metrics; prepare results.
4. **Optimization** — apply modifications to alleviate the bottleneck (source code,
   runtime). Then loop back to Preparation.

### 1.16 Hands-on: the GNU profiler (gprof) [s67-69]

gprof is a **hybrid of instrumentation and time-based sampling** [s67]. It records
**events** (user-function instrumentation), **metrics** (exclusive time, inclusive time,
counts, time/call), and produces two **profiles** (flat profile and a basic call graph).
Three phases and their commands:

| Phase | What happens | Command |
|---|---|---|
| **Instrumentation** | probes added to source at compile time | `gfortran -pg source.f90 -o app.exe` |
| **Measurement** | a daemon collects event metrics while running; a binary `gmon.out` is generated | `./app.exe` |
| **Postprocessing** | turns the binary profile into human-readable tables | `gprof app.exe gmon.out` |

The hands-on program is compiled and run with `./bin/cfd_serial.exe 128 30`.

**Flat profile** — summary over time and call paths [s68]. Column meanings *as defined on
the slide*:
- **% time** — time spent in each function (%)
- **calls** — number of calls
- **cumulative seconds** — time spent up to each function (s)
- **self seconds** — time spent in this function only (s)  ← *exclusive*
- **self ms/call** — average time spent in the function per call
- **total ms/call** — average time in function + descendants per call  ← *inclusive*

In the worked output the hotspot is obvious: `__core_MOD_evolve` takes **99.88%** of time
(175.32 self seconds over 2000 calls, 87.66 ms/call), and everything else is ~0%.

**Call graph** — results resolved on the calling path [s69]. Reading rules:
- Parent routines are listed *above* and children *below* each routine's entry.
- Self-called (recursive) routines are marked with `[+]`.
- Two warnings: **performance overhead can be large**, and **MPI and external routines are
  not instrumented** by gprof (so it is a single-process, CPU-only tool — for MPI you need
  the tools in Section 2).

---

## Section 2 — MPI and OpenMP: common bottlenecks and how to spot them (`mpi.pdf`)

### 2.1 Why parallelize, and why it gets hard [s2-7]

The building-painter analogy. One worker paints only so fast, and a bigger roller doesn't
help — **you need more workers** [s2-3]. Multiple workers do more *if organized properly*
[s4]. Organizing them means three things that also become the sources of overhead:
**distribute the workload** [s5], **synchronize**, and **exchange information** [s6-7].
Parallel speed is a *team-management* problem.

### 2.2 The three causes of MPI bottlenecks [s8-9]

The whole deck hangs on this list [s9]:

1. **Dependencies** — a rank must wait for another before it can proceed.
2. **Message transfers** — moving bytes over the network costs time.
3. **Workload distribution** — unequal work leaves some ranks idle.

### 2.3 Network metrics: latency and bandwidth [s10-11]

- **Latency** = "the time it takes for a message of size 0 to get to a destination" [s10].
  It **dominates the performance of small messages**, so the fix is to **bundle multiple
  small messages into a larger one** [s11].
- **Bandwidth** = "the maximum rate at which data can flow over the network" [s10]. It
  **dominates the performance of large messages**; the fix is to **improve network
  topology or mapping** [s11].

Both combine in the message-time model:

```
T = L + (1/BW) · V
```

time = latency `L` + (1 / bandwidth `BW`) × message volume `V`. Small `V` → the `L` term
dominates; large `V` → the `V/BW` term dominates. This single formula justifies both fixes.

### 2.4 Workload imbalance = "MPI wait time" [s12-13]

When ranks reach a collective (e.g. a `REDUCE`) at different times, the early ones **sit in
a barrier inside the MPI call** — this idle time is called **"MPI wait time"** [s12]. The
timeline shows three ranks finishing their `COMPUTATION` at different moments; the ranks
that finish early show a red **`wait`** block before the reduction can complete [s13].
Imbalanced computation therefore surfaces as time *inside* MPI calls, which is why naive
timing blames MPI when the real cause is uneven work.

### 2.5 Dependencies and serialization [s14-19]

A ring/pipeline pass-the-token pattern [s14]:

```c
MPI_Recv(&token, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Send(&token, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```

Each rank must receive before it can send, so the computations start staggered like a
staircase. **What happens as the number of ranks grows?** [s15-16] The staircase gets
longer: rank 0 starts immediately, but the last rank waits through every predecessor. The
conclusion on the slide: **"Scaling efficiency is limited by the waiting time inside MPI
calls"** [s17]. A real Vampir timeline [s18] shows this as diagonal bands of
`MPI_Sendrecv` / `MPI_Allreduce` across ranks P0–P20.

The fix [s19]: **use more efficient patterns or non-blocking operations**, and **overlap**
communication with GPU computing and/or data movement. The redrawn timeline replaces the
serialized `RECV`/`SEND` staircase with each rank doing `COMP` then a short `WAIT` — the
waits shrink because work now overlaps the transfers.

### 2.6 The POP efficiency model [s20-28]

"How do we *measure* the performance of an MPI application?" The answer is the **POP**
(Performance Optimisation and Productivity CoE) metric hierarchy [s20]. It is a tree of
**multiplicative efficiencies**, each a ratio in `[0,1]`, built from one underlying
picture: `P` processes, each spending computation time `C_i`, over a total runtime `T`.
Notation: `<C_i>` is the average of `C_i` over processes; `max(C_i)` is the slowest
process's computation.

Working down the tree [s21-28]:

- **Parallel process efficiency** = `<C_i> / T` [s21] — the fraction of total time spent,
  on average, in useful computation. It factorizes into the next two:

- **Load balance** = `<C_i> / max(C_i)` [s22] — average vs slowest computation. It is 1
  when every rank computes the same amount; below 1 means some ranks do more work than
  others. (The "collapse computations" diagram slides all compute blocks left; the yellow
  gap between the average and the max is the imbalance.) Its underlying cause is
  **instruction balance**, measured with hardware counters — **PAPI, PERF, LIKWID** [s23].

- **Communication efficiency** = `max(C_i) / T` [s24] — what you could still reach *if load
  were perfectly balanced*; the shortfall from 1 is pure communication cost. Note the
  factorization: `Parallel process efficiency = Load balance × Communication efficiency`.

  Communication efficiency itself splits into two [s25-26], using an **ideal-network**
  runtime `T(BW=∞)` — the time the program *would* take on a network with zero latency and
  infinite bandwidth, obtained by simulating the trace with **DIMEMAS (L=0, BW=∞)**.
  (Definitions restated: **latency** = time to deliver a zero-size message; **bandwidth** =
  bytes/second transferred.)

  - **Transfer efficiency** = `T(BW=∞) / T` [s25] — how much of the runtime is lost to
    *actual* data transfer (finite bandwidth/latency). Its causes: **overhead of small
    messages** and **network congestion with many procs** (and collectives).
  - **Serialization efficiency** = `max(C_i) / T(BW=∞)` [s26] — time lost to
    **dependencies** *even on a perfect network*. This is where the pass-the-token
    staircase of §2.5 shows up. Factorization: `Communication efficiency = Transfer
    efficiency × Serialization efficiency`.

- **Computational scalability** = `Σ C_i(N_1) / Σ C_i(N_n)` [s28] — does the *total*
  computational work stay constant as you scale from `N_1` to `N_n` processes? It is
  broken into **instruction scaling** and **IPC scaling**, and is hurt by **extra
  computation for domain decomposition** and **memory bottlenecks**.

At the top, **Global efficiency = Parallel efficiency × Computational scalability** — the
product of "are the processes busy with useful work?" and "is the total work staying
bounded?". The whole point of the tree: a single low number at the top is decomposed until
you can name *which* effect (imbalance, transfer, serialization, or growing work) is
responsible. The `T_ideal` diagram [s27] visualizes it: starting from the real timeline,
you successively remove transfer, then serialization, then load-balance losses to reach the
ideal.

### 2.7 POP metrics at scale [s29]

The intended use: **watch how each number changes with the number of MPI ranks to see what
limits scaling.** The worked table:

| metric (cores →) | 48 | 96 | 192 | 384 | 768 |
|---|---|---|---|---|---|
| Global Efficiency | 0.93 | 0.94 | 0.93 | 0.84 | 0.76 |
| ↳ Parallel Efficiency | 0.93 | 0.91 | 0.87 | 0.77 | 0.68 |
| ↳↳ Load balance | 0.99 | 0.98 | 0.98 | 0.97 | 0.95 |
| ↳↳ Communication Efficiency | 0.94 | 0.92 | 0.89 | 0.79 | 0.72 |
| ↳↳↳ **Serialisation** | 0.95 | 0.94 | 0.92 | 0.85 | **0.81** |
| ↳↳↳ Transfer efficiency | 0.99 | 0.99 | 0.97 | 0.94 | 0.89 |
| ↳ Computational Scaling | 1.00 | 1.03 | 1.07 | 1.09 | 1.12 |
| ↳↳ Instruction Scaling | 1.00 | 0.99 | 0.97 | 0.95 | 0.92 |
| ↳↳ **IPC Scaling** | 1.00 | 1.05 | 1.10 | 1.18 | **1.27** |

How to read it: load balance and transfer stay high, so those are *not* the problem.
**Communication efficiency, driven by serialisation (0.95 → 0.81), is what erodes parallel
efficiency** — i.e. dependencies dominate at scale. Meanwhile IPC scaling *rises* to 1.27
(computation actually gets more efficient per core, e.g. smaller per-rank data fits cache),
which is why computational scaling stays ≥ 1. There is a **Score-P plugin** that computes
this table automatically.

### 2.8 POP metrics for OpenMP, and the hybrid tree [s30-31]

For a shared-memory (OpenMP) program the bottleneck causes are [s30]:

- **Serialization** — time spent computing *outside* OpenMP. Ask: "are there costly serial
  regions that could exploit OpenMP?"
- **OpenMP parallel efficiency**, hurt by:
  - **Load balance** — improve scheduling (and avoid barriers where possible).
  - **Overhead of forking** — *extend parallel regions across multiple loops* instead of
    forking repeatedly.
  - **Synchronizations**.
  - **Reduction, Atomic**.

For **hybrid MPI+OpenMP** the tree gains a thread branch [s31]: `Global efficiency →
Parallel efficiency (→ Parallel process efficiency, → Thread efficiency) +
Computational scalability`, where **Thread efficiency → OpenMP parallel efficiency + Serial
region efficiency**. (Reference handout: `pop-coe.eu/.../pop_hybrid_metrics_additive_handout.pdf`.)

### 2.9 Scalasca + Score-P [s32-37]

**Scalasca** is a "scalable performance analysis of large scale applications" tool from
**Jülich Supercomputing Centre** [s32]. It scales by exploiting the available processors
and memory, and is the **ideal tool to investigate synchronization issues and workload
imbalance in communication-intensive applications with thousands of cores.** It instruments
**MPI, OpenMP, and user-defined routines**, and supports **hardware-counter metrics via
PAPI**. The division of labour: **Score-P** does source-code instrumentation; **CUBE /
Vampir** visualize profiles and traces (ITAC works too). Scalasca can measure four things at
once [s34]: **MPI** communication, **threads** (fork/join), **hardware counters**, and
**user** source-code regions.

**Main vs derived metrics** [s36]:
- **Main**: Time (computation; MPI: synchronization / management / communications / MPI
  I/O; OpenMP); counts; bytes transferred (point-to-point vs collective); MPI-I/O bytes;
  plus hardware counters (PAPI) and — for traces — timestamps, locations, communicators.
- **Derived**: MPI/thread efficiency from the POP assessment (plugin); workload balance and
  synchronization patterns — **computational imbalance**, **idle threads**, **late
  receiver**, **wait at barrier**.

Two **inefficiency patterns** worth naming [s36-37]:
- **Late receiver** — an `MPI_Send` blocks because the matching `MPI_Recv` is posted late;
  the sender wastes time waiting for the receiver to be ready.
- **Wait at barrier** — processes that reach `MPI_Barrier` early wait for the last one to
  arrive (the arrow marks the wasted interval).

The pattern gallery [s37] also covers `MPI_Reduce` (root waits), `MPI_Allreduce`, parallel-
region-body imbalance, and the non-blocking `MPI_Isend/Irecv/Wait` alternatives. (Reference:
`vi-hps.org/.../vi-hps-tw-original-Scalasca_Patterns.pdf`.)

### 2.10 The Scalasca / Score-P workflow [s35, s38-43]

The **CFD-Jacobi** motivating case [s38]: a plot of time and speedup vs MPI tasks, with
**speedup = `t_1 / t_N`** (serial time over N-task time). Speedup saturates (≈18–19 at 40
tasks) and the slide asks "what does prevent an efficient scaling?" — the workflow below is
how you find out.

**Step 1 — Instrument** (add a prefix at the compile & link stage) [s39, s35]:

```bash
scorep [--mpp=mpi --user --openmp --verbose ...] mpif90 source.f90 -o myapp.exe
```

The options choose which events to instrument (MPI, user regions, OpenMP). With CMake or
autotools the invocation differs — check the user guide.

**Step 2 — Measure** (add a prefix to `mpirun`; `-t` enables tracing) [s39, s35]:

```bash
scalasca -analyze [-t] mpirun -np NP ./myapp.exe
```

Results land in a folder `scorep_myapp_<NP>X<M>` (NP MPI ranks × M OpenMP threads).
**Golden rule: never trace at first** — a first pass produces only a *profile*
(`profile.cubex`); tracing (`-t`, producing `traces.otf2` and `scout.cubex`) comes later,
after filtering.

**Step 2½ — Size the trace before tracing** [s40]:

```bash
scorep-score scorep_cfd_14_trace/profile.cubex
```

```
Estimated aggregate size of event trace:            1181MB
Estimated requirements for largest trace buffer:      85MB   (max_buf)
Estimated memory requirements (SCOREP_TOTAL_MEMORY):  87MB
```

The hint: set `SCOREP_TOTAL_MEMORY=87MB` to avoid intermediate buffer flushes (which
perturb timing), or reduce requirements with USR-region filters. The accompanying table
breaks time and buffer size by region class — `ALL` 291.42 s (100%), **`MPI` 192.44 s
(66%)**, `USR` 59.01 s (20.2%), `COM` 39.63 s (13.6%), `SCOREP` 0.35 s (0.1%) — so you can
already read off how much time is spent in MPI vs user routines.

**Step 2¾ — Inspect and filter** [s41-42]. `scorep-score -r` prints a *flat profile* of the
traced events ordered by buffer size:

```bash
scorep-score -r scorep_cfd_14_trace/profile.cubex
```

Here the top buffer consumers are I/O routines run at the end (`cfdio_mp_colfunc_`,
`cfdio_mp_hue2rgb_`) — large buffers, uninteresting for analysis, so filter them out. Write
a filter file:

```
SCOREP_REGION_NAMES_BEGIN
  EXCLUDE
        cfdio_mp_colfunc_
        cfdio_mp_hue2rgb_
SCOREP_REGION_NAMES_END
```

Re-check the estimate with the filter applied:

```bash
scorep-score -f filter.file scorep_cfd_14_trace/profile.cubex
```

The trace collapses from 1181 MB to **5 MB** (`max_buf` 85 MB → 320 kB;
`SCOREP_TOTAL_MEMORY` 87 MB → 4097 kB). Satisfied, repeat the measurement with
`scalasca analyse -f filter.file`.

**Step 3 — Postprocess & visualize** [s43]:

```bash
scalasca -examine [options] scorep_myapp_NPXM     # derives metrics, opens the profile in CUBE
cube scorep_myapp_NPXM/summary.cubex               # open the .cubex GUI directly
```

Two useful environment variables: `SCOREP_PAPI_METRIC` (e.g. `PAPI_TOT_INS`,
`PAPI_TOT_CYC`) selects hardware counters; `SCOREP_TOTAL_MEMORY` raises the buffer to avoid
flushes. (Full list at the Score-P `scorepmeasurementconfig` docs.)

### 2.11 Reading the CUBE GUI and manual instrumentation [s44-46]

The **CUBE GUI** [s44] has three linked panels, read left-to-right as "what → where →
who":
- **Metric tree** (left) — granular derived metrics (Time, Visits, bytes sent/received,
  `PAPI_TOT_INS`, `PAPI_TOT_CYC`, …).
- **Call tree** (middle) — a **color-coded box identifies the hotspots** in the call path.
- **System tree** (right) — the **distribution of MPI communications and barriers across
  ranks** (MPI Rank 0…23). The selected metric's value colors each level.

**Manual instrumentation** [s45] — Score-P auto-instruments user routines, but you can label
arbitrary regions with a name:

```fortran
#include "scorep/SCOREP_User.inc"                    ! (after "implicit none")

SCOREP_USER_REGION_DEFINE( myhandle1 )               ! declare a handle

SCOREP_USER_REGION_BEGIN( myhandle, "block", SCOREP_USER_REGION_TYPE_COMMON )
   ...                                               ! code named "block"
SCOREP_USER_REGION_END( myhandle )
```

The CUBE GUI can also host the **POP-assessment plugin** [s46], which reports the POP
efficiencies (e.g. Parallel Efficiency 0.83, Load Balance 0.92, Communication 0.99) and
hardware counters via PAPI directly on the measured run, listing candidate call paths to
optimize.

### 2.12 Case study: POP assessment and the Jacobi blocking→non-blocking fix [s47-50]

Run the POP assessment across rank counts and watch the collapse [s47]:

| ranks | 4 | 12 | 24 | 48 |
|---|---|---|---|---|
| Parallel Efficiency | 0.87 (very good) | 0.44 (fair) | 0.18 (very poor) | 0.06 (very poor) |
| Load Balance | 0.97 | 0.85 | 0.93 | 0.91 |
| Communication Efficiency | 0.91 | 0.51 | 0.19 | 0.07 |

Load balance stays fine, but **communication efficiency craters** — the slide asks "try a
different algorithm?". The fix [s48]: convert the blocking `mpi_sendrecv` halo exchange to
**non-blocking** `mpi_isend` + `mpi_irecv` followed by a single `mpi_waitall`:

```
call mpi_isend(grid(1,1),      ny, MPI_DOUBLE, myrank-1, 0, MPI_COMM_WORLD, requests(1), ierr)
call mpi_irecv(grid(1,0),      ny, MPI_DOUBLE, myrank-1, 0, MPI_COMM_WORLD, requests(2), ierr)
call mpi_isend(grid(1,local_nx), ny, MPI_DOUBLE, myrank+1, 0, MPI_COMM_WORLD, requests(3), ierr)
call mpi_irecv(grid(1,local_nx+1),ny, MPI_DOUBLE, myrank+1, 0, MPI_COMM_WORLD, requests(4), ierr)
call mpi_waitall(4, requests, MPI_STATUSES_IGNORE, ierr)
```

The result [s49-50]: **non-blocking improves scalability** — the blocking version peaks at
speedup ≈9 (12 tasks) then falls, while non-blocking reaches ≈17 (24 tasks) before it too
degrades. The POP assessment explains *why*: **serialization efficiency jumps from 0.586
(blocking) to 0.985 (non-blocking)** — "non blocking reduces dependencies" — with parallel
efficiency rising 0.383 → 0.548. This is the whole method in one example: the metric that
moved (serialization) named the bottleneck (dependencies), and the fix (non-blocking)
targeted exactly it.

### 2.13 ITAC — Intel Trace Analyzer and Collector [s51]

Source-code instrumentation providing a summary plus a **Trace Collector and Analyzer for
communication hotspots** [s51]: MPI and OpenMP time and imbalance, USR-routine
instrumentation, and — like DIMEMAS — **ideal traces to measure interconnect vs waiting
time**. Workflow:

```bash
module load intel-oneapi-mpi intel-oneapi-compilers intel-oneapi-itac
mpiifort -trace [-tcollect] mycode.f90      # instrument at compile
srun ./mycode                                # produces *.stf
traceanalyzer *.stf                          # open the GUI
```

### 2.14 VTune [s52-57]

Intel **VTune** requires source-code instrumentation and analyzes **CPU and (Intel) GPU**
performance for **MPI, OpenMP and OpenMP-target** codes, providing traces of CPU/GPU
utilization, memory access and cache bandwidth [s52]. Its collections:

```bash
vtune -collect <collection-name> ./matrix
#   performance-summary | hotspot | memory-access | hpc-performance
```

The worked example is an OpenMP prime-counting loop (`!$omp parallel do reduction(+:total)`)
[s53]:

```bash
module load intel-oneapi-vtune
ifort -g -qopenmp -O2 -parallel-source-info=2 prime_openmp.f90 -o prime_openmp.exe
vtune -collect hpc-performance ./prime_openmp.exe
```

`hpc-performance` reports **time spent in the implicit barriers** of each parallel region
[s54]. The summary [s55]: Elapsed 15.247 s, x87 GFLOPS 4.988, CPI 0.899, 12 threads, but
**Effective CPU Utilization only 13.4%** (6.436 of 48 logical cores). Serial time is
negligible (0.101 s, 0.7%); the parallel region is 99.3% of time but its **OpenMP Potential
Gain is 6.917 s (45.4%)** — nearly half the parallel time is wasted.

The diagnosis and fix [s56-57]: grouping by *OpenMP Barrier-to-Barrier Segment* pins the
potential gain on **Imbalance = 4.703 s** under a **static schedule** — "workload is not
balanced". Switching to a **dynamic schedule** drops imbalance to **0.001 s** and elapsed
time from 10.342 s to **5.711 s**. Same story as the MPI Jacobi case: the metric that lit
up (imbalance) named the cause, and the targeted fix (dynamic scheduling) removed it.

---

## Likely exam angles

1. **Profiling vs tracing** — define both [Deck1 s13]; know that profiling *summarizes*
   over the interval (few MB) while tracing *records the event stream* with
   timestamp/location/event-type/communicator (up to hundreds of GB), and that "results
   depend on the test case."
2. **The three costs of measurement** — overhead, perturbation, accuracy [Deck1 s17], and
   the mitigation recipe: focus on a set of events → pick a test case → apply filters.
3. **Sampling vs instrumentation** — the mechanism of each [Deck1 s32-33] and the pros/cons
   table [Deck1 s36]: sampling = negligible overhead / statistical / needs long regular
   runs / compile with debug symbols; instrumentation = exact frequency / suitable for
   irregular codes / large overhead on short routines. Know that gprof is a *hybrid*.
4. **Exclusive vs inclusive time** — self (function body) vs total (function + descendants)
   [Deck1 s28, s68], and reading a gprof flat profile / call graph, including the caveats
   that gprof's overhead can be large and **it does not instrument MPI**.
5. **Wall vs CPU time** — definitions and the two regimes [Deck1 s41-42]: CPU<wall when
   waiting (I/O, network), CPU>wall on multicore, ideal `CPU = wall × cores`; that
   `SYSTEM_CLOCK`=wall and `CPU_TIME`=CPU-and-thread-sensitive [Deck1 s45-46]; and that
   `top` shows >100% CPU for OpenMP.
6. **Timer functions** — which is which: `system_clock`/`cpu_time` (Fortran, serial),
   `omp_get_wtime` (OpenMP), `MPI_Wtime` (MPI) [Deck1 s44-47].
7. **Roofline** — operational intensity = FLOPs/byte, performance = FLOPs/s, the slanted
   `y = BW·x` roof vs the flat peak-performance roof, and memory-bound (left) vs
   compute-bound (right of the ridge) [Deck1 s51-53].
8. **PMPI interposition** — MPI_/PMPI_ weak/strong symbols and how a wrapper accumulates
   time and bytes [Deck1 s55]; that VTune/Scalasca/TAU all use library interposition.
9. **Network model `T = L + V/BW`** — latency dominates small messages (bundle them);
   bandwidth dominates large messages (improve topology/mapping) [Deck2 s10-11].
10. **The three MPI bottleneck causes** — dependencies, message transfers, workload
    distribution [Deck2 s9] — and how each appears (serialized staircase; transfer time;
    MPI wait time at a barrier).
11. **The POP efficiency hierarchy — the centerpiece.** Be able to draw the tree and write
    every formula [Deck2 s21-28]:
    - Parallel process efficiency `= <C_i>/T` = Load balance × Communication efficiency
    - Load balance `= <C_i>/max(C_i)`
    - Communication efficiency `= max(C_i)/T` = Transfer × Serialization
    - Transfer efficiency `= T(BW=∞)/T`
    - Serialization efficiency `= max(C_i)/T(BW=∞)`
    - Computational scalability `= ΣC_i(N_1)/ΣC_i(N_n)`; Global eff = Parallel eff ×
      Computational scalability.
    Know that `T(BW=∞)` is simulated by **DIMEMAS (L=0, BW=∞)**, that load balance is
    diagnosed with HW counters (PAPI/PERF/LIKWID), and that serialization ← dependencies,
    transfer ← small-message overhead + congestion.
12. **Reading the POP-at-scale table** [Deck2 s29] — identify which efficiency erodes
    (serialisation/communication) vs which stays flat (load balance, transfer) to conclude
    "dependencies limit scaling."
13. **OpenMP bottlenecks** — serialization outside OpenMP, load balance (scheduling),
    fork overhead (extend regions), synchronization, reduction/atomic [Deck2 s30]; the
    hybrid tree adds Thread efficiency = OpenMP parallel × Serial region [Deck2 s31].
14. **Late receiver vs wait at barrier** — the two Scalasca patterns [Deck2 s36].
15. **The Scalasca/Score-P workflow end-to-end** [Deck2 s39-43] — `scorep …
    mpif90` (instrument) → `scalasca -analyze [-t] mpirun` (measure) → `scorep-score`
    (size the trace, `SCOREP_TOTAL_MEMORY`) → filter file + `scorep-score -f` → `scalasca
    -examine` / `cube …summary.cubex`. Know the rule **"never trace at first"** and why
    (buffer flushes perturb timing).
16. **Case-study reasoning** — blocking→non-blocking Jacobi raises serialization efficiency
    0.59→0.98 ("reduces dependencies") [Deck2 s48-50]; static→dynamic OpenMP schedule cuts
    imbalance 4.703 s→0.001 s and halves elapsed time [Deck2 s56-57]. Expect to be asked
    "which metric identifies the bottleneck and which fix targets it."
17. **Tool-to-model mapping** [Deck1 s59-61, Deck2 s51-52] — gprof (serial CPU, no MPI);
    Scalasca/Score-P + CUBE/Vampir (MPI/OpenMP at scale); ITAC (Intel MPI/OpenMP traces,
    ideal interconnect); VTune (CPU/Intel-GPU, memory & OpenMP imbalance); Nsight (GPU);
    and the gotcha that a profiler must be built with the *same compiler and MPI* as the
    application.
