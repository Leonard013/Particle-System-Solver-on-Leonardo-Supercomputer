# Module 2 — OpenMP (shared-memory parallelism)

*Course notes reconstructed from the CINECA / PoliMI lecture deck "OpenMP" by Alessandro Romeo, PhD (76 slides). Slide references are given as [sNN] or [sNN–MM]. This chapter follows the deck's own order: Introduction → the `parallel` construct and race conditions → work-sharing constructs → data clauses → synchronization constructs → explicit work-sharing and loop optimizations.*

The deck's stated agenda [s2]: Introduction; `parallel` construct and race conditions; Work-sharing constructs; Scheduling directives; Data clauses; Synchronization constructs; Explicit work-sharing and loop optimizations. Recommended references [s3]: the official OpenMP Reference Guides, Tim Mattson's OpenMP tutorial, the official OpenMP compilers/tools list, and Stack Overflow's `openmp` tag.

---

## 1. Introduction: why shared-memory parallelism [s4–13]

### 1.1 SPMD — the mental model [s5]

Real numerical codes rarely spell out a different instruction stream per processor. Instead they use **MIMD** hardware (Multiple Instruction, Multiple Data — each core can run its own instructions on its own data) by running **the same program on different portions of the data**. Because the instruction stream a program follows is usually *data-dependent* (branches taken depend on the values being processed), the same program text ends up executing differently on each core — that is MIMD execution in practice.

This is the **SPMD** model — *Single Program, Multiple Data* — and it is the deck's principal way of specifying parallel algorithms. The illustrating diagram: the program starts once, splits into several concurrent `Print` operations, all meet at a **barrier**, then one more `Print` runs, and the program ends. Keep this shape in mind; every OpenMP program looks like it.

### 1.2 Two kinds of parallelism [s6]

- **Thread / task parallelism** is based on *partitioning the operations of the algorithm*. If the algorithm is a sequence of independent operations (task 1, task 2, …), each task can be sent to a different CPU. This is "different work on different cores."
- **Data parallelism** means *spreading the data* across the processors. Every processor performs the **same operation**, but on a **different data set** — typically distributing the elements of an array across the computing units. This is "same work, different slices of data," and it is the workhorse of numerical HPC.

### 1.3 Processes vs. threads [s7–8]

A **process** [s7] is an instance of a program being executed: it contains the program code plus its current activity, and **each process has a complete, private set of its own variables**. An OS runs many processes at once (e.g., a compiler while you edit text) via **context switching** — saving one process's state in its Process Control Block (PCB) and loading another's. Context switches between separate memory contexts are relatively expensive because nothing is shared.

A **thread** [s8] is a unit of execution *inside* a process. The key picture: in a single-threaded process there is one `code / data / files` region plus one set of `registers` and one `stack`; in a **multithreaded process** the threads **share** `code`, `data`, and `files`, but each thread has its **own** `registers` and its **own** `stack`. Two consequences the deck emphasizes:

- The number of threads is **dynamic** during a process's execution (it can rise and fall).
- Running several concurrent threads **might** improve performance — this is where parallelism comes from.

This shared `data` region is exactly the "shared memory" that OpenMP exploits, and the private stacks are what make per-thread private variables possible. Threads are cheaper to switch between than processes because they already share most of their context.

### 1.4 Why OpenMP? [s9–10]

Consider a serial routine [s9]:

```cpp
void add(double* x, double* y, size_t size) {
    for (size_t i = 0; i < size; i++)
        x[i] = x[i] + y[i];
}
```

On a multi-core machine this **uses only one hardware core**, performs the additions **one by one**, and makes **no concurrent access to memory** — three idle cores. The deck's verdict: **"Waste of resources!"**

You *could* parallelize by hand [s10] — literally writing one loop per core (`core 1` does `0 … size/4`, `core 2` does `size/4 … size/2`, and so on) so all four cores work at once. It works, but it is tedious, error-prone, and hard-codes the core count. OpenMP exists to get this effect **without** rewriting the loop by hand.

### 1.5 What OpenMP can do [s11]

OpenMP is a standard supporting three parallelism paradigms:

- **Threaded parallelism** (multi-core, shared memory) — the focus of this module.
- **Vectorized execution** (SIMD).
- **Offload execution on GPUs.**

Its features fall into five families [s11], which map directly onto the rest of this chapter:

| Family | Role | Example |
|---|---|---|
| Parallel Control Structures | govern flow of control | `parallel` |
| Work Sharing | distribute work among threads | `for` / `sections` |
| Data Environment | scope variables | `shared`, `private` |
| Synchronization | coordinate thread execution | `barrier` |
| Runtime functions | query/set the runtime environment | `omp_set/get_*`, env variables |

### 1.6 Directive-based programming [s12]

OpenMP is **directive based**: you add directives (pragmas) to otherwise-serial code. This makes it **easy to port serial codes** to multi-core CPUs (and GPUs), and it works with **C/C++ and Fortran**. The syntax:

```c
// C/C++
#pragma omp construct [clauses]
{
    // code to parallelize
}
```

```fortran
!$omp construct [clauses]
    ! code to parallelize
!$omp end construct
```

### 1.7 Compiler flags [s13]

You must pass the right flag to *enable* OpenMP:

| Compiler | C/C++ | Fortran |
|---|---|---|
| GNU | `gcc/g++ … -fopenmp` | `gfortran … -fopenmp` |
| Intel | `icc/icpc … -qopenmp` | `ifort … -qopenmp` |
| NVHPC | `nvc/nvc++ … -mp` | `nvfortran … -mp` |
| IBM | `xlc/xlc++ … -qsmp=omp` | `xlf90 … -qsmp=omp` |

**Critical gotcha [s13]:** if the OpenMP flag is missing, the pragmas are **silently ignored** — the code still compiles (no error) but runs serially. Always remember the flag.

---

## 2. The `parallel` construct and race conditions [s14–27]

### 2.1 Hello, `#pragma`! [s15–16]

Starting from a serial "Hello World" [s15], you parallelize it by including `omp.h` and wrapping the statement in a `parallel` region [s16]:

```cpp
#include <iostream>
#include <omp.h>
int main() {
#pragma omp parallel
    {
        std::cout << "Hello World!" << std::endl;
    }
    return 0;
}
```

The output is **"Hello World!" printed several times**. **Warning [s16]:** you must `#include <omp.h>` whenever you use OpenMP *functions*.

### 2.2 Fork–join parallelism [s17]

This is the execution model behind every `parallel` region:

- **Fork:** the **primary (master) thread** spawns a **team** of secondary threads as needed. Parallelism is added *incrementally* until performance goals are met.
- **Join:** at the end of the parallel region the team ends and **only the primary thread remains**; between regions, execution is serial (just the primary thread).

So a program is a serial spine with parallel regions bulging out of it — fork at each region's start, join at its end.

### 2.3 What actually happened [s18]

In the Hello example [s18]:

- Threads execute the code in the block **redundantly** — every thread runs the whole block.
- Each thread evaluates `std::cout` once, so you get one line **per thread**.
- Threads are synchronized at the end of the parallel region by an **implicit barrier**.

So the number of lines equals the number of threads in the team — which raises the question: how is that number set?

### 2.4 Setting the number of threads [s19]

Three mechanisms, from "outside the code" to "inside the code":

- **Environment variable, whole terminal session:** `export OMP_NUM_THREADS=16; ./program`
- **Environment variable, single run:** `OMP_NUM_THREADS=16 ./program`
- **In the directive (clause):** `#pragma omp parallel num_threads(16)`
- **Via the runtime API:** `void omp_set_num_threads(int num_threads)` — affects *subsequent* parallel regions.

Thread numbering [s20]: OpenMP thread IDs run **from `0` up to `num_threads - 1`**.

**Best practices [s20]:**
- Rule of thumb: set the number of (software) threads **equal to the number of (hardware) cores**.
- **Avoid "magic numbers"** hard-coded in the source; prefer the `OMP_NUM_THREADS` option — otherwise "bad things might happen."

### 2.5 Priority hierarchy: how many hellos? [s21]

When several mechanisms disagree, which wins? Consider:

```cpp
int main() {
    omp_set_num_threads(2);          // set via API
#pragma omp parallel num_threads(4)  // clause on the region
    {
        omp_set_num_threads(8);      // affects only *future* regions
        std::cout << "Hello World!" << std::endl;
    }
}
```

Run as `OMP_NUM_THREADS=1 ./hello`. The **hierarchy** (lowest priority → highest) is:

1. `OMP_NUM_THREADS` (environment) — weakest,
2. `omp_set_num_threads(int)` (API),
3. **`num_threads` clause** — strongest.

So the region runs with **4 threads** (the clause overrides both the env var and the earlier API call). The `omp_set_num_threads(8)` *inside* the region only changes *subsequent* regions, of which there are none — so it has no visible effect here. Answer: **4 hellos.**

### 2.6 The runtime API [s22–23]

Most-used runtime functions [s22]:

| Function | Returns | Purpose |
|---|---|---|
| `omp_set_num_threads(int)` | void | number of threads for the **next** parallel region |
| `omp_get_num_threads()` | int | number of threads in the **currently executing** region |
| `omp_get_max_threads()` | int | max threads available for a subsequent region |
| `omp_get_thread_num()` | int | ID (0-based) of the calling thread in the current region |
| `omp_get_num_procs()` | int | number of processors available |
| `omp_get_wtime()` | double | wall-clock time in seconds since some fixed point in the past |
| `omp_get_wtick()` | double | timer resolution in seconds |

**Timing with `omp_get_wtime()` [s23]:**
- Measures **wall time** (real elapsed time, in seconds), *including* time when the CPU is idle.
- Returns the same global wall-clock value regardless of which thread calls it — it is **not specific to one thread**.
- High-resolution, but the actual precision depends on the platform and the OpenMP implementation.
- Ideal for **benchmarking whole parallel regions**, because it accounts for real time — including time spent waiting, doing I/O, or synchronizing.
- To measure a *specific* thread's performance, time it **inside** that thread individually.

(The deck poses `clock()` / `time()` from the C library and `std::chrono` as things to compare against; the point is that `clock()` typically measures CPU time summed across threads, which is misleading for parallel code — wall time is what you want.)

### 2.7 Data races [s24–25]

A **race condition** [s24] is "a flaw that occurs when the timing or ordering of events affects a program's correctness." A **data race** happens when two memory accesses:

1. **target the same location**,
2. **are performed concurrently by two threads**,
3. **are not both reads** (at least one is a write), and
4. **are not synchronization operations**.

Tools to handle data races [s24]: **barriers**, **atomic operations**, an **explicit data-sharing policy**, and more (all covered later).

The canonical example [s25]:

```c
int tid, nthreads{4};
omp_set_num_threads(nthreads);
#pragma omp parallel
    {
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
        for (int i = 0; i < 1e5; i++) {}   // lengthy work
        printf("Thread %d of %d.\n", tid, nthreads);
    }
```

Because `tid` is declared **outside** the region it is **shared**, so all threads write to the same variable. Two runs can print different things — sometimes garbled (`Thread 0 … Thread 0 … Thread 3 … Thread 3`), sometimes clean (`0,1,2,3`). The non-determinism is the data race; the fix (previewed here, formalized in §4) is to make `tid` private to each thread.

### 2.8 Exercise 1 [s26–27]

Task [s26]: add a `parallel` region in the right place; use `omp_get_wtime()` to measure time; get thread IDs inside the region; compile; run; optionally vary `OMP_NUM_THREADS`. "Do you get what you expected?"

The trap [s27]: putting only `#pragma omp parallel` around a loop that sums `c[i] = a[i] + b[i]` makes **every** thread run the **whole** loop (`i = 0 … n`) redundantly. So each element of `c` is summed `nthreads` times. **"ACHTUNG: `parallel` is executing the same code redundantly on every thread."** This motivates work-sharing.

---

## 3. Work-sharing constructs [s28–35]

### 3.1 Manual work-sharing [s29–30]

Before the automatic construct, the deck shows the manual version to build intuition [s29]: use the thread ID and the thread count to give each thread a **disjoint slice** of the iteration space. For 100 iterations and 4 threads: thread 0 → `0…24`, thread 1 → `25…49`, thread 2 → `50…74`, thread 3 → `75…100`. Each thread then runs only `for (i = start; i <= end; i++) c[i] = a[i] + b[i];`.

Exercise 1 revisited [s30] computes the bounds explicitly:

```cpp
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    std::size_t start = tid * n / nthreads;
    std::size_t end   = (tid + 1) * n / nthreads - 1;
    for (std::size_t i = start; i <= end; i++)
        c[i] = a[i] + b[i];
}
```

Now each element of `c` is summed **exactly once** — the output is correct. This is what work-sharing *is*; the `for` construct just automates it.

### 3.2 Work-sharing in SPMD [s31]

Properties of a work-sharing region:
- Work is shared **within a team** of threads.
- The region has **no internal barriers** — threads do not wait for each other *inside* the region.
- There is an **implicit barrier at the end**.
- **Every thread must encounter the same work-sharing regions and barriers** (you cannot have some threads skip a work-sharing construct — that is undefined).

### 3.3 The `for` construct [s32–34]

```cpp
#pragma omp parallel for
for (i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
}
```

Behavior [s32]:
- A team of threads is formed at the parallel region.
- **Loop iterations are split among the threads** (not run redundantly).
- There is a barrier at the end of the loop. *(The slide labels this "explicit"; elsewhere the deck consistently calls the end-of-work-sharing barrier **implicit** — see s31 and s34 — which is the accurate term. It is inserted automatically and can be suppressed with `nowait`, §5.9.)*

Two important notes:
- **`private(i)` is not needed** — the loop iterator is **private by default** in a `for` construct.
- **ACHTUNG [s32]: each loop iteration must be independent of the others.** If iteration *k* depends on iteration *k−1*, you cannot simply parallelize the loop.

Equivalent spellings [s33]: you may nest `#pragma omp for` inside a `#pragma omp parallel` block, or use the combined `#pragma omp parallel for` — they do the same thing. Use the combined form for a single loop; use the split form when one parallel region contains several work-shared loops.

That split form is shown with **two** consecutive `#pragma omp for` loops inside one `parallel` [s34]. The timeline is: **fork** (team created) → **distribute work** (loop 1) → **barrier** (synchronize) → **distribute work** (loop 2) → **barrier** → **join** (team destroyed). Each work-shared loop carries its own implicit barrier.

### 3.4 Exercise 1, once more [s35]

Switch from the plain `parallel` (redundant) version to `parallel for`; compile; run; optionally vary `OMP_NUM_THREADS`. Question: "Is the workload correctly split among the threads?" (Yes — that is the whole point of the `for` construct.)

---

## 4. Data clauses (the data environment) [s36–45]

### 4.1 Memory model [s37]

OpenMP's memory model has two kinds of memory:
- **Shared memory** — available to **all** threads.
- **Private memory** — available to **only one** thread.

The picture: a central "Shared Memory" pool, with each thread `T` also owning its own "Private Memory." **ACHTUNG [s37]: it is up to the programmer to choose the proper memory scope for each variable.** Wrong scoping is the usual source of data races and wrong answers.

### 4.2 `shared` [s38]

```cpp
int x{3};
omp_set_num_threads(8);
#pragma omp parallel shared(x)
{
    auto tid = omp_get_thread_num();
    x += 1;
    printf("Thread %d has x: %d.\n", tid, x);
}
```

`shared` variables have **one copy in shared memory, seen by all threads**. Here all 8 threads increment the *same* `x`, so the printed values are non-deterministic (e.g., `4, 5, 6, 7` in some interleaving) — this is a data race. **Note [s38]:** variables are **shared by default**, so the `shared(x)` clause is redundant in this example.

### 4.3 `private` [s39–40]

```cpp
int x{3};
omp_set_num_threads(3);
#pragma omp parallel for private(x)
for (size_t i = 0; i < 9; i++) {
    auto tid = omp_get_thread_num();
    x = i;
    printf("Thread %d has x: %d.\n", tid, x);
}
printf("Final x: %d.\n", x);
```

`private` variables are **uninitialized copies** of the global variable, each visible only to its own thread [s39]. Because the code assigns `x = i` before using it, each thread's copy is fine. Crucially, **`Final x: 3`** — the global `x` keeps its **pre-region** value; the private copies are discarded at the join.

The danger [s40]: if you *don't* initialize the private copy (e.g., `x += i` with `x = i` commented out), you read **garbage** (the slide shows values like `-608901888`). **ACHTUNG: be careful about initialization!** A `private` variable does **not** inherit the outside value.

### 4.4 `firstprivate` [s41]

```cpp
int x{3};
omp_set_num_threads(3);
#pragma omp parallel for firstprivate(x)
for (size_t i = 0; i < 3; i++) {
    auto tid = omp_get_thread_num();
    printf("Thread %d has x: %d.\n", tid, x);  // prints 3 first
    x = i;
    printf("Thread %d has x: %d.\n", tid, x);
}
printf("Final x: %d.\n", x);
```

`firstprivate` variables are per-thread copies that **are initialized to the global variable's value** (each thread starts with `x = 3`). Still, `Final x: 3` — the outside variable is untouched. **Best practice [s41]: prefer `firstprivate` to avoid unwanted uninitialized variables.**

### 4.5 `lastprivate` [s42]

```cpp
int x{3};
omp_set_num_threads(3);
#pragma omp parallel for lastprivate(x)
for (size_t i = 0; i < 3; i++) {
    x = i;
    printf("Thread %d has x: %d.\n", omp_get_thread_num(), x);
}
printf("Final x: %d.\n", x);
```

`lastprivate` variables are **uninitialized** per-thread copies (like `private`), **but** when the region ends the value from the **logically last iteration** is **copied back** to the global variable. Output: `Final x: 2` (the value at `i = 2`, the last iteration) — *not* the original 3.

### 4.6 Summary of data clauses [s43]

| Clause | Sharing | Initialization | Value seen *after* the region |
|---|---|---|---|
| `shared` | all threads | global value | last value written |
| `private` | one thread | **none** | pre-parallel-region value |
| `firstprivate` | one thread | global value | pre-parallel-region value |
| `lastprivate` | one thread | **none** | value of the **last iteration** |

**Best practice [s43]: prefer `firstprivate` over `private`** to avoid uninitialized-variable bugs.

**Default data-sharing policy [s43]:**
- **`shared`** if the variable is defined **outside** the parallel region;
- **`private`** if the variable is defined **inside** the parallel region, or if it is a **loop iterator** inside a `for` construct.

### 4.7 `default` [s44–45]

The `default` clause changes the default scoping behavior for the region.

```cpp
int x = 1, y = 1;
omp_set_num_threads(4);
#pragma omp parallel default(shared) private(x)   // y shared, x private
{ x += 1; y += 1; }
// Final x: 1   (x was private → outside value unchanged)
// Final y: 5   (y shared → 1 + 4·1)

#pragma omp parallel default(private) shared(x)   // x shared, y private
{ x += 2; y += 2; }
// Final x: 9   (x shared → 1 + 4·2)
// Final y: 5   (y private → outside value unchanged)
```

So `default(shared)` / `default(private)` set the fallback for any variable you don't scope explicitly [s44]. (Note the shared increments — `y += 1`, `x += 2` — are themselves latent data races; the slide shows the ideal totals.)

**`default(none)` [s45]:**

```cpp
int x = 1, y = 1;
omp_set_num_threads(4);
#pragma omp parallel default(none) shared(y) private(x)
{ x += 3; y += 3; }
// Final x: 1,  Final y: 13   (1 + 4·3)
```

**ACHTUNG [s45]:** with `default(none)` you are **forced** to specify the sharing policy for **every** variable used in the region; omit one and the **compiler errors out**. **Best practice [s45]: always use `default(none)`** and scope every variable explicitly — it turns silent scoping bugs into compile-time errors.

---

## 5. Synchronization constructs [s46–61]

### 5.1 The problem: relentless summation [s47–48]

Serial baseline [s47]:

```cpp
int sum{0}, n{10000};
for (size_t i = 1; i < n; i++)
    sum += i;
// Sum: 49995000 (correct),  Runtime ~4.1e-05 s
```

Naïvely adding `#pragma omp parallel for` [s48]:

```cpp
#pragma omp parallel for
for (size_t i = 1; i < n; i++)
    sum += i;
// Sum: 36949316  (WRONG — expected 49995000),  Runtime ~9.6e-05 s
```

The answer is **wrong** *and* slower. `sum += i` is a read-modify-write on a shared variable performed concurrently — a textbook data race. **ACHTUNG [s48]: data race — and here `private` is *not* the right fix** (a private `sum` would give each thread its own partial sum that never gets combined).

### 5.2 Data races, restated [s49]

A data race occurs whenever **two or more threads access the same memory location** (with at least one write). The cure is **synchronization** to enforce a legal ordering of operations. **ACHTUNG [s49]: it is up to the programmer to choose the correct synchronization method** — and, as the timings below show, the choice matters enormously for performance.

### 5.3 The toolbox [s50]

Two categories of solution:

1. **Mutual Exclusion (Mutex)** — define a block executed by only one thread at a time. Constructs: **`critical`**, **`atomic`**, **`barrier`**.
2. **Reduction** — perform a global operation combining a quantity from all threads. Operations: sum, subtraction, multiplication; logical `or`/`and`/…; maximum, minimum.

### 5.4 `critical` [s51–52]

`critical` prevents multiple threads from entering a code section at the same time [s51]. It has **no implicit barrier** at the end — it simply forces the threads through the section **one at a time**.
- **Pro:** prevents the data race.
- **Con:** the other threads **wait their turn, idle**.

```cpp
#pragma omp parallel for
for (size_t i = 1; i < n; i++) {
#pragma omp critical
    sum += i;
}
// Sum: 49995000 (correct),  Runtime ~0.000229 s
```

Correct now — but note the runtime jumped from ~4e-05 (serial) to ~2.3e-04 [s52]. Serializing every single update destroys the parallel benefit.

### 5.5 `atomic` [s53–54]

`atomic` guarantees **mutually exclusive access to one specific memory location** [s53]. It is more restricted than `critical` but usually cheaper because it can map to a single hardware atomic instruction.

```cpp
#pragma omp parallel for
for (size_t i = 1; i < n; i++) {
#pragma omp atomic
    sum += i;
}
// Sum: 49995000 (correct),  Runtime ~0.000147 s
```

Correct, and faster than `critical` (~1.5e-04 vs ~2.3e-04) — but still slower than serial.

The operations `atomic` supports [s54]:
- **Read:** `v = x;` → `#pragma omp atomic read`
- **Write:** `x = v;`
- **Update:** `x++`, `x--`, `-x`, `+=`, `*=`, … → `#pragma omp atomic update`
- **Capture:** `v = x++;` … → `#pragma omp atomic capture { temp = shared_var; shared_var += 5; }`

**Limitations [s54]:** `x` and `v` must be **scalar**; **no operator overloading**; **no complex expressions**. If you need more than a single scalar update, use `critical` (or, better, `reduction`).

### 5.6 `barrier` [s55–56]

`barrier` prevents any thread from continuing past a line until **all** threads have reached it [s55].
- **Pro:** prevents data races (by forcing a synchronization point).
- **Con:** threads that arrive early **wait idle**.
- **Best practice [s55]: use `barrier` as little as possible.** Too many barriers is a smell — reconsider the algorithm.

```cpp
#pragma omp parallel
{
    auto tid = omp_get_thread_num();
    printf("Thread %d here\n", tid);
#pragma omp barrier
    printf("Thread %d passed the barrier\n", tid);
}
```

Without the barrier [s56], "here" and "passed" interleave freely per thread (`0 here / 0 passed / 1 here / 1 passed`). With the barrier, **all** "here" lines print **before any** "passed" line — every thread must reach the barrier before any thread crosses it.

### 5.7 `reduction` [s57–58]

The right tool for the summation:

```cpp
#pragma omp parallel for reduction(+:sum)
for (size_t i = 1; i < n; i++)
    sum += i;
// Sum: 49995000 (correct),  Runtime ~7.2e-05 s
```

`reduction` **combines values into a single cumulative variable** — a convenient, efficient way to gather each thread's contribution [s57]. Under the hood each thread gets its **own private copy** of `sum`, initialized to the operator's identity element; the private partials are merged into the global `sum` at the end of the region [s58]. Correct **and** fast — close to serial, with no explicit locking.

**Syntax [s58]:** `reduction(operation:var)`. Supported operations: `+`, `-`, `*`, `.and.`, `.or.`, `.eqv.`, `.neqv.`, `.max`, `.min`, `.iand`, `.ior`, `.ieor` (the `.`-prefixed and bitwise ones come from the Fortran spelling; in C/C++ you use `&&`, `||`, `&`, `|`, `^`, `max`, `min`).

### 5.8 The performance verdict [s60–61]

**Exercise 2 [s60]:** compute π from the integral

$$\pi = \int_0^1 \frac{4}{1+x^2}\,dx \approx \sum_{i=0}^{N} \frac{4}{1+x_i^2}\,\Delta x_i$$

Parallelize the loop; fix the data race first with `critical`, then `atomic`, then `reduction`; time each; and check behavior against `OMP_NUM_THREADS`.

The measured runtimes [s61] are the punchline of the whole synchronization chapter:

| Implementation | Runtime (s) |
|---|---|
| Serial | 0.14 |
| `critical` | **10.60** |
| `atomic` | **8.34** |
| `reduction` | **0.04** |

`critical` and `atomic` are **~60–75× slower than serial** because they serialize every iteration; `reduction` is **~3.5× faster than serial** because it parallelizes without contention. **Best practice [s61]: use `reduction` whenever possible, especially in SPMD algorithms.**

### 5.9 `nowait` [s59]

The implicit barrier at the end of a work-sharing loop can be removed with `nowait`:

```cpp
// Without nowait: threads wait for the whole loop to finish before calling d()
#pragma omp parallel
{
#pragma omp for
    for (size_t i = 0; i < 10; i++) { c(); }
    d();
}

// With nowait: each thread calls d() as soon as it finishes its own iterations
#pragma omp parallel
{
#pragma omp for nowait
    for (size_t i = 0; i < 10; i++) { c(); }
    d();
}
```

Use `nowait` when the work after the loop does **not** depend on other threads having finished the loop — it removes an idle wait. Use it carefully: if `d()` reads results produced across the loop, removing the barrier reintroduces a race.

---

## 6. Explicit work-sharing constructs [s62–67]

Where `for` shares the iterations of *one* loop, these constructs assign *distinct blocks of code* to threads.

### 6.1 `sections` / `section` [s63]

```cpp
#pragma omp parallel sections
{
#pragma omp section
    printf("Hello\n");
#pragma omp section
    printf("Hi\n");
#pragma omp section
    printf("Bye\n");
}
```

`sections` creates **different tasks assigned to different threads**. Each `section` is an independent block run **concurrently** by a (first-available) thread. Use it when you have **multiple independent blocks of code** to run at the same time. **ACHTUNG [s63]:** variables defined in the surrounding parallel region are **not available inside a `section`** — scope your data accordingly.

### 6.2 `single` [s64–65]

```cpp
#pragma omp parallel
{
#pragma omp single
    printf("Hello\n");
}
```

`single` assigns a region to the **first available thread** — **exactly one** thread runs it (which thread is **arbitrary**), and **all other threads skip the block** [s64]. There is an **implicit barrier** at the end, so the others wait for the chosen thread to finish.

With several `single` blocks [s65], each is run once by some (possibly different) arbitrary thread; `single nowait` drops the trailing barrier so the others don't wait:

```cpp
#pragma omp parallel num_threads(4)
{
    auto tid = omp_get_thread_num();
#pragma omp single
    printf("Hello from thread %d\n", tid);   // once, some thread; others wait
#pragma omp single
    printf("Bye from thread %d\n", tid);     // once, some thread; others wait
#pragma omp single nowait
    printf("Bye bye from thread %d\n", tid); // once, some thread; NO waiting
}
```

Typical use: one-off work inside a parallel region — initialization, logging.

### 6.3 `master` [s66]

```cpp
#pragma omp parallel
{
#pragma omp master
    printf("Hello\n");
}
```

`master` assigns a region **exclusively to the master thread** (thread 0). Unlike `single`, there is **no implicit barrier** — the other threads do not wait. Use it for work that must be done specifically by the master (setup, printing).

### 6.4 Comparison [s67]

| Construct | Who executes it | Synchronization | Use case | Sees parallel-region vars? |
|---|---|---|---|---|
| `sections` / `section` | each section runs on a first-available thread | **implicit barrier** at end of `sections` block | independent tasks run concurrently by different threads | **No** |
| `single` | one thread, chosen **arbitrarily** | others **wait** (unless `nowait`) | one thread needs to run a block (init, logging) | **Yes** |
| `master` | only the **master** thread (thread 0) | **no barrier** | only the master needs to run a block (setup, printing) | **Yes** |

The two axes to remember: **who** runs the block (any-one vs. specifically-master) and **whether there is a barrier** (`single` yes, `master` no).

---

## 7. Loop optimizations [s68–76]

### 7.1 Nested loops and the nested-parallelism trap [s69]

Matrix multiplication is the standard example [s69]: $c_{ij} = \sum_{k=0}^{n} a_{ik} b_{kj}$, three nested loops over `i, j, k`. The tempting-but-wrong approach:

```cpp
#pragma omp parallel for
for (size_t i = 0; i < n; i++) {
#pragma omp parallel for      // <-- ignored!
    for (size_t j = 0; j < n; j++) { /* ... */ }
}
```

**ACHTUNG [s69]:**
- **Nested parallelism is disabled in OpenMP** (by default).
- The **second `pragma` is effectively ignored.**
- The inner region gets a **team of one thread**, so each inner loop runs single-threaded.
- You still pay **overhead** for creating that inner (useless) region.

So don't nest `parallel for`. Parallelize the outer loop only — or collapse.

### 7.2 `collapse(n)` [s70]

```cpp
#pragma omp parallel for collapse(2)
for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) { /* ... */ }
}
```

`collapse(n)` **merges `n` nested loops into a single iteration space** [s70], which is then shared among the threads. Each thread computes its assigned portion of the *combined* `i×j` space; threads synchronize and join as usual. This gives more total iterations to distribute (better load balance for small outer loops), *without* nested parallelism.

**Caveat [s70]:** `collapse` can introduce **additional scheduling overhead**, sometimes *increasing* runtime — **always time your program after collapsing.**

**Exercise 3 [s71]:** matrix addition `C[i*M+j] = A[i*M+j] + B[i*M+j]`. Parallelize; try **without** `collapse`, then **with** `collapse(2)`; vary `OMP_NUM_THREADS`. "Do you get the expected speed-up?"

### 7.3 Scheduling: how iterations are assigned to threads [s72–75]

This answers "how does OpenMP decide which thread gets which iterations?" The loop is split into **chunks**; the **schedule** decides how chunks are distributed.

**`static` [s72]** — the **default**.
```cpp
#pragma omp parallel for schedule(static)
for (i = 0; i < 100; i++) { /* ... */ }
```
- The loop is divided into **fixed** chunks **up front**.
- **Smallest overhead** (assignment decided once, at compile/entry time).
- Default chunk size divides the iterations **equally** (thread 0 → `0–25`, thread 1 → `25–50`, …).
- Specify a chunk size explicitly with `schedule(static, 25)`.

**`dynamic` [s73]**
```cpp
#pragma omp parallel for schedule(dynamic)
for (i = 0; i < 100; i++) { /* ... */ }
```
- Each thread **requests and executes a chunk**, then comes back for another, until no chunks remain.
- Good for **unbalanced workloads** (some iterations much heavier than others; some threads finish faster).
- **Default chunk size is 1** — which means high scheduling overhead.

**`guided` [s74]**
```cpp
#pragma omp parallel for schedule(guided)
for (i = 0; i < 100; i++) { /* ... */ }
```
- Like `dynamic`, but the **chunk size decreases over time** — it starts with **large** chunks and shrinks toward the default.
- Also for unbalanced workloads, but it **mitigates `dynamic`'s overhead** by handing out big chunks early and only fine-tuning near the end.

**Summary [s75]:**

| Schedule | Overhead | Default chunk | Use case |
|---|---|---|---|
| `static` | Small | equally divided among threads | small work imbalance |
| `dynamic` | Big | 1 | unbalanced workload |
| `guided` | Medium | 1 | unbalanced workload |

There is also **`auto`** [s75] — the compiler/runtime chooses the schedule for you.

**Best practice [s75]: use `static` for generic loops** (lowest overhead); switch to another schedule only when you *know* the loop/architecture behaves better with it.

**Exercise 4 [s76]:** SAXPY, `y[i] = scalar*x[i] + y[i]`. Parallelize; add work-sharing; **experiment with the `schedule` clause**; vary `OMP_NUM_THREADS`. Questions: "What is the best schedule for this work? Do you get the expected speed-up?" (SAXPY is perfectly balanced — every iteration costs the same — so `static` is the natural best choice, matching the s75 best practice.)

---

## 8. Likely exam angles

These are the discriminations the deck sets up repeatedly and is most likely to test:

1. **`critical` vs. `atomic` vs. `reduction`** — the central comparison. Know that all three fix the summation data race, but `critical` serializes an arbitrary block (slowest), `atomic` serializes a single scalar update (faster, but restricted to scalar read/write/update/capture, no complex expressions or overloaded operators), and `reduction` gives each thread a private partial that is merged at the end (fastest — the s61 table: 10.60 s / 8.34 s / 0.04 s vs. 0.14 s serial). If asked "which and why," the answer is almost always `reduction`.

2. **Spotting a data race** — recognize the four conditions [s24]: same location, concurrent, at least one write, no synchronization. Classic instances: a shared accumulator (`sum += i`) [s48], and a shared `tid` written by all threads [s25]. Know the fixes and *when each is wrong* (a `private` accumulator loses the partial sums — s48).

3. **Data clauses** — reproduce the s43 table from memory: `shared` (all threads, keeps last write), `private` (uninitialized, outside value unchanged), `firstprivate` (initialized from outside, outside value unchanged), `lastprivate` (uninitialized, copies the **last iteration** back out). Know the default rules (outside → shared, inside/loop iterator → private) and why `default(none)` is best practice (forces explicit scoping, turns bugs into compile errors).

4. **`parallel` vs. `parallel for`** — the redundant-execution trap [s27]: a bare `parallel` makes every thread run the whole loop; `for` splits the iterations. Remember the independence requirement: each iteration must not depend on another [s32].

5. **Choosing a schedule** — `static` (low overhead, balanced work, the default), `dynamic` (chunk 1, big overhead, imbalanced work), `guided` (shrinking chunks, medium overhead), `auto`. Be ready to justify `static` for balanced kernels like SAXPY and matrix addition.

6. **`single` vs. `master` vs. `sections`** — who runs the block and whether there's a barrier: `single` = one arbitrary thread + implicit barrier (removable with `nowait`); `master` = thread 0 only + **no** barrier; `sections` = independent blocks to different threads + implicit barrier, with no access to enclosing parallel-region variables [s63, s67].

7. **The thread-count hierarchy** [s21]: `num_threads` clause > `omp_set_num_threads()` > `OMP_NUM_THREADS`. Given a snippet with all three, state how many threads run (and note that `omp_set_num_threads` inside a region affects only *later* regions).

8. **Barriers, implicit and explicit** — implicit at the end of `parallel` [s18], of each work-sharing `for` [s31/s34], and of `single`/`sections`; explicit via `#pragma omp barrier` [s55]; removed via `nowait` [s59]. `master` and `critical` have **no** implicit barrier.

9. **Nested parallelism & `collapse`** — nested `parallel for` is disabled/ignored, gives a 1-thread inner team plus overhead [s69]; `collapse(n)` is the correct way to expose a merged iteration space, at some scheduling-overhead cost [s70].

10. **Timing** — `omp_get_wtime()` measures wall time (not per-thread, includes idle/I/O/sync), which is why it is the right benchmarking tool for parallel regions [s23]; contrast with CPU-time timers like `clock()`.

11. **The silent-flag gotcha** [s13] — without `-fopenmp` (or the compiler's equivalent), pragmas compile without error but run serially.

---

*Gaps / notes on the source deck:* This module is a threaded-CPU introduction and does **not** cover several topics named in general OpenMP syllabi:
- **No `task`/`taskwait` construct** (task-based parallelism) — the deck's "explicit work-sharing" is `sections`/`single`/`master` only.
- **No false sharing / cache-line contention** discussion.
- **No NUMA, thread affinity, or the `OMP_PROC_BIND` / `OMP_PLACES` environment variables.**
- **No `OMP_SCHEDULE` environment variable** — scheduling is shown only via the `schedule` clause.
- **No explicit Amdahl's-law treatment** — speedup is shown empirically through the runtime tables (s52, s53, s57, s61) rather than via the formula.
- SIMD/`simd` and GPU **offload** are mentioned as OpenMP capabilities [s11] but not taught here.
The `s32` slide labels the end-of-`for` barrier "explicit"; the deck's own s31/s34 (and OpenMP semantics) make clear it is the **implicit** work-sharing barrier — reconciled in §3.3 above.
