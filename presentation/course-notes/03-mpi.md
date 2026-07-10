# Module 3 — MPI (distributed-memory parallelism)

*Source: `OMP_MPI/MPI.pdf` — "MPI: Message Passing Interface", Alessandro Romeo (CINECA), CINECA/PoliMI training school. The slides are numbered 77–164 (this deck continues the OpenMP deck), so every reference below uses those printed slide numbers, e.g. [s101] or [s127–131].*

This chapter follows the deck's own order. The whole talk is organized around the roadmap on the summary slide [s78]: a **basic introduction**, then **communicators / groups / types of communication**, then **point-to-point** communication (blocking and non-blocking), then **how to avoid deadlocks**, then **collective communications**, and finally **collective performance**. Work through the sections in that order and you have the course.

---

## 1. What MPI is and why it exists [s77–86]

### 1.1 Definition and goals [s81–82]
MPI stands for **Message Passing Interface**. It is an **Application Programming Interface (API)**: the standard specifies *what each routine call looks like* and *how each routine must behave*, but deliberately **does not specify how a routine is implemented** [s81]. The deck quotes the standard: MPI "is a message-passing application programmer interface, together with protocol and semantic specifications for how its features must behave in any implementation." Although we will use C, MPI can be called from C/C++, Fortran, and other languages (Python, C#, Java, …).

MPI's three stated goals are **high performance**, **portability**, and **flexibility** [s82]. It supports both **point-to-point** and **collective** communication, and it has become the *de facto* standard for communication among processes in a program running on a **distributed-memory system**.

### 1.2 The distributed-memory model [s82–83]
This is the mental model to internalize. In MPI, **many instances of your program run in parallel** as separate processes ("tasks"). Each parallel process has its **own private memory space** [s83]. Because memory is not shared, **all data movement and synchronization is explicit** and happens through function calls: data is physically copied from the address space of one process into the address space of another [s83]. This works both *within* a shared-memory node (intranode) and *between* nodes (internode). Contrast this with OpenMP [s82], where a master thread simply forks into threads that share one memory — MPI never shares memory, it sends messages.

The deck's pros/cons table [s83] is worth memorizing:

| Pros | Cons |
|---|---|
| Communications are heavily optimized on HPC systems | Explicit message passing is **error-prone** |
| Highly **portable** | Serial programs usually need to be **completely rewritten** |
| Highly **scalable** | High memory overheads |
| Long history, large existing code base | Non-uniform access: remote data is slower than local data; data is scattered across separate address spaces |

The banner takeaway [s83]: **minimize message passing.** Excessive communication or synchronization degrades performance and should be avoided unless strictly necessary.

### 1.3 Standards and implementations [s84–86]
The **MPI Forum** defines the *standard* — the set of commands needed to run Fortran and C codes — and has released successive versions **MPI-1, MPI-2, MPI-3, MPI-4** (MPI-2 added explicit shared-memory programming; later versions keep extending it) [s84]. The standard is then realized by several **library implementations**: `intelmpi`, `openmpi`, `spectrum_mpi`, `mpich` [s84]. Compilation and execution are **implementation-dependent** [s84].

Practically, MPI ships as **libraries, header files and other options usable with standard compilers** (e.g. gcc). To hide those details, vendors provide a **wrapped compiler** and a **launcher** program [s85]. You access the wrappers/launchers by loading a compiler module; the common launchers are **`mpirun`, `mpiexec`, or `srun`** (the latter under SLURM). Be aware which underlying compiler the wrapper uses, since implementations like openmpi/intelmpi expose several [s85].

The main MPI languages are **Fortran and C** [s86]. The **C++ API was dropped in MPI-3** ("it offered no real advantage over the C bindings, being a simple wrapper layer"), so you *can* write C++ but you use the C API from it, and C++ support is deprecated [s86]. Python users get MPI through the **`mpi4py`** package [s86].

---

## 2. The MPI programming model [s87–91]

### 2.1 SPMD [s88]
The adopted model is **SPMD — Single Program, Multiple Data**. **Every task runs exactly the same program** [s88]. Many MPI tasks are launched at the start of execution; each has its **own local memory, completely separated** from the others, *unless* a **communication step transfers some data**. **Synchronization** may be needed to keep the parallel program correct. Because all ranks run the same binary, you use each process's rank (Section 4) inside `if` branches to make different ranks do different things.

### 2.2 The four families of MPI calls [s89]
There are hundreds of MPI calls; the course teaches only the important ones, grouped into four categories [s89]:
1. Calls to **initialize, manage, and terminate** communications.
2. Calls to **communicate between pairs** of processors — **point-to-point**.
3. Calls to **communicate among groups** of processors — **collective**.
4. Calls to create **data types and topologies**.

### 2.3 C vs Fortran bindings [s90]
- **C**: `#include "mpi.h"`. All MPI commands are **functions**; the **error code is the return value**; arrays are passed as `void*` pointers; commands are **case-sensitive**; the standard defines no C++ API.
- **Fortran**: `use` the MPI **module**. All MPI commands are **subroutines**; the **error code is returned as an extra final argument** (`ierr`) — easy to forget, and forgetting it causes hard-to-detect errors; arrays are passed by reference; commands are **case-insensitive**. The one exception is `MPI_Wtime()`, which is a function. (You may also see the older `#include "mpif.h"`, not recommended.)

### 2.4 General program structure [s91]
Every MPI program has the same skeleton [s91]: include MPI → (serial startup code) → **initialize the MPI environment** (parallel code begins) → **do work and message passing** → **terminate the MPI environment** (parallel code ends) → (serial code) → end. Call format: in C, `rc = MPI_Xxxxx(parameter, ...)` where `rc` is the error code, equal to `MPI_SUCCESS` on success. In Fortran, `CALL MPI_XXXXX(parameter, ..., ierr)` where `ierr` carries the code.

---

## 3. Getting started: Init, Finalize, compile, run [s92–95]

### 3.1 MPI_Init and MPI_Finalize [s92]
```c
int MPI_Init(int* argc_p, char*** argv_p);
int MPI_Finalize(void);
```
**`MPI_Init`** tells MPI to do all necessary setup — e.g. allocate message buffers and decide which process gets which rank [s92]. **`MPI_Finalize`** tells MPI you are done and any resources it allocated can be freed [s92]. All parallel/MPI work must sit between these two calls. **Exercise 1** is to take a plain "Hello world" and wrap it in Init/Finalize.

### 3.2 Compiling and running on Leonardo [s93–94]
On the **Leonardo Booster** (GPU) partition [s93]: `ml gcc/12.2.0 openmpi/4.1.6--gcc--12.2.0`, then compile with `mpif90` (Fortran), `mpicc` (C), or `mpic++`/`mpicxx` (C++), producing `<name>.exe`, and run **all** codes with `srun -n <number> ./<executable>`.
On the **Leonardo DCGP** (CPU) partition [s94]: `ml intel-oneapi-compilers intel-oneapi-mpi`, then compile with `mpiifx` (Fortran), `mpiicx` (C), `mpiicpx` (C++), and again run with `srun -n <number> ./<executable>`.

### 3.3 Hello MPI world [s95]
The three-language "Hello world" [s95] just does `MPI_Init` → print a greeting → `MPI_Finalize`. Its only lesson is the mandatory Init/Finalize bracket and the language-specific call syntax (C uses `err = MPI_Init(&argc, &argv)`; Fortran uses `call mpi_init(ierr)`). This is **Exercise 1**.

---

## 4. Communicators, ranks, and groups [s96–98]

### 4.1 Communicators and ranks [s96]
A **communicator** is a collection (group) of processes; **all tasks inside a communicator can communicate with each other**. The **default communicator is `MPI_COMM_WORLD`**, which contains **all** the tasks available to the program [s96]. **Every communication call takes a communicator argument.**

Inside a communicator each task has a unique ID called its **rank**, numbered **0 to n−1**. Ranks identify the **source and destination** of a communication [s96]. Two fundamental queries:
```c
int MPI_Comm_size(MPI_Comm comm, int *size);   // number of processes in comm
int MPI_Comm_rank(MPI_Comm comm, int *rank);   // this process's rank in comm
```
Important subtlety: **a rank is a logical task ID, not necessarily tied to a specific hardware core or processor** [s96].

### 4.2 Groups [s97]
The deck distinguishes the two [s97]:
- A **group** is an *ordered collection of processes*, each with a rank.
- A **communicator** *holds a group* of processes that can communicate.

Groups let you run collective operations on a **subset** of processes. The typical workflow is: get the group underlying `MPI_COMM_WORLD` (via `MPI_COMM_GROUP`), build a new group from a subset of ranks (`MPI_GROUP_INCL`), then create a new communicator from that group (`MPI_COMM_CREATE`) [s97]. Group routines exist mainly to *specify which processes will build a new communicator*. Any process can be killed with:
```c
int MPI_Abort(MPI_Comm comm, int errorcode);
```

### 4.3 First real example and non-determinism [s98]
The example [s98] calls `MPI_Comm_size` and `MPI_Comm_rank` and prints "Hello I am `<rank>` of `<nprocs>`". The lesson is highlighted: **"Non-deterministic competition among tasks!"** — because all ranks run concurrently and write to the same output, the order of the printed lines is not fixed from run to run. This is **Exercise 2**.

---

## 5. Two kinds of communication [s99]
MPI offers two communication styles [s99]:
- **Point-to-point**: one process sends a message to *one* other process, naming that process's **rank** and a unique **tag** to identify the message. The receiver posts a matching receive (with or without a specific tag) and handles the data.
- **Collective**: a whole group communicates together (e.g. a manager process broadcasts to all workers). A program that only ever does individual sends/receives cannot use the network optimally — collectives exist to do these group patterns efficiently.

---

## 6. Point-to-point communication [s100–113]

### 6.1 The basic Send/Recv [s101]
Point-to-point is the **basic communication method**: communication **between two processes**, a source **A** and a destination **B**, taking place **inside a communicator**, with source and destination identified by their **ranks** [s101].
```c
int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
             int dest, int tag, MPI_Comm comm);
int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
             int source, int tag, MPI_Comm comm, MPI_Status *status);
```

### 6.2 The message: envelope + body [s102]
Every message has an **envelope** and a **body** [s102]:
- **Envelope** = `source`, `destination`, `communicator`, `tag`.
- **Body** = `buffer`, `count`, `datatype`.

A message is exchanged **only if sender and receiver specify the correct (matching) envelope** [s102]. Passing the datatype explicitly (rather than raw bytes) is what lets MPI programs run correctly in **heterogeneous** environments [s102].

### 6.3 Master–worker example [s103]
The C/Fortran example [s103] checks the rank: `if (myid == 0)` do `MPI_Send`, `else if (myid == 1)` do `MPI_Recv`. This introduces the **master–worker programming model**: **rank 0 is the master process** and usually handles I/O [s103].

### 6.4 Datatypes [s104]
MPI datatypes come in two kinds [s104]:
- **Basic / elementary** types, for portability: `MPI_INT`, `MPI_FLOAT`, `MPI_DOUBLE`, `MPI_CHAR`, `MPI_REAL`, `MPI_BYTE`, `MPI_PACKED`, etc. Their names mirror the C/Fortran type with an `MPI_` prefix (C's `int` → `MPI_INT`).
- **Derived types**, built with the `MPI_Type_xxx` functions (Section 6.5).

MPI uses **handles** to refer to types/structures that differ between C and Fortran [s104]: in C/C++ handles are **macros for structs** (`#define MPI_INT …`); in Fortran they are `INTEGER` (except in MPI-3 / F2008, where they become Fortran derived types).

### 6.5 Derived datatypes (intro) [s105]
MPI communication normally requires that the multiple data items be **of the same type and contiguous in memory** [s105]. When your message contains **non-contiguous data of one type**, or **contiguous data of different types**, you have four options [s105]:
1. Make **multiple MPI calls**, one per element.
2. Use **`MPI_BYTE`** to sidestep type matching (treat everything as raw bytes).
3. Use **`MPI_PACKED`**: the user *explicitly packs* data into a contiguous buffer before sending and *unpacks* it after receiving.
4. Use a **derived datatype**: user-defined, built up from basic types; lets MPI automatically gather/scatter data to and from non-contiguous buffers, avoiding manual pack/unpack.

### 6.6 MPI_Status and MPI_Get_count [s106]
`MPI_Status` is a struct with at least three members: **`MPI_SOURCE`, `MPI_TAG`, `MPI_ERROR`** [s106]. After a receive you can read `status.MPI_SOURCE` and `status.MPI_TAG` to learn who actually sent the message and with what tag (useful with wildcards, Section 6.9). The **amount of data received is not stored in a directly accessible field**; retrieve it with:
```c
int MPI_Get_count(MPI_Status *status, MPI_Datatype datatype, int *count);
```

### 6.7 Five conditions for a communication to succeed [s107]
For a point-to-point communication to succeed [s107]:
1. The **sender must specify a valid destination rank.**
2. The **receiver must specify a valid source rank.**
3. The **communicator must be the same** on both sides.
4. The **tags must match.**
5. The receiver's **buffer must be large enough.**

Warning: check every argument carefully — **the call may succeed but with wrong data** [s107].

### 6.8 Blocking communication and its semantics [s108–110]
Once a message is assembled, the sending process has two possibilities: it can **buffer** the message, or it can **block** [s108]. Crucially, **unlike `MPI_Send`, `MPI_Recv` always blocks until a matching message has been received** — so when `MPI_Recv` returns you know the data is in the receive buffer (barring error) [s108].

**Completion** means both memory locations used in the transfer can be **safely accessed** [s109]:
- A **blocking send** returns once it is **safe to modify the application (send) buffer** for reuse. *Safe does **not** imply the data was actually received* — it may still be sitting in a **system buffer** [s109].
- A blocking send can therefore be **synchronous** (the program does not continue until the data has been received) or **asynchronous** (a system buffer holds the data for eventual delivery) [s109].
- A **blocking receive** only returns after the **data has arrived and is ready to use** [s109].

How buffers are actually used is **implementation-specific** — MPI defines only the *semantics* of blocking communication (buffered sends excepted), not the buffering mechanism; the user does not know the message's "path" as long as it fits in the buffer [s110]. The flow diagram [s110]: Task 0 posts `MPI_Send` → waits until data is safely sent → reuses data → continues; Task 1 posts `MPI_Recv` → waits until data is available → uses received data → continues.

### 6.9 Blocking example and wildcards [s111–112]
The example [s111–112]: rank 0 sets `number = 0` and sends it to rank 1; rank 1 sets `number = 1`, then receives into `number` (so its value becomes 0 after the receive). The `printf` output ordering across ranks is again **non-deterministic** [s112]. Key rule introduced here — **wildcards are accepted only on the receiver side** [s111]:
- **`MPI_ANY_SOURCE`** — receive from any source.
- **`MPI_ANY_TAG`** — receive with any tag.

The actual source and tag are then reported back in the receive's **`status`** parameter. This is **Exercise 3.0**.

### 6.10 Interlude: tree-structured communication [s113]
A motivating aside about *why collectives matter* [s113]. Consider summing values held by 8 processes onto process 0. The naïve scheme needs `comm_sz − 1 = 7` receives and 7 adds *all done by process 0*. A **tree-structured** scheme needs only **3** rounds on process 0, and no other process does more than two receives/adds — giving **more concurrent work**. For 1024 tasks the difference is 1023 vs ~10 steps. The rhetorical question — **"Who decides the best strategy?"** — sets up the answer: let the **MPI library** pick the algorithm by using collective calls instead of hand-rolled loops.

---

## 7. Non-blocking communication [s114–124]

### 7.1 Concept [s114]
**Non-blocking** send/receive routines **return almost immediately** and do **not** wait for the communication to complete [s114]. They simply **"request"** that the MPI library perform the operation when possible — you cannot predict when that will be. Consequences [s114]:
- It is **unsafe to modify the application buffer** until you *know* the requested operation actually happened; **"wait" routines** exist to establish this.
- Non-blocking communication is used primarily to **overlap computation with communication**.
- Most point-to-point routines exist in both blocking and non-blocking form, and **the programmer must insert code to check for completion.**

### 7.2 Blocking vs non-blocking timeline [s115]
Side by side [s115]: with **blocking** `MPI_RECV`, execution **STOPs** at the receive, then does calculations, then more, then works on new data. With **non-blocking** `MPI_IRECV`, you *post the data request*, then immediately do calculations and more calculations *while the transfer proceeds*, and only call **`MPI_Wait`** (mandatory) right before you actually need the received data.

### 7.3 The core non-blocking calls [s116–117]
```c
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm, MPI_Request *req);
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm, MPI_Request *req);
```
Each returns a **request handle** (`req`) identifying the in-flight operation [s116]. Completion is forced with:
```c
int MPI_Wait(MPI_Request *req, MPI_Status *status);
int MPI_Waitall(int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses);
```
`MPI_Wait` **blocks until** the operation named by `req` completes; `MPI_Waitall` waits for `count` non-blocking operations at once [s116]. To *poll* instead of block [s117]:
```c
int MPI_Test(MPI_Request *req, int *flag, MPI_Status *status);
int MPI_Testall(int count, MPI_Request *array_of_requests, int *flag, MPI_Status *array_of_statuses);
```
`MPI_Test` **checks** whether an operation is complete **without waiting**; it sets the output **`flag`** to true if complete, false otherwise. You can then use an `if` on `flag` to decide whether to do other work while the operation is still pending [s117].

### 7.4 Worked examples 4.1 and 4.2 [s118–123]
**Exercise 4.1** [s118–120]: each of ranks 0 and 1 posts an `MPI_Irecv` *first*, then does its `MPI_Send`, then `MPI_Wait`s. Posting the receive before the send is what lets the two-way exchange proceed without blocking. The successive slides show that the printed results depend on whether the `MPI_Wait` has completed before the values are read — the final correct version [s120] has `MPI_Wait` in both branches so both `a` and `b` hold the exchanged data.

**Exercise 4.2** [s121–123]: uses `MPI_Isend` + `MPI_Irecv` with **request arrays** (`req0[2]`, `req1[2]`) and a single `MPI_Waitall(2, …)` per rank. The instructive contrast is *when* you touch the receive buffer:
- If `b[0] += 10.0` is executed **before** `MPI_Waitall` [s122], the data has **not arrived yet**, so you modify a stale/undefined value — wrong result.
- If `b[0] += 10.0` is executed **after** `MPI_Waitall` [s123], `b` already holds the received value (3.0), so the output correctly shows **13.0**.

The lesson: **never read or write a non-blocking buffer until the matching wait/test confirms completion.**

### 7.5 Why bother: overlap [s124]
A major inefficiency of blocking communication is the **waiting time between sending and receiving** [s124]. With non-blocking calls you can **overlap communication with calculations that do not depend on the message**, cutting the idle time: post `isend(a)`/`irecv(a)`, do independent work, and only `mpi_wait` when `a` is finally needed.

---

## 8. Deadlock [s125–134]

### 8.1 What a deadlock is [s126]
A **deadlock** (or race condition) occurs when **two or more processes are blocked, each waiting for the other to make progress** [s126]. Diagram: process 0 will proceed only after process 1 takes action B, while process 1 will proceed only after process 0 takes action A — neither can move.

### 8.2 The exercise series 5.1–5.5 — analyze each [s127–131]
These slides all ask "**Deadlock?**". The tell in the deck is simple: **the deadlocking cases show no program output; the working cases show terminal output.** Learn to reason through them:

- **5.1 [s127] — DEADLOCK.** Both ranks call **`MPI_Recv` first, then `MPI_Send`**. Each process blocks in its receive waiting for a message the other has not sent yet (because the other is also stuck in its own receive). Classic *incorrect ordering*.
- **5.2 [s128] — DEADLOCK.** Both ranks call `MPI_Send` then `MPI_Recv`, but the **tags are misaligned**: each sends with **tag 10** yet receives with **tag 11**. No receive can ever match the messages in flight, so the receives hang regardless of buffering. This is the *misaligned tags* failure.
- **5.3 [s129] — DEADLOCK.** Both ranks call **`MPI_Send` first** (tags matched this time) but on a **large array** (`count = 10000`). The message is too big to be buffered, so each `MPI_Send` blocks waiting for a matching receive that the other process has not reached — the *send-first with no buffering* failure. (Note this same pattern can appear to "work" for small messages when the implementation buffers them — which is exactly why relying on it is unsafe.)
- **5.4 [s130] — WORKS.** Rank 0 does **`MPI_Recv` then `MPI_Send`** while rank 1 does **`MPI_Send` then `MPI_Recv`**. Rank 1's send matches rank 0's receive, then rank 0's send matches rank 1's receive. Output is printed.
- **5.5 [s131] — WORKS.** The mirror image: rank 0 **`MPI_Send` then `MPI_Recv`**, rank 1 **`MPI_Recv` then `MPI_Send`**. Correct ordering, output printed.

The unifying rule: for a paired exchange, **one side's send must be matched by the other side's receive being posted in a compatible order**, with matching tags, and you must not rely on the implementation buffering a blocking send.

### 8.3 Causes and fixes; MPI_Sendrecv [s132]
**Reasons for deadlock in MPI** [s132]: incorrect **ordering** of `MPI_Send`/`MPI_Recv`; **misaligned tags**; other misaligned sends/receives (e.g. a rank not actually in the communicator). **Deadlock can be avoided** by: checking that sends/receives and tags line up; using **non-blocking** send/receive; or using **`MPI_Sendrecv`**, which simultaneously posts a send and a receive in one safe call:
```c
int MPI_Sendrecv(void *snd_buf, int snd_count, MPI_Datatype snd_type, int dest, int tag,
                 void *rcv_buf, int rcv_count, MPI_Datatype rcv_type, int src, int tag,
                 MPI_Comm comm, MPI_Status *status);
```
Three ranks are involved: the calling process, the source, and the destination [s132].

### 8.4 "The final level boss": ring exchange [s133–134]
**Exercise 6.1** [s133–134]: every task declares arrays `A` and `B`; every element of `A` is initialized to the process's own rank. `A` is the send buffer, `B` the receive buffer, and the processes form a **ring**. Each process computes its neighbors:
```c
right = (myid + 1) % numprocs;
left  = myid - 1;
if (left < 0) left = numprocs - 1;   // wrap-around: "What happens if left is < 0?"
```
Then a single `MPI_Sendrecv(A, 1, MPI_INT, right, 123, B, 1, MPI_INT, left, 123, MPI_COMM_WORLD, &status)` sends `A` to the right neighbour and receives into `B` from the left neighbour [s134]. Because send+receive happen in one call, the ring (including the wrap-around from the last process back to the first) completes with **no deadlock** and works for **any number of tasks**. The output confirms each task received its left neighbour's rank (task 0 receives 3, task 1 receives 0, task 2 receives 1, task 3 receives 2).

---

## 9. Collective communication [s135–154]

### 9.1 Semantics and warnings [s136]
Communications involving **groups of processes** are **collectives** [s136]. Rules:
- Calls occur between processes in the **same communicator**, and **every process must call the collective function.**
- Collectives **do not interfere** with point-to-point calls.
- **No tags** are required.
- **Receive buffers must match in size.**
- In **MPI 1.0–2.0 all collectives are blocking**; **MPI-3 introduced non-blocking collectives**, designed to replace loops of point-to-point calls and to be both more concise and more efficient.

Big warning [s136]: **MPI does not define behavior when a process fails to take part in a collective call** — possible outcomes are a crash, a deadlock, or wrong results. So *all* ranks in the communicator must reach *every* collective.

### 9.2 Barrier [s137]
```c
int MPI_Barrier(MPI_Comm comm);
```
`MPI_Barrier` **stops all processes until they are all synchronized** — useful to make sure every rank has reached the same point [s137]. Typical uses: reading data from file and distributing it, synchronizing data across tasks, computing a value from all tasks, synchronizing tasks. Warning: **severe performance impact if used too often** [s137].

### 9.3 Manual broadcast (motivation) [s138]
Before showing `MPI_Bcast`, the deck shows the manual version [s138]: rank 0 sets `a`, then **loops over all other ranks doing `MPI_Send`**, while every other rank does one `MPI_Recv`. It works but is verbose and inefficient — motivating the real collective.

### 9.4 Broadcast [s139–140]
```c
int MPI_Bcast(void *buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
```
**One process (the `root`) sends the same message to all other ranks in the communicator** [s139]. **All processes must specify the same `root` and communicator.** The canonical use case is **user input**: rank 0 reads input from disk, then broadcasts it to everyone. The example [s140] replaces the whole send-loop of s138 with a single `MPI_Bcast(a, 2, MPI_FLOAT, 0, MPI_COMM_WORLD)`. This is **Exercise 7**.

### 9.5 Gather [s141–142]
```c
int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm);
```
**The root process collects data elements from all processes and stores them in rank order** [s141]. Critical note: **`recvcount` is the size of the single block expected from each process, NOT the size of the final assembled array** [s141]; the blocks are stored contiguously. The example [s142] is a `Print_vector` routine: only the root allocates the full buffer `b`, calls `MPI_Gather`, and prints the assembled vector; non-root ranks just contribute their `local_b`.

### 9.6 Scatter [s143–144]
```c
int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm);
```
The inverse of gather: **the root's message is split into `n` equal segments, and the `i`-th segment goes to the `i`-th process** [s143]. Same caveat: **`sendcount` is the size of the single block sent to each process, NOT the full array** [s143]. The example [s144] is a `Read_vector` routine: the root allocates and reads the full vector, then `MPI_Scatter`s equal chunks into each rank's `local_a`.

### 9.7 Allgather [s145–147]
```c
int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
```
**Allgather = a gather followed by a broadcast**: every process ends up with the full assembled array [s145]. The motivating application [s146–147] is **matrix–vector multiplication** `y = A·x`. Computing `y[i]` needs the entire `i`-th row of `A` **and all components of `x`**. If row `i` of `A` lives on process `q`, then `y[i]` also lives on `q`; but each process only holds its slice of `x`. **Allgather is how every process obtains all of `x` before the inner loop**: the code [s147] does `MPI_Allgather(local_x, local_n, MPI_DOUBLE, x, local_n, …, comm)` to build the full `x`, then runs the local `local_m × n` multiply. *(The slide's signature also shows an `int root` argument and the code has a typo `MPIDOUBLE`; the all-* collectives do not take a root — cross-check with the standard, s79.)*

### 9.8 All-to-all [s148]
```c
int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
```
**All-to-all redistributes each process's content so that every process ends up knowing a piece of every other's buffer** (like transposing a matrix of blocks) [s148]. It is useful (e.g. matrix transpositions) but is an **expensive call** and should be used only when strictly needed. *(Again the slide lists a spurious `int root`.)*

### 9.9 Variable-size variants: Scatterv / Gatherv [s149]
MPI offers more complex distribution calls; in particular you can **customize the length of the arrays** each process sends/receives [s149]:
```c
int MPI_Scatterv(void* sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype,
                 void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
int MPI_Gatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                void* recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm);
```
The `*counts` arrays give per-process block sizes and `displs` gives per-process offsets, so blocks need not be equal (unlike plain Scatter/Gather). This is **Exercise 8**.

### 9.10 Reduction: Reduce and Allreduce [s150–154]
```c
int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype type,
               MPI_Op op, int root, MPI_Comm comm);
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype type,
                  MPI_Op op, MPI_Comm comm);
```
A **reduction takes values from many processes and combines them into a single value** (sum, average, min, …), using special MPI calls to avoid race conditions [s150]. It **collects** data from all processes, **reduces** them with an operation, and either **stores the result on a single process (`MPI_Reduce`)** or **distributes the result to all processes (`MPI_Allreduce`)** [s150].

The built-in operations `MPI_Op` [s151]: `MPI_MAX`, `MPI_MIN`, `MPI_SUM`, `MPI_PROD`, `MPI_LAND` (logical AND), `MPI_BAND` (bitwise AND), `MPI_LOR`, `MPI_BOR`, `MPI_LXOR`, `MPI_BXOR`, `MPI_MAXLOC` (max **and its location**), `MPI_MINLOC` (min and its location).

The example [s152] sets `a[0] = my_rank`, `a[1] = 2*my_rank`, then `MPI_Reduce(a, res, 2, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD)`; only rank 0 prints the summed `res`. This is **Exercise 9**.

Two correctness points on reductions:
- **Collectives use no tags, so they are matched only by communicator and by the order in which they are called** [s153]. If different processes issue their `MPI_Reduce` calls in different orders, they mismatch.
- **`MPI_Reduce(&x, &x, 1, …)` is illegal** — using the same buffer as both send and receive is **aliasing of an output argument**; the result is unpredictable (wrong answer, crash, or accidentally correct) [s153].

**All-reduction** [s154]: `MPI_Allreduce` is used when **all** processes need the result of, say, a global sum to continue a larger computation. There is no destination/root because everyone gets the result; implementations often use a **butterfly** communication pattern to distribute it efficiently.

---

## 10. Performance: timing and scaling [s155–163]

### 10.1 Measuring time: MPI_Wtime / MPI_Wtick [s156]
```c
double MPI_Wtime(void);   // elapsed wall-clock seconds since some point in the past
double MPI_Wtick(void);   // resolution of MPI_Wtime, in seconds
```
`MPI_Wtime` returns **wall-clock seconds** as a floating-point number; the reference "time in the past" does not change during the process's life [s156]. It is **portable** (returns seconds, not ticks), high-resolution, and **local to the node that called it** — different nodes are not guaranteed to agree on absolute time. `MPI_Wtick` returns the clock resolution (e.g. `1e-3` if the hardware counter ticks every millisecond). **Neither returns an error code**, and calling them before `MPI_Init` or after `MPI_Finalize` is undefined [s156].

### 10.2 Timing patterns [s157–158]
The basic pattern [s157]: `start = MPI_Wtime()` → timed code → `finish = MPI_Wtime()` → print `finish - start`. `MPI_Wtime` measures wall-clock time **including idle time**. But a parallel run produces **`nprocs` separate times, one per process** — and we usually want a *single* number, namely **the time of the slowest process** (all would ideally start together and we report when the last one finishes). We cannot force identical start instants, but we can get close with `MPI_Barrier` [s157].

The robust pattern [s158]:
```c
MPI_Barrier(comm);                 // line everyone up first
local_start = MPI_Wtime();
/* code to be timed */
local_finish = MPI_Wtime();
local_elapsed = local_finish - local_start;
MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);  // take the slowest
if (my_rank == 0) printf("Elapsed time = %e seconds\n", elapsed);
```
Barrier to synchronize the start, then `MPI_Reduce` with **`MPI_MAX`** to get the slowest rank's elapsed time [s158]. Because OS/interaction noise never makes a program faster than on a "perfect quiet" system, one often reports the **minimum run-time over several runs** [s158].

### 10.3 The scaling model [s159–161]
Using the matrix–vector code as the running example [s159]: the serial cost is `t_serial(n) ∝ n²` (the outer loop runs `m` times, the inner does `n` multiplications + `n` additions). The parallel time is modeled as
> **t_parallel(n, p) = t_serial(n) / p + t_overhead/comm**

where in MPI the **overhead typically comes from communication** and depends on both problem size `n` and process count `p` [s159]. Splitting the matrix rows across `p` processes reduces the work per process by a factor of `p`, i.e. **`n²/p` operations each** [s160]. The two regimes [s160]:
- **Small `p`, large `n`** → the `t_serial/p` term dominates (you are compute-bound; parallelism pays off).
- **Small `n`, large `p`** → the `t_overhead/comm` term dominates (here due to the `Allgather`) — communication swamps the useful work.

Standard derived metrics [s161]:
> **Speed-up** `S(p) = t_serial / t_parallel(n, p)`  and  **Efficiency** `E(p) = S(p) / p = (t_serial · n_s) / (t_parallel(n,p) · p)`.

The measured tables confirm the model: **for small `p` and large `n`, efficiency is nearly linear (≈1.0); for large `p` and small `n` it is far from linear** (e.g. efficiency collapses to ~0.15 at `p = 16`, `n = 1024`, but stays ~0.97 at `n = 16384`) [s161]. The run-time table even shows run-times getting *worse* going from 8 to 16 processes at order 1024 [s159].

### 10.4 Collectives dominate scaling [s162]
Vendors work hard to optimize collectives, but **parallel scaling is often dictated by the MPI collectives** anyway [s162]. **All-to-all-type communications are especially time-consuming at high message sizes** — the `MPI_Alltoall` timing curve rises sharply as messages grow, and an Intel APS profile of GROMACS on 128 nodes shows MPI calls (e.g. `MPI_Bcast`, `MPI_Comm_split`, `MPI_Sendrecv`, `MPI_Gatherv`, `MPI_Scatter`) making up the bulk of runtime [s162].

### 10.5 Performance tips (including hybrid MPI/OpenMP) [s163]
Practical advice to reduce collective cost [s163]:
- **Avoid unnecessary `MPI_Barrier`s** (often left over from debugging).
- Consider **non-blocking collectives.**
- Define **communicators over a subset** of processes (fewer tasks per collective).
- Introduce **algorithms that avoid many-process collectives** — e.g. **domain decomposition** / meshes — and prefer **local communication** (point-to-point such as `MPI_SendRecv`).
- Use **hybrid MPI/OpenMP** to reduce the number of MPI processes taking part in collective calls.
- **Do not spend too much time in communication rather than computation.**

The **domain decomposition** picture [s163] captures the whole philosophy: partition the problem spatially so each process talks mostly to its neighbours via point-to-point calls, keeping communication **localized** instead of global.

---

## Likely exam angles

- **Deadlock diagnosis (the heart of the deck, s127–134).** Given a two-process send/receive snippet, decide whether it deadlocks and why. Master the three causes: (1) both sides **receive-first** → always deadlock; (2) **misaligned tags** (send tag ≠ recv tag) → receives never match; (3) both sides **send-first with a large message** → the blocking send cannot buffer and stalls. Know the three fixes: correct ordering, non-blocking calls, or `MPI_Sendrecv`. Be ready to rewrite a deadlocking exchange with `MPI_Sendrecv` and handle the ring wrap-around (`left = numprocs - 1`).
- **Blocking semantics (s108–110).** Explain that a blocking `MPI_Send` returning means only that the send buffer is safe to reuse — **not** that the message was received (it may be in a system buffer); `MPI_Recv` always blocks until the data has actually arrived. Distinguish synchronous vs asynchronous sends and remember buffering is implementation-defined.
- **Non-blocking correctness (s114–124).** Why you must not touch a buffer between `MPI_Isend`/`MPI_Irecv` and the matching `MPI_Wait`/`MPI_Waitall` (the s122 vs s123 "13.0" example). Difference between `MPI_Wait` (blocks) and `MPI_Test` (polls, sets `flag`). The purpose: overlap communication with independent computation.
- **Choosing the right collective (s135–154).** Map a scenario to a call: distribute input from rank 0 → `MPI_Bcast`; hand out equal chunks → `MPI_Scatter`; collect chunks in rank order → `MPI_Gather`; everyone needs the whole assembled array → `MPI_Allgather`; global sum on root → `MPI_Reduce`; global sum on everyone → `MPI_Allreduce`; unequal chunks → `MPI_Scatterv`/`MPI_Gatherv`; full redistribution/transpose → `MPI_Alltoall` (expensive). Remember the `send/recv count` is per-block, not the whole array.
- **Collective rules and traps (s136, s153).** Every rank in the communicator must call the collective (else crash/deadlock/wrong result); collectives use **no tags** and are matched by communicator + call order; `MPI_Reduce(&x, &x, …)` is **illegal aliasing**.
- **Communicators/ranks/groups (s96–98).** `MPI_Comm_size` vs `MPI_Comm_rank`; ranks 0…n−1; `MPI_COMM_WORLD`; why printed output is non-deterministic; group→communicator construction (`MPI_COMM_GROUP` → `MPI_GROUP_INCL` → `MPI_COMM_CREATE`).
- **Datatypes (s104–105).** Basic vs derived; four ways to send non-contiguous/mixed data (multiple calls, `MPI_BYTE`, `MPI_PACKED`, derived types); why explicit datatypes give heterogeneous portability.
- **Timing and scaling (s156–161).** The `MPI_Barrier` + `MPI_Wtime` + `MPI_Reduce(MPI_MAX)` idiom and why you take the slowest rank; the model `t_parallel = t_serial/p + t_overhead`; definitions of speed-up and efficiency; the "large p, small n → communication-bound, efficiency far from linear" regime.
- **Conceptual foundations (s81–90).** MPI as an API (specifies behavior, not implementation); distributed memory with explicit, private-address-space communication; SPMD; and the C-vs-Fortran binding differences (return code vs `ierr` argument, case sensitivity).

---

*Fidelity notes: (a) slide references use the numbers printed on the slides (77–164). (b) A few signatures on the slides contain typos that contradict real MPI — the all-collectives `MPI_Allgather`/`MPI_Alltoall`/`MPI_Allreduce` are shown with a spurious `int root` argument, and code on s147/s160 writes `MPIDOUBLE` for `MPI_DOUBLE`; the deck itself points to the current MPI-4.1 standard (s79) as the authority. (c) On s133 the prose says each process "sends to its left" while the code and output (s134) send to `right` and receive from `left`; I described the mechanics the code actually implements. No page ranges failed to load; the full deck (s77–164) was read.*
