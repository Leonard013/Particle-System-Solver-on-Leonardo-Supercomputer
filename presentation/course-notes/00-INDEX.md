# Course notes — index

Complete reconstruction of the course from its lecture decks (every chapter cites
slide numbers, so any claim can be checked against the PDFs). Written for someone
who did not attend. Read in order; then read `08-course-to-project-map.md`, which
walks the Particles project through the course lens.

| Chapter | Source deck(s) | Content |
|---|---|---|
| [01 — Intro to HPC & Leonardo](01-intro-hpc-leonardo.md) | `Introduction/IntroToHPC.pdf`, `IntroToLeonardo.pdf` | Why parallelism (Moore/Dennard, power wall), architectures (Flynn, NUMA, networks), SPMD, scaling laws (speedup/efficiency/Amdahl, strong vs weak), Leonardo (Booster/DCGP, storage, modules, SLURM) |
| [02 — OpenMP](02-openmp.md) | `OMP_MPI/OMP.pdf` | Threads vs processes, fork-join, parallel/for, data clauses, race conditions, critical vs atomic vs reduction (with the 10.6s/8.3s/0.04s benchmark), schedules, collapse, single/master/sections |
| [03 — MPI](03-mpi.md) | `OMP_MPI/MPI.pdf` | Distributed memory, ranks/communicators, Send/Recv semantics, non-blocking + Wait/Test, deadlock taxonomy + Sendrecv, all collectives (Bcast/Scatter/Gather/Allgather(v)/Reduce/Allreduce/Alltoall), timing idiom, scaling model |
| [04 — CUDA](04-cuda.md) | `GPUcomputing/Accelerating_Scientific_Applications_on_NVIDIA_GPUs.pdf` | Latency vs throughput design, SM/warp/SIMT, occupancy & latency hiding, indexing math, grid-stride, memory hierarchy, unified/pinned memory, prefetch, arithmetic intensity + roofline, tiling, streams |
| [05 — OpenACC](05-openacc.md) | `GPUcomputing/openacc/03-Shukla-OpenAcc.pdf` | Directive-based offload, kernels vs parallel, data clauses/regions (structured + unstructured), gang/worker/vector, the Laplace case study (naive GPU slower than serial → data region → 68×), OpenMP-offload dictionary |
| [06 — Profiling I](06-profiling-cpu-mpi.md) | `Profiling/slides/introduction.pdf`, `mpi.pdf` | Profiling vs tracing, sampling vs instrumentation, timers (wall vs CPU), gprof, PAPI, PMPI, roofline (CPU view), the POP efficiency model (load balance / transfer / serialization), Scalasca/Score-P workflow, VTune |
| [07 — Profiling II](07-profiling-gpu-roofline.md) | `Profiling/slides/gpu_profiling.pdf`, `GPUPythonProfiling/slides/GPURooflineModelForHPC.pdf` | Anatomy of an offloaded region, nsys (timeline, NVTX, stats), async/streams, MPS, Score-P/Vampir for GPU+MPI, ncu, arithmetic intensity, hierarchical roofline, tensor cores, roofline pitfalls |
| [08 — Course → project map](08-course-to-project-map.md) | (synthesis) | Every design decision and result of the Particles project traced back to the specific course concept (with slide refs) it instantiates |

Companion documents in `presentation/`:
- `EXAM_PREP.md` — condensed exam-prep guide + Q&A bank + numbers card.
- `PRESENTATION.md` — the slide-by-slide spec of the exam deck.
