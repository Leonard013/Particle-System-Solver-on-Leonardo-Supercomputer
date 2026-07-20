# Particle System Solver

Completed Polimi PhD School HPC final project: a reproducible 2D all-pairs particle solver with C++ and Python baselines plus OpenMP, CUDA, MPI, Numba CPU, and Numba CUDA ports. The dominant force kernel is `O(N²)`; all ports preserve the numerical model and the inner force-summation order.

## Verified Results

Leonardo A100 runs use the official input (`N = 2231`, 200 steps) and HDF5-disabled timing:

- The six-way course-gate matrix—C++ serial, OpenMP, CUDA, Python, Numba CPU, and Numba CUDA—passes all 15 pairwise checks at `rtol=atol=2e-3`.
- C++ serial, OpenMP, CUDA, MPI, Python, and Numba CPU produce bitwise-identical position and velocity arrays. Numba CUDA's largest relative velocity difference is `1.158e-3`, within the course gate.
- OpenMP reaches `29.2×` speedup at 32 threads (`91%` efficiency).
- CUDA reaches `31.1×` at the official size and up to `543×` in the size study.
- MPI reaches `113.7×` at 128 ranks (`89%` efficiency) for `N = 35,919`.

See [reports/summary.md](reports/summary.md), [reports/scaling_summary.md](reports/scaling_summary.md), [reports/mpi_summary.md](reports/mpi_summary.md), and [LEONARDO_RUN_STATUS.md](LEONARDO_RUN_STATUS.md) for the full evidence trail.

## Implementations and Layout

| Program | Source | Execution model |
|---|---|---|
| C++ serial | `src/cpp/particles.cpp` | Reference baseline |
| OpenMP | `src/cpp/particles_omp.cpp` | Shared-memory outer-loop parallelism |
| CUDA | `src/cpp/particles.cu` | One GPU thread per particle |
| MPI | `src/cpp/particles_mpi.cpp` | Block decomposition plus `MPI_Allgatherv` |
| Python | `src/python/particles.py` | NumPy/Python reference baseline |
| Numba CPU | `src/python/particles_numba.py` | `@njit(parallel=True)` and `prange` |
| Numba CUDA | `src/python/particles_numba_cuda.py` | `@cuda.jit` kernels |

Inputs live in `input/`, including size-study cases under `input/scaling/`. Automation is split between top-level run/submit scripts and `scripts/`. Raw SLURM logs are in `logs/`; parsed CSVs, plots, validation output, and profiler captures are in `reports/`. Exam material is under `presentation/`.

## Build and Run

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Choose a CMake preset appropriate to the host:

```bash
cmake --preset generic-x86-nogpu
cmake --build --preset generic-x86-nogpu -j
```

Use `macos-arm64`, `generic-x86-nvidia`, or `leonardo-a100` when applicable. CUDA and MPI targets are built only when their toolchains are available. On Leonardo, first run `source scripts/env.leonardo.sh`; on macOS use `source scripts/env.macos.sh`.

Representative no-output benchmark commands are:

```bash
./build/generic-x86-nogpu/particles_serial input/Particles.in none 0
OMP_NUM_THREADS=32 ./build/generic-x86-nogpu/particles_omp input/Particles.in none 0
python3 src/python/particles.py input/Particles.in none 0
python3 src/python/particles_numba.py input/Particles.in none 0
```

For CUDA use `particles_cuda`; for MPI use, for example, `mpirun -np 32 particles_mpi ...`. Passing `none 0` disables HDF5 and keeps I/O out of benchmark timings.

## Validation

Generate the six matrix outputs in `output/`, then run:

```bash
bash scripts/validate_all.sh
```

The script executes all 15 comparisons and exits nonzero if any comparison fails. On Leonardo, `bash run_validate.sh` adds strict `1e-12` checks for pairs expected to agree and a separate optional MPI comparison. The known Numba CPU versus Numba CUDA strict mismatch is informational; its course-gate result remains mandatory.

To compare any two files directly:

```bash
python3 tools/validate_particles_h5.py reference.h5 candidate.h5
```

## Reproducing Leonardo Evidence

The submission pipeline assumes the Leonardo build and virtual environment already exist:

```bash
bash scripts/submit_pipeline.sh
sbatch submit_python_ref.sh       # required for the full 15-pair matrix
bash run_validate.sh              # run after jobs finish, on the login node
```

MPI scaling and profiling are submitted separately with `run_mpi_all.sh`, `run_mpi_nodes.sh`, and `submit_profile.sh`. Regenerate committed summaries and plots with:

```bash
python3 tools/parse_benchmarks.py logs/cpp_all_*.out logs/py_all_*.out --outdir reports --plots
python3 tools/parse_scaling.py logs/scaling_*.out --outdir reports --plots
python3 tools/parse_mpi.py logs/mpi_all_*.out logs/mpi_nodes_*.out --outdir reports --plots
python3 tools/amdahl_fit.py --outdir reports
```

## Reproducibility Policy

Do not enable fast-math, change the inner `j` accumulation order, or introduce pair-symmetry updates without revalidating every implementation. CUDA intentionally uses round-to-nearest intrinsics to prevent contraction. Benchmark changes should record hardware, compiler flags, input size, repetitions, and whether HDF5 output was disabled.

The corrected exam deck is [presentation/Particles_exam_fixed.pptx](presentation/Particles_exam_fixed.pptx), with its editable content specification in [presentation/PRESENTATION.md](presentation/PRESENTATION.md).
