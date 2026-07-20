# Repository Guidelines

## Project Structure & Module Organization

Core implementations live in `src/cpp/` (serial C++17, OpenMP, MPI, and CUDA) and `src/python/` (NumPy, Numba CPU, and Numba CUDA). Simulation inputs are under `input/`; use `input/scaling/` for size studies. `tools/` contains HDF5 validation and report parsers, while `scripts/` and top-level `run_*.sh`/`submit_*.sh` files automate local and Leonardo/SLURM workflows. Generated HDF5 files belong in `output/`, scheduler output in `logs/`, and committed benchmark summaries and plots in `reports/`. Presentation material is kept in `presentation/`.

## Build, Test, and Development Commands

- `python3 -m pip install -r requirements.txt` installs Python, plotting, and HDF5 dependencies.
- `cmake --preset generic-x86-nogpu` configures a release CPU build; use `macos-arm64`, `generic-x86-nvidia`, or `leonardo-a100` when appropriate.
- `cmake --build --preset generic-x86-nogpu -j` builds host-available targets (normally serial and OpenMP). MPI/CUDA require a matching toolchain and preset.
- `./build/generic-x86-nogpu/particles_serial input/Particles.in none 0` runs without HDF5 output, the preferred benchmark mode.
- `bash scripts/validate_all.sh` compares previously generated `output/Particles_*.h5` files. On Leonardo, source `scripts/env.leonardo.sh` and use the matching submission scripts.

## Coding Style & Naming Conventions

Use four-space indentation. Follow existing C++ conventions: C++17, `PascalCase` types, `camelCase` functions and locals, and `constexpr` model constants. Python uses type hints, dataclasses, `snake_case`, and uppercase module constants. Keep compiler warnings clean (`-Wall -Wextra -Wpedantic`). Do not enable fast-math or reorder the inner all-pairs force loop; reproducible floating-point behavior is a project invariant.

## Testing Guidelines

There is no separate unit-test suite. Treat numerical regression as mandatory: generate reference and optimized HDF5 outputs, then run `python3 tools/validate_particles_h5.py reference.h5 candidate.h5`. The course gate uses `rtol=atol=2e-3`; same-family ports should also pass the stricter checks in `run_validate.sh`. Report particle-count or shape mismatches rather than hiding them.

## Commit & Pull Request Guidelines

Recent history favors concise, imperative subjects with a project prefix, for example `Particles: add MPI scaling harness`; scoped conventional subjects such as `chore: ...` are also used. Keep each commit focused. Pull requests should identify affected execution models, build preset and hardware, validation commands/results, and performance impact. Link relevant issues and include updated tables or plots for benchmark changes; screenshots are only needed for presentation or visualization changes.
