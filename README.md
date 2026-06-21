# Particle System Solver

This repository contains serial baseline implementations of a particle system solver for a parallel programming project.

The main objective is to optimize and parallelize the particle dynamics phase, which is dominated by an `O(N^2)` all-pairs force computation.

Students may implement optimized versions using one or more of the following programming models:

- OpenMP
- MPI
- CUDA
- OpenACC
- Hybrid approaches
- Numba CPU
- Numba CUDA

Both C++ and Python baseline implementations are provided.

---

## Repository Structure

```text
.
|-- CMakeLists.txt
|-- CMakePresets.json
|-- input
|   `-- Particles.in
|-- logs
|-- output
|-- README.md
|-- requirements.txt
|-- scripts
|   |-- env.leonardo.sh
|   |-- env.macos.sh
|   `-- validate_all.sh
|-- src
|   |-- cpp
|   |   `-- particles.cpp
|   `-- python
|       `-- particles.py
|-- submit_cpp.sh
|-- submit_cuda.sh
|-- submit_install_pyenv.sh
|-- submit_numba_cuda.sh
|-- submit_numba.sh
|-- submit_omp.sh
|-- submit_python.sh
|-- submit_validate.sh
`-- tools
    |-- validate_particles_h5.py
    `-- visualize_particles_h5.py
```

---

## Baseline Implementations

### C++ Baseline

The serial C++17 implementation is located in:

```text
src/cpp/particles.cpp
```

This version is intended to be optimized and parallelized using, for example:

- OpenMP
- CUDA
- OpenACC
- MPI
- Hybrid CPU/GPU approaches

The C++ code uses a structure-of-arrays layout for particle data. 
This layout is suitable for SIMD vectorization, cache blocking, GPU offloading, and MPI data packing.

The main computational kernel is:

```cpp
computeForces(...)
```

This `O(N^2)` all-pairs interaction loop is the primary target for optimization and parallelization.

---

### Python Baseline

A serial Python/NumPy implementation is available in:

```text
src/python/particles.py
```

This version is intended for students who prefer to work with Python-based acceleration tools such as:

- Numba CPU
- Numba CUDA

The Python code follows the same numerical model and output format as the C++ baseline.

---

## Input File

An example input file is provided in:

```text
input/Particles.in
```

The input file contains the following parameters, one per line, after removing empty lines and comment-only lines:

```text
generatingGridNx
generatingGridNy
generatingGridXs
generatingGridXe
generatingGridYs
generatingGridYe
screenGridNx
screenGridNy
screenGridXs
screenGridXe
screenGridYs
screenGridYe
maxFractalIterations
timeSteps
dt
outputEvery
```

You are free to modify the input parameters during development and benchmarking.

In particular, you are encouraged to study how the parallel speed-up changes as the problem size increases.

---

## Benchmark / No-Output Mode

For performance benchmarking, HDF5 output should normally be disabled.

The recommended C++ benchmark command is:

```bash
./path/to/particles_serial input/Particles.in none 0
```

The recommended Python benchmark command is:

```bash
python3 ./path/to/particles.py input/Particles.in none 0
```

The argument:

```text
none
```

disables HDF5 output.

The final command-line argument controls the output frequency. In benchmark mode, use:

```text
0
```

When HDF5 output is disabled, no `.h5` file is generated.

Please clearly state in your report whether benchmark timings were collected with or without HDF5 output enabled.

---

## Correctness and Validation

To check correctness, compare the final validation quantities printed by the optimized version against those printed by the original serial implementation.

The program prints quantities such as:

- sum of particle positions
- sum of particle velocities
- weighted position sums
- total momentum
- kinetic energy
- potential-like energy
- total energy-like quantity

Small numerical differences are expected and acceptable because of:

- different floating-point reduction orders
- different thread scheduling
- fused multiply-add instructions
- compiler optimizations
- CPU versus GPU arithmetic differences

However, the results should remain numerically consistent with the serial baseline.

For HDF5-based validation, use the validation tools provided in:

```text
tools/
```

For example:

```bash
python3 tools/validate_particles_h5.py output/reference.h5 output/optimized.h5
```

A convenience validation script is also provided:

```bash
scripts/validate_all.sh
```

---

## HDF5 Output and Visualization

The original program can optionally generate `.h5` files containing:

- particle weights
- time steps
- particle positions
- particle velocities
- screen data for visualization

These files may become large and are mainly intended for debugging or visualization.

To generate an HDF5 output file with the C++ version:

```bash
./src/cpp/particles_serial input/Particles.in output/Particles_cpp.h5 1000
```

To generate an HDF5 output file with the Python version:

```bash
python3 src/python/particles.py input/Particles.in output/Particles_python.h5 1000
```

The generated HDF5 file can be visualized using:

```bash
python3 tools/visualize_particles_h5.py output/Particles_cpp.h5
```

For official performance measurements, it is recommended to disable HDF5 output.

---

## Python Environment Installation

To install the Python environment on Leonardo, see:

```text
submit_install_pyenv.sh
```

The installation procedure is:

```bash
module purge
module load cuda/12.2
module load gcc/12.2.0
module load cmake/3.27.9
module load hdf5/1.14.3--gcc--12.2.0-spack0.22
module load python/3.11.7

python3 -m venv particles_venv --system-site-packages
source particles_venv/bin/activate

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --no-cache-dir -r requirements.txt

deactivate
```

For installation on a local machine, make sure that:

- Python is available
- CUDA drivers are installed, if CUDA or Numba CUDA will be used
- the required Python packages in `requirements.txt` are installed

---

## CMake Compilation

Example CMake configuration files are provided:

```text
CMakeLists.txt
CMakePresets.json
```

These have been tested for:

- OpenMP and CUDA on Leonardo
- OpenMP on macOS Apple Silicon

The tested compiler toolchains include:

- GNU
- NVCC
- Clang

Other C++17-compatible compilers may also work, provided that they support the required OpenMP/CUDA/HDF5 configuration.

In general, you are free to modify any files, including the compilation CMake files.

---

### Leonardo

Before compiling on Leonardo, source the provided environment script:

```bash
source scripts/env.leonardo.sh
```

Then configure, build, and install with:

```bash
cmake --preset leonardo-a100
cmake --build --preset leonardo-a100 -j
cmake --install build/leonardo-a100
```

---

### macOS Apple Silicon

Before compiling on macOS Apple Silicon, source the provided environment script:

```bash
source scripts/env.macos.sh
```

Then configure, build, and install with:

```bash
cmake --preset macos-arm64
cmake --build --preset macos-arm64 -j
cmake --install build/macos-arm64
```

---

### Generic x86 CPU Without NVIDIA GPU

This configuration has not been extensively tested:

```bash
cmake --preset generic-x86-nogpu
cmake --build --preset generic-x86-nogpu -j
cmake --install build/generic-x86-nogpu
```

---

### Generic x86 CPU With NVIDIA GPU

This configuration has not been extensively tested:

```bash
cmake --preset generic-x86-nvidia
cmake --build --preset generic-x86-nvidia -j
cmake --install build/generic-x86-nvidia
```

---

## Manual Compilation

The C++ code can also be compiled manually.

Without HDF5 support:

```bash
cd src/cpp
g++ -O3 -std=c++17 -Wall -Wextra -pedantic particles.cpp -o particles_serial
```

With HDF5 support:

```bash
cd src/cpp
g++ -O3 -std=c++17 -Wall -Wextra -pedantic -DUSE_HDF5 particles.cpp -o particles_serial -lhdf5_cpp -lhdf5
```

Depending on the system, the HDF5 compiler wrapper may also be used:

```bash
cd src/cpp
h5c++ -O3 -std=c++17 -Wall -Wextra -pedantic -DUSE_HDF5 particles.cpp -o particles_serial
```

---

## Running the Code

### C++ Serial Baseline

From the repository root:

```bash
./src/cpp/particles_serial input/Particles.in none 0
```

or, if the executable has been installed elsewhere:

```bash
./path/to/particles_serial input/Particles.in none 0
```

---

### Python Baseline

From the repository root:

```bash
python3 src/python/particles.py input/Particles.in none 0
```

---

## Job Submission Scripts

Several example submission scripts are provided:

```text
submit_cpp.sh
submit_omp.sh
submit_cuda.sh
submit_python.sh
submit_numba.sh
submit_numba_cuda.sh
submit_validate.sh
submit_install_pyenv.sh
```

These scripts are mainly intended for Leonardo, but can be adapted to other Linux-based HPC systems with minor changes.

---

## Development Guidelines

When implementing an optimized version:

1. Preserve the numerical model and force law.
2. Compare validation quantities against the serial baseline.
3. Benchmark with HDF5 output disabled unless explicitly studying I/O performance.
4. Report the input size, number of particles, number of time steps, hardware, compiler, and compilation flags.
5. Discuss strong and/or weak scaling where appropriate.

The main optimization target is the particle dynamics phase, especially the force computation.

---

## Notes on Floating-Point Reproducibility

Parallel implementations may not produce bitwise-identical results to the serial baseline.

This is normal.

Differences may arise from:

- different summation orders
- thread scheduling
- GPU execution order
- vectorization
- fused multiply-add instructions
- compiler optimization flags

The optimized implementation should nevertheless remain close to the serial result within a reasonable numerical tolerance.

---

## Further Instructions

Please read the comments in the source code for additional implementation details and suggestions.

If you find issues in the code or in the provided scripts, please report them to `m.celoria@cineca.it`

