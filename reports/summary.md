# Particles benchmark summary

- Serial baseline pure-dynamics time (median): **3.516 s**

| model | threads | reps | N | median t (s) | best GInt/s | speedup vs serial | efficiency |
|---|---|---|---|---|---|---|---|
| serial | - | 3 | 2231 | 3.516 | 0.283 | 1.00x | - |
| omp | 1 | 3 | 2231 | 3.514 | 0.283 | 1.00x | 100% |
| omp | 2 | 3 | 2231 | 1.758 | 0.566 | 2.00x | 100% |
| omp | 4 | 3 | 2231 | 0.8804 | 1.13 | 3.99x | 100% |
| omp | 8 | 3 | 2231 | 0.4412 | 2.26 | 7.97x | 100% |
| omp | 16 | 3 | 2231 | 0.2297 | 4.34 | 15.31x | 96% |
| omp | 32 | 3 | 2231 | 0.1204 | 8.31 | 29.20x | 91% |
| cuda | - | 3 | 2231 | 0.1131 | 8.8 | 31.10x | - |
| numba | - | 3 | 2231 | 0.1277 | 7.83 | 27.52x | - |
| numba_cuda | - | 3 | 2231 | 0.2967 | 3.42 | 11.85x | - |
