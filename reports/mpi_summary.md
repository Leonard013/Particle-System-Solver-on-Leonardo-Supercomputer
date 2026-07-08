# MPI benchmark summary

Strong-scaling speedup vs the measured serial medians (M=3.516s, L=57.13s, XL=227.7s).

| kind | size | ranks | nodes | median dyn (s) | speedup | efficiency |
|---|---|---|---|---|---|---|
| strong | L | 32 | 1 | 1.885 | 30.31x | 95% |
| strong | L | 64 | 2 | 0.9957 | 57.38x | 90% |
| strong | L | 128 | 4 | 0.6367 | 89.73x | 70% |
| strong | M | 1 | 1 | 3.517 | 1.00x | 100% |
| strong | M | 2 | 1 | 1.76 | 2.00x | 100% |
| strong | M | 4 | 1 | 0.8822 | 3.99x | 100% |
| strong | M | 8 | 1 | 0.4464 | 7.88x | 98% |
| strong | M | 16 | 1 | 0.2357 | 14.92x | 93% |
| strong | M | 32 | 1 | 0.1498 | 23.47x | 73% |
| strong | M | 64 | 2 | 0.09119 | 38.55x | 60% |
| strong | M | 128 | 4 | 0.09199 | 38.22x | 30% |
| strong | XL | 64 | 2 | 3.727 | 61.10x | 95% |
| strong | XL | 128 | 4 | 2.002 | 113.72x | 89% |
| weak | W1 | 1 | 1 | 3.517 | - | 100% |
| weak | W16 | 16 | 1 | 3.718 | - | 95% |
| weak | W4 | 4 | 1 | 3.552 | - | 99% |
| weak | W64 | 64 | 2 | 3.804 | - | 92% |
