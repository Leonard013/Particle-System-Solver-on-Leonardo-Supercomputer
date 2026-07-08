# Problem-size scaling (GInteractions/s vs N)

Same Mandelbrot window at increasing grid resolution; benchmark mode (`none 0`).
Speedup = GInt/s ratio vs serial at the same N; XXL serial is extrapolated
(serial rate is size-independent: median 0.283 GInt/s across measured sizes).

| size | N | steps | model | reps | median dyn (s) | GInt/s | speedup vs serial |
|---|---|---|---|---|---|---|---|
| S | 567 | 200 | serial | 2 | 0.2276 | 0.282 | 1.00x |
| S | 567 | 200 | omp32 | 3 | 0.01112 | 5.775 | 20.48x |
| S | 567 | 200 | cuda | 3 | 0.03601 | 1.783 | 6.32x |
| M | 2231 | 200 | serial | 2 | 3.516 | 0.283 | 1.00x |
| M | 2231 | 200 | omp32 | 3 | 0.1188 | 8.374 | 29.59x |
| M | 2231 | 200 | cuda | 3 | 0.1123 | 8.86 | 31.31x |
| L | 8996 | 200 | serial | 2 | 57.13 | 0.2833 | 1.00x |
| L | 8996 | 200 | omp32 | 3 | 1.93 | 8.387 | 29.61x |
| L | 8996 | 200 | cuda | 3 | 0.44 | 36.79 | 129.85x |
| XL | 35919 | 50 | serial | 2 | 227.7 | 0.2833 | 1.00x |
| XL | 35919 | 50 | omp32 | 3 | 7.645 | 8.438 | 29.78x |
| XL | 35919 | 50 | cuda | 3 | 0.6122 | 105.4 | 371.93x |
| XXL | 143768 | 10 | omp32 | 2 | 25.62 | 8.068 | 28.49x* |
| XXL | 143768 | 10 | cuda | 3 | 1.343 | 153.9 | 543.37x* |

`*` = vs extrapolated serial rate.
