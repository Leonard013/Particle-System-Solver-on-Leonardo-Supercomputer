#!/usr/bin/env python3
"""
Fit Amdahl's law to the measured OpenMP strong-scaling data (course: Intro to
HPC, "Measuring performance").

Model:  S(p) = 1 / ((1-f) + f/p),  f = parallel fraction.
Linearized:  1/S = (1-f) + f * (1/p)  ->  least-squares line in x = 1/p.

Reads reports/benchmarks.csv (from tools/parse_benchmarks.py), writes
reports/amdahl_omp.png and prints the fit.

Usage: python tools/amdahl_fit.py [--csv reports/benchmarks.csv] [--outdir reports]
"""
from __future__ import annotations
import argparse, csv, statistics
from collections import defaultdict

import numpy as np


def load(csv_path):
    groups = defaultdict(list)
    serial = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["model"] == "omp" and row["threads"]:
                groups[int(row["threads"])].append(float(row["pure_dynamics_s"]))
            elif row["model"] == "serial":
                serial.append(float(row["pure_dynamics_s"]))
    if not serial or not groups:
        raise SystemExit("need serial + omp rows in the csv")
    t_serial = statistics.median(serial)
    threads = sorted(groups)
    speedup = [t_serial / statistics.median(groups[p]) for p in threads]
    return threads, speedup, t_serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="reports/benchmarks.csv")
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    p, S, t_serial = load(args.csv)
    x = 1.0 / np.asarray(p, dtype=float)     # 1/p
    y = 1.0 / np.asarray(S, dtype=float)     # 1/S
    f, a = np.polyfit(x, y, 1)               # slope=f (parallel), intercept=(1-f)
    serial_frac = a
    s_max = 1.0 / serial_frac if serial_frac > 0 else float("inf")

    print(f"Amdahl fit over p={p}")
    print(f"  parallel fraction f       = {f:.5f}")
    print(f"  serial fraction (1-f)     = {serial_frac:.5f}  ({serial_frac*100:.2f}%)")
    print(f"  asymptotic max speedup    = {s_max:.0f}x")
    for pi, si in zip(p, S):
        pred = 1.0 / (serial_frac + f / pi)
        print(f"  p={pi:>2}  measured {si:6.2f}x   Amdahl {pred:6.2f}x")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pp = np.linspace(1, max(p), 256)
    amdahl = 1.0 / (serial_frac + f / pp)

    plt.figure(figsize=(6, 4))
    plt.plot(pp, pp, "k--", alpha=.5, label="ideal")
    plt.plot(pp, amdahl, "-", color="C1",
             label=f"Amdahl fit (serial = {serial_frac*100:.2f}%)")
    plt.plot(p, S, "o", color="C0", label="measured (median of 3)")
    plt.xlabel("OpenMP threads")
    plt.ylabel("speedup vs serial")
    plt.title("OpenMP strong scaling vs Amdahl's law")
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.annotate(f"max speedup $1/(1-f)$ ≈ {s_max:.0f}×",
                 xy=(0.97, 0.05), xycoords="axes fraction", ha="right",
                 fontsize=9, color="dimgray")
    plt.tight_layout()
    out = f"{args.outdir}/amdahl_omp.png"
    plt.savefig(out, dpi=130)
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
