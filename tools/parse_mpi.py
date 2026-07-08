#!/usr/bin/env python3
"""
Parse the MPI benchmark logs (run_mpi_all.sh / run_mpi_nodes.sh).

Markers:
  ----- BENCH mpi ranks=<P> size=<M|L|XL> rep=<r> -----        (strong scaling)
  ----- BENCH mpi_weak ranks=<P> size=W<P> rep=<r> -----       (weak scaling)
Each block self-reports Particles/Steps/Pure dynamics time/GInteractions/s.

Strong-scaling speedup uses the serial medians measured earlier:
  M from reports/benchmarks.csv, L/XL from reports/scaling.csv.
Weak-scaling efficiency = t(p=1) / t(p)   (course definition).

Usage:
  python tools/parse_mpi.py logs/mpi_all_*.out logs/mpi_nodes_*.out \
      --outdir reports --plots
"""
from __future__ import annotations
import argparse, csv, os, re, statistics, sys
from collections import defaultdict

MARK = re.compile(r"-{2,}\s*BENCH\s+(?P<model>mpi(?:_weak)?)\s+ranks=(?P<ranks>\d+)\s+size=(?P<size>\S+)\s+rep=(?P<rep>\d+)\s*-{2,}")
RE_N = re.compile(r"Particles:\s+(\d+)")
RE_T = re.compile(r"Pure dynamics time:\s+([0-9.eE+-]+)\s*s")
RE_G = re.compile(r"Pure dynamics performance:\s+([0-9.eE+-]+)\s*GInteractions/s")


def parse(paths):
    runs, cur = [], None
    for p in paths:
        try:
            lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
        except OSError as e:
            print(f"[warn] cannot read {p}: {e}", file=sys.stderr); continue
        for ln in lines:
            if (m := MARK.search(ln)):
                if cur: runs.append(cur)
                cur = {"model": m["model"], "ranks": int(m["ranks"]), "size": m["size"],
                       "rep": int(m["rep"]), "N": None, "time": None, "giups": None}
                continue
            if cur is None: continue
            if (mm := RE_N.search(ln)) and cur["N"] is None: cur["N"] = int(mm[1])
            if (mm := RE_T.search(ln)):                      cur["time"] = float(mm[1])
            if (mm := RE_G.search(ln)):                      cur["giups"] = float(mm[1])
        if cur: runs.append(cur); cur = None
    return [r for r in runs if r["time"] is not None]


def median_from_csv(path, model, threads=None):
    try:
        vals = []
        for row in csv.DictReader(open(path)):
            if row["model"] != model: continue
            if threads is not None and row.get("threads") not in (str(threads), threads): continue
            vals.append(float(row["pure_dynamics_s"]))
        return statistics.median(vals) if vals else None
    except OSError:
        return None


def scaling_serial(path, size):
    try:
        vals = [float(r["pure_dynamics_s"]) for r in csv.DictReader(open(path))
                if r["model"] == "serial" and r["size"] == size]
        return statistics.median(vals) if vals else None
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--bench-csv", default="reports/benchmarks.csv")
    ap.add_argument("--scaling-csv", default="reports/scaling.csv")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    runs = parse(args.logs)
    if not runs:
        print("No MPI runs found.", file=sys.stderr); return 1

    with open(os.path.join(args.outdir, "mpi.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "size", "ranks", "rep", "N", "pure_dynamics_s", "giups"])
        for r in sorted(runs, key=lambda r: (r["model"], r["size"], r["ranks"], r["rep"])):
            w.writerow([r["model"], r["size"], r["ranks"], r["rep"], r["N"], r["time"], r["giups"]])

    serial = {"M": median_from_csv(args.bench_csv, "serial"),
              "L": scaling_serial(args.scaling_csv, "L"),
              "XL": scaling_serial(args.scaling_csv, "XL")}

    g = defaultdict(list)
    for r in runs: g[(r["model"], r["size"], r["ranks"])].append(r["time"])
    med = {k: statistics.median(v) for k, v in g.items()}

    md = ["# MPI benchmark summary", "",
          "Strong-scaling speedup vs the measured serial medians "
          f"(M={serial['M']:.4g}s, L={serial['L']:.4g}s, XL={serial['XL']:.4g}s).", "",
          "| kind | size | ranks | nodes | median dyn (s) | speedup | efficiency |",
          "|---|---|---|---|---|---|---|"]
    for (model, size, ranks) in sorted(med, key=lambda k: (k[0], k[1], k[2])):
        t = med[(model, size, ranks)]
        nodes = max(1, (ranks + 31) // 32)
        if model == "mpi" and serial.get(size):
            sp = serial[size] / t
            eff = sp / ranks
            md.append(f"| strong | {size} | {ranks} | {nodes} | {t:.4g} | {sp:.2f}x | {eff*100:.0f}% |")
        elif model == "mpi_weak":
            base = med.get(("mpi_weak", "W1", 1))
            weff = (base / t) if base else None
            md.append(f"| weak | {size} | {ranks} | {nodes} | {t:.4g} | - | "
                      + (f"{weff*100:.0f}% |" if weff else "- |"))
    open(os.path.join(args.outdir, "mpi_summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    if not args.plots:
        return 0
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] no matplotlib: {e}", file=sys.stderr); return 0

    # --- strong scaling: MPI (M/L/XL) vs OpenMP (M) ---
    plt.figure(figsize=(6.5, 4.5))
    ranks_all = sorted({k[2] for k in med if k[0] == "mpi"})
    plt.plot(ranks_all, ranks_all, "k--", alpha=.5, label="ideal")
    style = {"M": ("o-", "C0"), "L": ("s-", "C1"), "XL": ("^-", "C2")}
    for size, (fmt, col) in style.items():
        pts = sorted([(k[2], serial[size] / med[k]) for k in med
                      if k[0] == "mpi" and k[1] == size and serial.get(size)])
        if pts: plt.plot(*zip(*pts), fmt, color=col, label=f"MPI, {size} (N≈{ {'M':2231,'L':8996,'XL':35919}[size] })")
    # OpenMP reference from benchmarks.csv
    try:
        omp = defaultdict(list)
        for row in csv.DictReader(open(args.bench_csv)):
            if row["model"] == "omp" and row["threads"]:
                omp[int(row["threads"])].append(float(row["pure_dynamics_s"]))
        pts = sorted([(p, serial["M"] / statistics.median(ts)) for p, ts in omp.items()])
        if pts: plt.plot(*zip(*pts), "d:", color="C3", label="OpenMP, M (1 node)")
    except OSError:
        pass
    plt.xscale("log", base=2); plt.yscale("log", base=2)
    plt.xlabel("MPI ranks / OpenMP threads"); plt.ylabel("speedup vs serial")
    plt.title("MPI strong scaling across nodes (32 ranks/node)")
    plt.grid(True, which="both", alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "mpi_speedup.png"), dpi=130); plt.close()

    # --- weak scaling efficiency ---
    base = med.get(("mpi_weak", "W1", 1))
    pts = sorted([(k[2], base / med[k] * 100) for k in med if k[0] == "mpi_weak" and base])
    if pts:
        plt.figure(figsize=(6, 4))
        plt.plot(*zip(*pts), "o-", color="C0")
        plt.axhline(100, ls="--", color="k", alpha=.5)
        plt.xscale("log", base=2)
        plt.xlabel("MPI ranks (work per rank constant, N ∝ √p)")
        plt.ylabel("weak-scaling efficiency  t(1)/t(p)  (%)")
        plt.title("MPI weak scaling"); plt.ylim(0, 115)
        plt.grid(True, which="both", alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "mpi_weak.png"), dpi=130); plt.close()
    print(f"[written] plots -> {args.outdir}/mpi_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
