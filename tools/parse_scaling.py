#!/usr/bin/env python3
"""
Parse the problem-size scaling log (run_scaling.sh) into a table + plots.

Markers:   ----- BENCH <model> size=<label> rep=<r> -----
Per block: Particles: <N> / Steps: <steps> / Pure dynamics time: <t> s
           Pure dynamics performance: <g> GInteractions/s

Usage:
    python tools/parse_scaling.py logs/scaling_*.out --outdir reports --plots
Outputs:
    reports/scaling.csv, reports/scaling_summary.md,
    reports/scaling_giups.png, reports/scaling_speedup.png
"""
from __future__ import annotations
import argparse, csv, os, re, statistics, sys
from collections import defaultdict

MARK = re.compile(r"-{2,}\s*BENCH\s+(?P<model>\w+)\s+size=(?P<size>\S+)\s+rep=(?P<rep>\d+)\s*-{2,}")
RE_N = re.compile(r"Particles:\s+(\d+)")
RE_S = re.compile(r"Steps:\s+(\d+)")
RE_T = re.compile(r"Pure dynamics time:\s+([0-9.eE+-]+)\s*s")
RE_G = re.compile(r"Pure dynamics performance:\s+([0-9.eE+-]+)\s*GInteractions/s")

SIZE_ORDER = {"S": 0, "M": 1, "L": 2, "XL": 3, "XXL": 4}
MODEL_ORDER = {"serial": 0, "omp32": 1, "cuda": 2}


def parse(paths):
    runs, cur = [], None
    for p in paths:
        try:
            text = open(p, encoding="utf-8", errors="replace").read().splitlines()
        except OSError as e:
            print(f"[warn] cannot read {p}: {e}", file=sys.stderr); continue
        for ln in text:
            if (m := MARK.search(ln)):
                if cur: runs.append(cur)
                cur = {"model": m["model"], "size": m["size"], "rep": int(m["rep"]),
                       "N": None, "steps": None, "time": None, "giups": None}
                continue
            if cur is None: continue
            if (mm := RE_N.search(ln)) and cur["N"] is None:     cur["N"] = int(mm[1])
            if (mm := RE_S.search(ln)) and cur["steps"] is None: cur["steps"] = int(mm[1])
            if (mm := RE_T.search(ln)):                          cur["time"] = float(mm[1])
            if (mm := RE_G.search(ln)):                          cur["giups"] = float(mm[1])
        if cur: runs.append(cur); cur = None
    return [r for r in runs if r["time"] is not None and r["giups"] is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    runs = parse(args.logs)
    if not runs:
        print("No scaling runs found.", file=sys.stderr); return 1

    with open(os.path.join(args.outdir, "scaling.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "size", "rep", "N", "steps", "pure_dynamics_s", "giups"])
        for r in sorted(runs, key=lambda r: (SIZE_ORDER.get(r["size"], 9), MODEL_ORDER.get(r["model"], 9), r["rep"])):
            w.writerow([r["model"], r["size"], r["rep"], r["N"], r["steps"], r["time"], r["giups"]])

    g = defaultdict(list)
    for r in runs: g[(r["model"], r["size"])].append(r)
    agg = {}
    for (model, size), rs in g.items():
        agg[(model, size)] = {
            "N": rs[0]["N"], "steps": rs[0]["steps"], "reps": len(rs),
            "giups": statistics.median([r["giups"] for r in rs]),
            "t": statistics.median([r["time"] for r in rs]),
        }

    sizes = sorted({s for (_, s) in agg}, key=lambda s: SIZE_ORDER.get(s, 9))
    serial_rates = [agg[("serial", s)]["giups"] for s in sizes if ("serial", s) in agg]
    serial_ref = statistics.median(serial_rates) if serial_rates else None

    md = ["# Problem-size scaling (GInteractions/s vs N)", "",
          "Same Mandelbrot window at increasing grid resolution; benchmark mode (`none 0`).",
          "Speedup = GInt/s ratio vs serial at the same N; XXL serial is extrapolated",
          f"(serial rate is size-independent: median {serial_ref:.3g} GInt/s across measured sizes)." if serial_ref else "",
          "", "| size | N | steps | model | reps | median dyn (s) | GInt/s | speedup vs serial |", "|---|---|---|---|---|---|---|---|"]
    for s in sizes:
        for m in ["serial", "omp32", "cuda"]:
            if (m, s) not in agg: continue
            a = agg[(m, s)]
            base = agg.get(("serial", s), {}).get("giups") or serial_ref
            sp = a["giups"] / base if base else None
            note = "*" if (m != "serial" and ("serial", s) not in agg) else ""
            md.append(f"| {s} | {a['N']} | {a['steps']} | {m} | {a['reps']} | {a['t']:.4g} | {a['giups']:.4g} | "
                      + (f"{sp:.2f}x{note} |" if sp else "- |"))
    md += ["", "`*` = vs extrapolated serial rate."]
    open(os.path.join(args.outdir, "scaling_summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    if args.plots:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[warn] matplotlib unavailable: {e}", file=sys.stderr); return 0
        style = {"serial": ("o-", "serial"), "omp32": ("s-", "OpenMP 32t"), "cuda": ("^-", "CUDA A100")}
        plt.figure(figsize=(6.5, 4.5))
        for m, (fmt, lbl) in style.items():
            pts = sorted([(agg[(m, s)]["N"], agg[(m, s)]["giups"]) for s in sizes if (m, s) in agg])
            if pts: plt.loglog(*zip(*pts), fmt, label=lbl)
        plt.xlabel("N (particles)"); plt.ylabel("GInteractions/s")
        plt.title("Problem-size scaling — all-pairs N-body (A100 node)")
        plt.grid(True, which="both", alpha=.3); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "scaling_giups.png"), dpi=130); plt.close()

        plt.figure(figsize=(6.5, 4.5))
        for m in ["omp32", "cuda"]:
            pts = []
            for s in sizes:
                if (m, s) not in agg: continue
                base = agg.get(("serial", s), {}).get("giups") or serial_ref
                if base: pts.append((agg[(m, s)]["N"], agg[(m, s)]["giups"] / base))
            pts.sort()
            if pts: plt.semilogx(*zip(*pts), style[m][0], label=style[m][1])
        plt.xlabel("N (particles)"); plt.ylabel("speedup vs serial")
        plt.title("Speedup vs problem size"); plt.grid(True, which="both", alpha=.3)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "scaling_speedup.png"), dpi=130); plt.close()
        print(f"[written] plots -> {args.outdir}/scaling_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
