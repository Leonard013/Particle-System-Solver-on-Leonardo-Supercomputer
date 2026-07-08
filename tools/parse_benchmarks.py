#!/usr/bin/env python3
"""
Parse Particles benchmark logs (produced by run_cpp_all.sh / run_py_all.sh) into
a results table + speedup/efficiency and plots.

Each timed run in those logs is preceded by a marker line of the form
    ----- BENCH <model> [threads=<T>] rep=<R> -----
followed by the program's own report, from which we read:
    Particles:                  <N>
    Pure dynamics time:            <t> s
    Pure dynamics performance:  <g> GInteractions/s

Usage:
    python tools/parse_benchmarks.py logs/cpp_all_*.out logs/py_all_*.out \
        --outdir reports --plots
Outputs:
    reports/benchmarks.csv          one row per run
    reports/summary.md              median table + speedup + efficiency
    reports/speedup_omp.png         OpenMP strong-scaling speedup vs threads
    reports/efficiency_omp.png      OpenMP parallel efficiency vs threads
    reports/throughput.png          GInteractions/s per model (best)
"""
from __future__ import annotations
import argparse, re, statistics, sys, csv, os
from collections import defaultdict

MARK = re.compile(r"-{2,}\s*BENCH\s+(?P<model>\w+?)(?:\s+threads=(?P<threads>\d+))?\s+rep=(?P<rep>\d+)\s*-{2,}")
RE_N   = re.compile(r"Particles:\s+(\d+)")
RE_T   = re.compile(r"Pure dynamics time:\s+([0-9.eE+-]+)\s*s")
RE_G   = re.compile(r"Pure dynamics performance:\s+([0-9.eE+-]+)\s*GInteractions/s")


def parse_files(paths):
    runs = []
    cur = None
    for p in paths:
        try:
            lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
        except OSError as e:
            print(f"[warn] cannot read {p}: {e}", file=sys.stderr); continue
        for ln in lines:
            m = MARK.search(ln)
            if m:
                if cur:
                    runs.append(cur)
                cur = {"model": m["model"],
                       "threads": int(m["threads"]) if m["threads"] else None,
                       "rep": int(m["rep"]), "N": None, "time": None, "giups": None}
                continue
            if cur is None:
                continue
            if (mm := RE_N.search(ln)) and cur["N"] is None:   cur["N"] = int(mm[1])
            if (mm := RE_T.search(ln)):                        cur["time"] = float(mm[1])
            if (mm := RE_G.search(ln)):                        cur["giups"] = float(mm[1])
        if cur:
            runs.append(cur); cur = None
    # keep only runs that actually captured a dynamics time
    return [r for r in runs if r["time"] is not None]


def key(r):
    return (r["model"], r["threads"])


def summarize(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[key(r)].append(r)
    rows = []
    for (model, threads), rs in groups.items():
        times = [r["time"] for r in rs]
        giups = [r["giups"] for r in rs if r["giups"] is not None]
        rows.append({
            "model": model, "threads": threads, "reps": len(rs),
            "N": rs[0]["N"],
            "t_median": statistics.median(times),
            "t_min": min(times),
            "giups_median": statistics.median(giups) if giups else None,
            "giups_best": max(giups) if giups else None,
        })
    return rows


def serial_baseline(rows):
    # prefer explicit 'serial'; else omp threads=1
    for r in rows:
        if r["model"] == "serial":
            return r["t_median"]
    for r in rows:
        if r["model"] == "omp" and r["threads"] == 1:
            return r["t_median"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    runs = parse_files(args.logs)
    if not runs:
        print("No benchmark runs found in logs.", file=sys.stderr); return 1
    rows = summarize(runs)

    # raw CSV
    csv_path = os.path.join(args.outdir, "benchmarks.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "threads", "rep", "N", "pure_dynamics_s", "giups"])
        for r in sorted(runs, key=lambda r: (r["model"], r["threads"] or 0, r["rep"])):
            w.writerow([r["model"], r["threads"] or "", r["rep"], r["N"], r["time"], r["giups"]])

    t_serial = serial_baseline(rows)

    # summary md
    md = ["# Particles benchmark summary", ""]
    md.append(f"- Serial baseline pure-dynamics time (median): **{t_serial:.4g} s**" if t_serial else "- (no serial baseline found)")
    md += ["", "| model | threads | reps | N | median t (s) | best GInt/s | speedup vs serial | efficiency |",
           "|---|---|---|---|---|---|---|---|"]
    def sortk(r):
        order = {"serial":0,"omp":1,"cuda":2,"numba":3,"numba_cuda":4}
        return (order.get(r["model"],9), r["threads"] or 0)
    for r in sorted(rows, key=sortk):
        sp = (t_serial / r["t_median"]) if t_serial else None
        eff = (sp / r["threads"]) if (sp and r["threads"]) else None
        md.append("| {model} | {th} | {reps} | {N} | {t:.4g} | {g} | {sp} | {eff} |".format(
            model=r["model"], th=r["threads"] if r["threads"] else "-", reps=r["reps"], N=r["N"],
            t=r["t_median"],
            g=(f"{r['giups_best']:.3g}" if r["giups_best"] else "-"),
            sp=(f"{sp:.2f}x" if sp else "-"),
            eff=(f"{eff*100:.0f}%" if eff else "-")))
    md_path = os.path.join(args.outdir, "summary.md")
    open(md_path, "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[written] {csv_path}\n[written] {md_path}")

    if args.plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[warn] matplotlib unavailable, skipping plots: {e}", file=sys.stderr); return 0
        omp = sorted([r for r in rows if r["model"] == "omp" and r["threads"]], key=lambda r: r["threads"])
        if omp and t_serial:
            th = [r["threads"] for r in omp]
            sp = [t_serial / r["t_median"] for r in omp]
            eff = [s / t for s, t in zip(sp, th)]
            plt.figure(figsize=(6,4))
            plt.plot(th, sp, "o-", label="measured")
            plt.plot(th, th, "k--", alpha=.5, label="ideal")
            plt.xlabel("OpenMP threads"); plt.ylabel("speedup vs serial"); plt.title("OpenMP strong scaling")
            plt.legend(); plt.grid(True, alpha=.3); plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, "speedup_omp.png"), dpi=130); plt.close()
            plt.figure(figsize=(6,4))
            plt.plot(th, [e*100 for e in eff], "s-")
            plt.axhline(100, ls="--", color="k", alpha=.5)
            plt.xlabel("OpenMP threads"); plt.ylabel("parallel efficiency (%)"); plt.title("OpenMP efficiency")
            plt.ylim(0, 110); plt.grid(True, alpha=.3); plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, "efficiency_omp.png"), dpi=130); plt.close()
        # throughput bar
        best = {}
        for r in rows:
            label = r["model"] if not (r["model"]=="omp") else f"omp{r['threads']}"
            if r["giups_best"]:
                best[label] = max(best.get(label, 0), r["giups_best"])
        if best:
            labels = list(best.keys()); vals = [best[k] for k in labels]
            plt.figure(figsize=(7,4))
            plt.bar(labels, vals)
            plt.ylabel("GInteractions/s (best)"); plt.title("Throughput by implementation")
            plt.xticks(rotation=45, ha="right"); plt.grid(True, axis="y", alpha=.3); plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, "throughput.png"), dpi=130); plt.close()
        print(f"[written] plots -> {args.outdir}/*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
