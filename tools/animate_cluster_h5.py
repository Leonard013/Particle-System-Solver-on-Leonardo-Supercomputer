#!/usr/bin/env python3

"""
Animate a single particle cluster from HDF5 solver output.

The full domain spans ~1090 units while a particle moves ~1.8 units over 200
steps, so whole-domain animations look static. This tool isolates one
gravitationally bound cluster and frames it tightly, which makes the collapse
visible.

Requires an every-step trajectory, produced with outputEvery = 1:

    ./build/<preset>/particles_serial input/Particles.in output/Particles_full_every1.h5 1

Usage:

    python3 tools/animate_cluster_h5.py output/Particles_full_every1.h5 \
        --save reports/cluster_evolution.gif

Optionally shrink the result with an ffmpeg palette pass:

    ffmpeg -i in.gif -vf "split[a][b];[a]palettegen=max_colors=96[p];\
[b][p]paletteuse=dither=bayer:bayer_scale=3" -loop 0 out.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


# ---------------------------------------------------------------------
# Clustering and rendering defaults
# ---------------------------------------------------------------------

LINK_THRESHOLD = 12.0
TRAIL_LEN = 18

BACKGROUND = "#07080d"
FRAME_COLOR = "#2a2f3d"
LABEL_COLOR = "#9aa4bb"
TITLE_COLOR = "#e8ecf5"
TRAIL_COLOR = "#39d6ff"
SPEED_CMAP = "plasma"


def select_cluster(
    first: np.ndarray,
    last: np.ndarray,
    nmin: int,
    nmax: int,
    forced: int | None,
) -> np.ndarray:
    """
    Group the initial positions with single-linkage clustering.

    Without --cluster, pick the cluster whose mean displacement is largest
    relative to its own extent: that is the one whose motion reads best.
    """
    labels = fcluster(linkage(first, method="single"), t=LINK_THRESHOLD, criterion="distance")

    if forced is not None:
        mask = labels == forced
        if not mask.any():
            raise SystemExit(f"cluster {forced} is empty; found {labels.max()} clusters")
        return mask

    best: tuple[float, np.ndarray, int, int] | None = None

    for cluster in range(1, labels.max() + 1):
        mask = labels == cluster
        count = int(mask.sum())

        if not (nmin <= count <= nmax):
            continue

        extent = float(max(np.ptp(first[mask, 0]), np.ptp(first[mask, 1])))
        score = float(np.linalg.norm(last[mask] - first[mask], axis=1).mean()) / extent

        if best is None or score > best[0]:
            best = (score, mask, cluster, count)

    if best is None:
        raise SystemExit(f"no cluster holds between {nmin} and {nmax} particles")

    print(f"cluster {best[2]}: {best[3]} particles, displacement/extent = {best[0]:.3f}")
    return best[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])

    parser.add_argument("filename", help="HDF5 file written with outputEvery = 1")
    parser.add_argument("--save", type=Path, required=True, help="output path, .gif or .mp4")
    parser.add_argument("--cluster", type=int, default=None, help="cluster id, default: auto-select")
    parser.add_argument("--nmin", type=int, default=60, help="smallest cluster to consider, default: 60")
    parser.add_argument("--nmax", type=int, default=250, help="largest cluster to consider, default: 250")
    parser.add_argument("--zoom", type=float, default=1.45,
                        help="frame width as a multiple of the initial cluster extent, default: 1.45")
    parser.add_argument("--fps", type=int, default=20, help="frames per second, default: 20")
    parser.add_argument("--stride", type=int, default=1, help="use every Nth saved frame, default: 1")
    parser.add_argument("--figsize", type=float, default=5.2, help="figure side in inches, default: 5.2")
    parser.add_argument("--dpi", type=int, default=100, help="figure dpi, default: 100")

    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    with h5py.File(args.filename, "r") as h5:
        positions = h5["/pos"][:, :, :2]
        velocities = h5["/vel"][:, :, :2]
        weights = h5["/weight"][:]
        steps = h5["/step"][:]
        dt = float(h5["/"].attrs.get("dt", 0.0))

    if positions.shape[0] < 2:
        raise SystemExit("need at least two frames; regenerate with outputEvery = 1")

    mask = select_cluster(positions[0], positions[-1], args.nmin, args.nmax, args.cluster)

    traj = positions[:: args.stride, mask, :]
    speed = np.linalg.norm(velocities[:: args.stride, mask, :], axis=2)
    steps = steps[:: args.stride]
    weight = weights[mask]
    count = int(mask.sum())

    # Frame on the initial cluster so the collapse stays large; the few ejected
    # particles simply leave the view.
    center_x = float(traj[0, :, 0].mean())
    center_y = float(traj[0, :, 1].mean())
    half = float(max(np.ptp(traj[0, :, 0]), np.ptp(traj[0, :, 1]))) * args.zoom / 2.0

    sizes = 22.0 + 130.0 * (weight - weight.min()) / max(float(np.ptp(weight)), 1e-12)

    fig, ax = plt.subplots(figsize=(args.figsize, args.figsize), dpi=args.dpi)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(center_x - half, center_x + half)
    ax.set_ylim(center_y - half, center_y + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_color(FRAME_COLOR)

    trails = [
        ax.plot([], [], "-", lw=0.7, color=TRAIL_COLOR,
                alpha=0.10 + 0.20 * (age / TRAIL_LEN))[0]
        for age in range(TRAIL_LEN)
    ]

    scatter = ax.scatter(
        traj[0, :, 0], traj[0, :, 1],
        s=sizes, c=speed[0], cmap=SPEED_CMAP,
        vmin=0.0, vmax=float(np.percentile(speed, 98)),
        edgecolors="none", alpha=0.95,
    )

    ax.set_title(f"One cluster of {count} particles — all-pairs gravity",
                 color=TITLE_COLOR, fontsize=11, pad=10)

    label = ax.text(0.025, 0.975, "", transform=ax.transAxes, va="top", ha="left",
                    color=LABEL_COLOR, fontsize=9, family="monospace")

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.042, pad=0.02)
    colorbar.set_label("speed", color=LABEL_COLOR, fontsize=9)
    colorbar.ax.tick_params(colors=LABEL_COLOR, labelsize=8)
    colorbar.outline.set_edgecolor(FRAME_COLOR)

    fig.tight_layout()

    def update(frame: int):
        scatter.set_offsets(traj[frame])
        scatter.set_array(speed[frame])

        for age, line in enumerate(trails):
            start = frame - (TRAIL_LEN - age)
            stop = start + 1

            if start < 0:
                line.set_data([], [])
                continue

            # One Line2D per trail age, NaN-separated so each particle keeps its
            # own segment instead of being joined into a single polyline.
            xs = np.full(3 * count, np.nan)
            ys = np.full(3 * count, np.nan)
            xs[0::3], xs[1::3] = traj[start, :, 0], traj[stop, :, 0]
            ys[0::3], ys[1::3] = traj[start, :, 1], traj[stop, :, 1]
            line.set_data(xs, ys)

        elapsed = f"  t = {steps[frame] * dt:.3f}" if dt else ""
        label.set_text(f"step {int(steps[frame]):>4d} / {int(steps[-1])}{elapsed}")

        return [scatter, label, *trails]

    anim = animation.FuncAnimation(fig, update, frames=traj.shape[0], blit=False)

    args.save.parent.mkdir(parents=True, exist_ok=True)

    if args.save.suffix == ".gif":
        anim.save(args.save, writer=animation.PillowWriter(fps=args.fps))
    else:
        anim.save(args.save, fps=args.fps)

    print(f"wrote {args.save} ({args.save.stat().st_size / 1e6:.2f} MB, {traj.shape[0]} frames)")


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
