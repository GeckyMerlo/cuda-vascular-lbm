#!/usr/bin/env python3
"""Plot CUDA benchmark metrics as a function of active cells."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    return sorted(rows, key=lambda row: float(row["active_cells"]))


def values(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def labels(rows: list[dict[str, str]]) -> list[str]:
    return [row.get("dx") or row.get("mesh_name", "") for row in rows]


def plot_metric(ax, rows: list[dict[str, str]], metric: str, ylabel: str) -> None:
    x = values(rows, "active_cells")
    y = values(rows, metric)
    ax.plot(x, y, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("active cells")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)

    for xi, yi, label in zip(x, y, labels(rows)):
        ax.annotate(str(label), (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CUDA benchmark CSV metrics.")
    parser.add_argument("--csv", type=Path, default=Path("output/cuda_benchmark_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/cuda_benchmark_vs_cells.png"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import matplotlib.pyplot as plt

    rows = read_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.csv}")

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
    plot_metric(axes[0], rows, "mlups", "MLUPS")
    plot_metric(axes[1], rows, "ms_per_step", "ms / step")
    plot_metric(axes[2], rows, "estimated_memory_bandwidth_gb_s", "estimated GB/s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
