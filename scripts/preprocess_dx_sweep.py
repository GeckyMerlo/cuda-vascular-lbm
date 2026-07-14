#!/usr/bin/env python3
"""Generate voxel domains for a decreasing dx sweep."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_DX_VALUES = "0.50,0.375,0.25,0.20,0.15,0.125,0.10"


def parse_dx_values(value: str) -> list[float]:
    dx_values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not dx_values:
        raise argparse.ArgumentTypeError("dx list cannot be empty")
    if any(dx <= 0.0 for dx in dx_values):
        raise argparse.ArgumentTypeError("all dx values must be positive")
    return dx_values


def dx_label(dx: float) -> str:
    return f"dx{dx:g}".replace(".", "p")


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(command), flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def parse_count(pattern: str, output: str) -> int | None:
    match = re.search(pattern, output)
    return int(match.group(1).replace(",", "")) if match else None


def parse_preprocess_output(output: str) -> dict[str, int | None]:
    nx = ny = nz = None
    grid_match = re.search(r"nx\s+=\s+(\d+),\s+ny\s+=\s+(\d+),\s+nz\s+=\s+(\d+)", output)
    if grid_match:
        nx, ny, nz = (int(value) for value in grid_match.groups())

    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "total_voxels": parse_count(r"total voxels\s+=\s+([\d,]+)", output),
        "solid_cells": parse_count(r"SOLID\s+:\s+([\d,]+)", output),
        "fluid_cells": parse_count(r"FLUID\s+:\s+([\d,]+)", output),
        "inlet_cells": parse_count(r"INLET\s+:\s+([\d,]+)", output),
        "outlet_cells": parse_count(r"OUTLET:\s+([\d,]+)", output),
    }


def write_summary_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def preprocess_one(args: argparse.Namespace, repo: Path, dx: float) -> dict[str, object]:
    geo_stem = Path(args.geo).stem
    label = dx_label(dx)
    output_file = args.output_dir / f"{geo_stem}_{label}_voxel_domain.bin"
    vtk_file = args.output_dir / f"{geo_stem}_{label}.vtk"

    command = [
        sys.executable,
        "src/mesh_preprocessing.py",
        "--geo",
        args.geo,
        "--dx",
        f"{dx:g}",
        "--batch-z",
        str(args.batch_z),
        "--output",
        str(output_file),
        "--vtk-output",
        str(vtk_file),
    ]

    start = time.perf_counter()
    completed = run_command(command, repo)
    elapsed_s = time.perf_counter() - start

    if completed.stdout:
        print(completed.stdout, end="")

    parsed = parse_preprocess_output(completed.stdout or "")
    active_cells = None
    if parsed["fluid_cells"] is not None:
        active_cells = (
            int(parsed["fluid_cells"])
            + int(parsed["inlet_cells"] or 0)
            + int(parsed["outlet_cells"] or 0)
        )

    row: dict[str, object] = {
        "geo": args.geo,
        "dx": dx,
        "status": "ok" if completed.returncode == 0 else "failed",
        "elapsed_s": elapsed_s,
        "output_file": str(output_file),
        "vtk_file": str(vtk_file),
        **parsed,
        "active_cells": active_cells,
    }

    write_summary_row(repo / args.summary_csv, row)

    if completed.returncode != 0 and not args.keep_going:
        raise SystemExit(completed.returncode)

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run mesh_preprocessing.py for progressively smaller dx values."
    )
    parser.add_argument("--geo", default="vena_cilindrica.geo")
    parser.add_argument("--dx-values", type=parse_dx_values, default=parse_dx_values(DEFAULT_DX_VALUES))
    parser.add_argument("--batch-z", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("msh/dx_sweep"))
    parser.add_argument("--summary-csv", default="msh/dx_sweep/preprocess_summary.csv")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()

    if args.batch_z <= 0:
        raise SystemExit("--batch-z must be positive")

    dx_values = sorted(args.dx_values, reverse=True)
    print("dx sweep:", ", ".join(f"{dx:g}" for dx in dx_values), flush=True)

    rows = [preprocess_one(args, repo, dx) for dx in dx_values]

    print("\nSummary:")
    for row in rows:
        print(
            f"dx={row['dx']} status={row['status']} "
            f"active_cells={row['active_cells']} elapsed_s={row['elapsed_s']:.2f} "
            f"file={row['output_file']}"
        )

    print(f"\nWrote {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
