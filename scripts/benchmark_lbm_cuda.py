#!/usr/bin/env python3
"""Benchmark and Nsight Compute helper for the CUDA LBM project."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_MESH = "msh/cilindric_vessel_stenosis30_voxel_domain.bin"
DEFAULT_BINARY = "build-cuda/vascular_lbm"
DEFAULT_BYTES_PER_CELL_STEP = 912.0

NCU_METRICS = [
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]

GPU_COUNTER_PERMISSION_ERROR = "ERR_NVGPUCTRPERM"


def run_command(command: list[str], cwd: Path, capture: bool = True) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(command), flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def parse_loaded_domain(output: str) -> tuple[int | None, int | None, int | None]:
    match = re.search(r"Loaded domain:\s+(\d+)\s+x\s+(\d+)\s+x\s+(\d+)\s+cells", output)
    if not match:
        return None, None, None
    return tuple(int(value) for value in match.groups())


def read_active_cells(stats_path: Path) -> int | None:
    if not stats_path.exists():
        return None

    with stats_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return None

    for row in reversed(rows):
        value = row.get("active_cells")
        if value:
            return int(value)

    return None


def numeric_metric_value(value: str) -> float | None:
    cleaned = value.strip().replace(",", "")
    if not cleaned or cleaned == "n/a":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_ncu_csv(csv_text: str) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    reader = csv.DictReader(csv_text.splitlines())

    for row in reader:
        name = row.get("Metric Name")
        value = row.get("Metric Value")
        if not name or value is None:
            continue
        parsed = numeric_metric_value(value)
        if parsed is None:
            continue
        values.setdefault(name, []).append(parsed)

    return {
        name: sum(metric_values) / len(metric_values)
        for name, metric_values in values.items()
        if metric_values
    }


def ncu_permission_denied(output: str) -> bool:
    return GPU_COUNTER_PERMISSION_ERROR in output


def print_ncu_permission_hint() -> None:
    print(
        "Nsight Compute cannot read NVIDIA GPU performance counters on this machine "
        f"({GPU_COUNTER_PERMISSION_ERROR}). The benchmark executable still ran, but "
        "DRAM throughput %, achieved occupancy %, and roofline need GPU counter "
        "access enabled by an administrator or a different GPU node.",
        file=sys.stderr,
    )


def write_csv_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_dx_from_mesh(mesh: str) -> float | None:
    match = re.search(r"_dx([0-9]+(?:p[0-9]+)?)_", Path(mesh).name)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def resolve_meshes(args: argparse.Namespace) -> list[str]:
    meshes: list[str] = []

    if args.mesh:
        meshes.extend(args.mesh)

    for pattern in args.mesh_glob:
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"warning: mesh glob matched nothing: {pattern}", file=sys.stderr)
        meshes.extend(matches)

    if not meshes:
        meshes.append(DEFAULT_MESH)

    unique_meshes = list(dict.fromkeys(meshes))
    return sorted(unique_meshes, key=lambda mesh: (parse_dx_from_mesh(mesh) is None, parse_dx_from_mesh(mesh) or 0.0))


def lbm_command(args: argparse.Namespace, mesh: str, steps: int, output_interval: int) -> list[str]:
    command = [
        str(args.binary),
        mesh,
        str(steps),
        str(output_interval),
        str(args.tau),
        "--warmup-steps",
        str(args.warmup_steps),
        "--outlet-kernel",
        args.outlet_kernel,
        "--max-particles",
        "0",
    ]
    if args.performance_computation:
        command.append("--performance-computation")
    return command


def run_benchmark(args: argparse.Namespace, repo: Path, mesh: str, block_size: int) -> dict[str, object]:
    output_interval = args.output_interval or max(args.steps, 1)
    command = lbm_command(args, mesh, args.steps, output_interval)

    start = time.perf_counter()
    completed = run_command(command, repo)
    elapsed_s = time.perf_counter() - start

    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    stats_path = repo / "output" / "lbm_stats.csv"
    active_cells = read_active_cells(stats_path)
    nx, ny, nz = parse_loaded_domain(completed.stdout or "")
    total_cells = nx * ny * nz if nx and ny and nz else None
    cells_for_rate = active_cells or total_cells

    if not cells_for_rate:
        raise SystemExit("Could not infer active cell count or total cell count.")

    simulated_steps = args.warmup_steps + args.steps
    ms_per_step = elapsed_s * 1000.0 / simulated_steps
    mlups = cells_for_rate * simulated_steps / elapsed_s / 1.0e6
    estimated_gbs = (
        args.bytes_per_cell_step * cells_for_rate * simulated_steps / elapsed_s / 1.0e9
    )

    result: dict[str, object] = {
        "mesh": mesh,
        "mesh_name": Path(mesh).name,
        "dx": parse_dx_from_mesh(mesh),
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "simulated_steps": simulated_steps,
        "output_interval": output_interval,
        "outlet_kernel": args.outlet_kernel,
        "tau": args.tau,
        "cuda_block_size": block_size,
        "active_cells": active_cells,
        "total_cells": total_cells,
        "elapsed_s": elapsed_s,
        "ms_per_step": ms_per_step,
        "mlups": mlups,
        "bytes_per_cell_step": args.bytes_per_cell_step,
        "estimated_memory_bandwidth_gb_s": estimated_gbs,
        "nsight_dram_throughput_pct": "",
        "nsight_achieved_occupancy_pct": "",
        "kernel_profile_csv": "",
    }

    if args.performance_computation:
        kernel_profile = repo / "output" / "kernel_profile.csv"
        if kernel_profile.exists():
            profile_copy = (
                repo
                / "output"
                / f"kernel_profile_{Path(mesh).stem}_block{block_size}.csv"
            )
            shutil.copyfile(kernel_profile, profile_copy)
            result["kernel_profile_csv"] = str(profile_copy.relative_to(repo))

    print(json.dumps(result, indent=2), flush=True)
    return result


def run_ncu_metrics(args: argparse.Namespace, repo: Path, mesh: str) -> dict[str, float | str]:
    ncu = shutil.which("ncu")
    if not ncu:
        print("ncu not found; skipping Nsight Compute metrics.", file=sys.stderr)
        return {}

    ncu_steps = args.ncu_steps or min(args.steps, 20)
    command = [
        ncu,
        "--csv",
        "--page",
        "raw",
        "--target-processes",
        "all",
        "--metrics",
        ",".join(NCU_METRICS),
        *lbm_command(args, mesh, ncu_steps, max(ncu_steps, 1)),
    ]

    completed = run_command(command, repo)
    if completed.stdout:
        print(completed.stdout, end="")
    output = completed.stdout or ""
    if ncu_permission_denied(output):
        print_ncu_permission_hint()
        return {
            "nsight_dram_throughput_pct": "permission_denied",
            "nsight_achieved_occupancy_pct": "permission_denied",
        }
    if completed.returncode != 0:
        print("ncu metric collection failed.", file=sys.stderr)
        return {}

    parsed = parse_ncu_csv(output)
    summary = {
        "nsight_dram_throughput_pct": parsed.get(NCU_METRICS[0], float("nan")),
        "nsight_achieved_occupancy_pct": parsed.get(NCU_METRICS[1], float("nan")),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_roofline(args: argparse.Namespace, repo: Path, mesh: str) -> None:
    ncu = shutil.which("ncu")
    if not ncu:
        print("ncu not found; skipping roofline report.", file=sys.stderr)
        return

    ncu_steps = args.ncu_steps or min(args.steps, 20)
    mesh_stem = Path(mesh).stem
    report_base = repo / f"{args.roofline_report}_{mesh_stem}"
    command = [
        ncu,
        "--section",
        "SpeedOfLight_RooflineChart",
        "--target-processes",
        "all",
        "--force-overwrite",
        "-o",
        str(report_base),
        *lbm_command(args, mesh, ncu_steps, max(ncu_steps, 1)),
    ]

    completed = run_command(command, repo)
    if completed.stdout:
        print(completed.stdout, end="")
    if ncu_permission_denied(completed.stdout or ""):
        print_ncu_permission_hint()
        return
    if completed.returncode != 0:
        print("ncu roofline collection failed.", file=sys.stderr)
        return

    print(f"Roofline report written to {report_base}.ncu-rep", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LBM CUDA benchmark and optional Nsight Compute profiling."
    )
    parser.add_argument("--binary", type=Path, default=Path(DEFAULT_BINARY))
    parser.add_argument(
        "--mesh",
        action="append",
        default=[],
        help="Mesh file to benchmark. Can be passed multiple times.",
    )
    parser.add_argument(
        "--mesh-glob",
        action="append",
        default=[],
        help="Glob for mesh files to benchmark, e.g. 'msh/dx_sweep/*_voxel_domain.bin'.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--output-interval", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--outlet-kernel", default="zhou_he")
    parser.add_argument(
        "--bytes-per-cell-step",
        type=float,
        default=DEFAULT_BYTES_PER_CELL_STEP,
        help="Model bytes moved per active cell per step for estimated bandwidth.",
    )
    parser.add_argument("--results-csv", default="output/cuda_benchmark_results.csv")
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--block-size",
        action="append",
        type=int,
        default=[],
        help=(
            "CUDA 1D thread block size to compile and benchmark. "
            "Can be passed multiple times; requires --build for sweeps."
        ),
    )
    parser.add_argument("--ncu", action="store_true", help="Collect Nsight Compute metrics.")
    parser.add_argument("--roofline", action="store_true", help="Write an Nsight roofline report.")
    parser.add_argument(
        "--performance-computation",
        action="store_true",
        help="Ask the executable to write per-LBM-kernel timing to output/kernel_profile.csv.",
    )
    parser.add_argument("--ncu-steps", type=int, default=0)
    parser.add_argument("--roofline-report", default="output/lbm_roofline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps must be non-negative")

    block_sizes = args.block_size or [256]
    for block_size in block_sizes:
        if block_size <= 0 or block_size > 1024:
            raise SystemExit("--block-size must be in the range 1..1024")

    if args.block_size and not args.build:
        raise SystemExit("--block-size requires --build so the binary is recompiled with that value")

    meshes = resolve_meshes(args)
    print("Benchmark meshes:")
    for mesh in meshes:
        print(f"  {mesh}")

    print("CUDA block sizes:")
    for block_size in block_sizes:
        print(f"  {block_size}")

    for block_size in block_sizes:
        if args.build:
            build = run_command(["make", "-B", "build", f"BLOCK_SIZE={block_size}"], repo)
            if build.stdout:
                print(build.stdout, end="")
            if build.returncode != 0:
                raise SystemExit(build.returncode)

        for mesh in meshes:
            result = run_benchmark(args, repo, mesh, block_size)

            if args.ncu:
                ncu_summary = run_ncu_metrics(args, repo, mesh)
                result.update(ncu_summary)

            write_csv_row(repo / args.results_csv, result)

            if args.roofline:
                run_roofline(args, repo, mesh)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
