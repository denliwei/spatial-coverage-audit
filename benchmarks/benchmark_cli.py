"""Quantitative, repeatable subprocess benchmark for the public audit CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="Measured runs per case")
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured runs per case")
    parser.add_argument("--synthetic-rows", type=int, default=100_000)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_predictions(rows: int) -> pd.DataFrame:
    if rows < 60:
        raise ValueError("synthetic-rows must be at least 60")
    index = np.arange(rows, dtype=np.int64)
    pair = index // 2
    design = np.where(index % 2 == 0, "random_marginal", "spatial_target_holdout")
    method_labels = np.array(["method_a", "method_b", "method_c"], dtype=object)
    domain_labels = np.array([f"D{value:02d}" for value in range(20)], dtype=object)
    method = method_labels[(pair // 5) % len(method_labels)]
    repeat = pair % 5
    domain = domain_labels[(pair // 15) % len(domain_labels)]
    y_true = (index % 1000).astype(float) / 10.0
    residual = ((index % 13).astype(float) - 6.0) / 20.0
    y_pred = y_true + residual
    base_half_width = np.where(index % 2 == 0, 0.55, 0.45)
    half_width = base_half_width + (index % 3).astype(float) / 20.0
    return pd.DataFrame(
        {
            "row_id": [f"s{value:06d}" for value in index],
            "method": method,
            "repeat": repeat,
            "design": design,
            "domain": domain,
            "y_true": y_true,
            "y_pred": y_pred,
            "lower": y_pred - half_width,
            "upper": y_pred + half_width,
        }
    )


def _base_command(predictions: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "spatial_coverage_audit",
        "--predictions",
        str(predictions),
        "--output-dir",
        str(output_dir),
        "--group-col",
        "domain",
        "--method-col",
        "method",
        "--repeat-col",
        "repeat",
        "--design-col",
        "design",
        "--pair-cols",
        "method,repeat",
        "--id-col",
        "row_id",
        "--alpha",
        "0.10",
    ]


def _golden_command(output_dir: Path) -> list[str]:
    return [
        *_base_command(ROOT / "examples" / "golden_predictions.csv", output_dir),
        "--calibration",
        str(ROOT / "examples" / "golden_calibration.csv"),
        "--score-col",
        "score",
        "--calibration-weight-col",
        "weight",
        "--target-weight-col",
        "target_weight",
    ]


def _csv_bundle_sha256(output_dir: Path) -> tuple[str, list[str]]:
    paths = sorted(output_dir.glob("*.csv"), key=lambda item: item.name)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), [path.name for path in paths]


def _normalized_json_sha256(result_path: Path) -> str:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["prediction_input"] = Path(result["prediction_input"]).name
    if result["calibration"] is not None:
        result["calibration"]["input"] = Path(result["calibration"]["input"]).name
    result["artifacts"] = {
        key: Path(value).name for key, value in result["artifacts"].items()
    }
    canonical = json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _run(command: list[str], output_dir: Path, env: dict[str, str]) -> tuple[float, dict[str, Any]]:
    start = time.perf_counter_ns()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark CLI failed with code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    status = json.loads(completed.stdout)
    if status.get("status") != "passed":
        raise RuntimeError(f"unexpected CLI status: {status}")
    csv_hash, csv_files = _csv_bundle_sha256(output_dir)
    identity = {
        "csv_bundle_sha256": csv_hash,
        "normalized_json_sha256": _normalized_json_sha256(output_dir / "audit_result.json"),
        "csv_artifacts": csv_files,
    }
    return elapsed_ms, identity


def _case(
    *,
    label: str,
    rows: int,
    input_paths: dict[str, Path],
    command_factory: Callable[[Path], list[str]],
    runs: int,
    warmup: int,
    root: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    timings: list[float] = []
    identities: list[dict[str, Any]] = []
    for iteration in range(warmup + runs):
        output_dir = root / f"{label}-{iteration:02d}"
        elapsed_ms, identity = _run(command_factory(output_dir), output_dir, env)
        if iteration >= warmup:
            timings.append(elapsed_ms)
            identities.append(identity)
    first_identity = identities[0]
    identical = all(identity == first_identity for identity in identities[1:])
    if not identical:
        raise RuntimeError(f"{label} outputs changed across repeated runs")
    return {
        "case": label,
        "rows": rows,
        "input_sha256": {
            name: _sha256(path) for name, path in sorted(input_paths.items())
        },
        "runtime_ms": {
            "runs": [round(value, 3) for value in timings],
            "min": round(min(timings), 3),
            "median": round(statistics.median(timings), 3),
            "max": round(max(timings), 3),
        },
        "output_identity": {**first_identity, "identical_across_runs": identical},
    }


def main() -> None:
    args = _parser().parse_args()
    if args.runs <= 0 or args.warmup < 0:
        raise ValueError("runs must be positive and warmup must be non-negative")
    env = os.environ.copy()
    source_path = str(ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_path, env.get("PYTHONPATH", "")) if item
    )
    with tempfile.TemporaryDirectory(prefix="spatial-coverage-benchmark-") as temporary:
        benchmark_root = Path(temporary)
        synthetic_path = benchmark_root / "synthetic_predictions.csv"
        synthetic = _synthetic_predictions(args.synthetic_rows)
        synthetic.to_csv(
            synthetic_path,
            index=False,
            lineterminator="\n",
            float_format="%.6f",
        )
        cases = [
            _case(
                label="golden",
                rows=16,
                input_paths={
                    "calibration": ROOT / "examples" / "golden_calibration.csv",
                    "predictions": ROOT / "examples" / "golden_predictions.csv",
                },
                command_factory=_golden_command,
                runs=args.runs,
                warmup=args.warmup,
                root=benchmark_root,
                env=env,
            ),
            _case(
                label="synthetic",
                rows=args.synthetic_rows,
                input_paths={"predictions": synthetic_path},
                command_factory=lambda output: _base_command(synthetic_path, output),
                runs=args.runs,
                warmup=args.warmup,
                root=benchmark_root,
                env=env,
            ),
        ]
    report = {
        "benchmark_schema": 1,
        "measurement": (
            "Subprocess wall time in milliseconds; includes interpreter startup, "
            "CSV input/output, and audit computation; excludes synthetic input generation."
        ),
        "configuration": {
            "measured_runs_per_case": args.runs,
            "warmup_runs_per_case": args.warmup,
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "cases": cases,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
