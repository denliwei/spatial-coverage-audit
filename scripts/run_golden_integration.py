"""Run and verify the public CLI against the deterministic golden CSV inputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CSV_BUNDLE_SHA256 = "f631d27da21308209519e557adfbf090e5353ebbf7b998b440465224ef421942"
EXPECTED_CSV_FILES = {
    "group_aggregate_summary.csv",
    "group_summary.csv",
    "overall_summary.csv",
    "random_vs_group_contrasts.csv",
    "target_weighted_quantiles.csv",
}


def _csv_bundle_sha256(output_dir: Path) -> str:
    """Hash artifact names and exact bytes in a platform-independent order."""

    digest = hashlib.sha256()
    for path in sorted(output_dir.glob("*.csv"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _command(output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "spatial_coverage_audit",
        "--predictions",
        str(ROOT / "examples" / "golden_predictions.csv"),
        "--calibration",
        str(ROOT / "examples" / "golden_calibration.csv"),
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
        "--score-col",
        "score",
        "--calibration-weight-col",
        "weight",
        "--target-weight-col",
        "target_weight",
        "--id-col",
        "row_id",
        "--alpha",
        "0.10",
    ]


def main() -> None:
    env = os.environ.copy()
    source_path = str(ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_path, env.get("PYTHONPATH", "")) if item
    )
    with tempfile.TemporaryDirectory(prefix="spatial-coverage-golden-") as temporary:
        output_dir = Path(temporary) / "audit"
        completed = subprocess.run(
            _command(output_dir),
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "golden CLI failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        status = json.loads(completed.stdout)
        if status.get("status") != "passed":
            raise AssertionError(f"unexpected CLI status: {status}")

        result_path = output_dir / "audit_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["status"] != "passed" or result["prediction_rows"] != 16:
            raise AssertionError("golden result status or row count changed")
        calibration = result["calibration"]
        if not math.isclose(calibration["exact_q"], 0.9, rel_tol=0.0, abs_tol=1e-15):
            raise AssertionError(f"exact conformal quantile changed: {calibration['exact_q']}")
        if calibration["weighted"]["unbounded_n"] != 2:
            raise AssertionError("weighted unbounded count changed")

        actual_files = {path.name for path in output_dir.glob("*.csv")}
        if actual_files != EXPECTED_CSV_FILES:
            raise AssertionError(
                f"CSV artifact set changed: expected {sorted(EXPECTED_CSV_FILES)}, "
                f"got {sorted(actual_files)}"
            )
        overall = _read_csv(output_dir / "overall_summary.csv")
        groups = _read_csv(output_dir / "group_summary.csv")
        contrasts = _read_csv(output_dir / "random_vs_group_contrasts.csv")
        weighted = _read_csv(output_dir / "target_weighted_quantiles.csv")
        if (len(overall), len(groups), len(contrasts), len(weighted)) != (8, 16, 4, 16):
            raise AssertionError("golden artifact dimensions changed")
        baseline = next(
            row
            for row in contrasts
            if row["method"] == "baseline" and row["repeat"] == "0"
        )
        if not math.isclose(
            float(baseline["random_minus_group_picp"]),
            0.5,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise AssertionError("golden random-versus-spatial coverage contrast changed")

        identity = _csv_bundle_sha256(output_dir)
        if identity != EXPECTED_CSV_BUNDLE_SHA256:
            raise AssertionError(
                "golden CSV identity changed: "
                f"expected {EXPECTED_CSV_BUNDLE_SHA256}, got {identity}"
            )
        print(
            "golden CLI integration: PASS "
            f"(16 rows, 5 CSV artifacts, sha256={identity})"
        )


if __name__ == "__main__":
    main()
