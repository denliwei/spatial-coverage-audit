"""Run release checks and write a concise machine-readable verification record."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release_verification.json"
GOLDEN_HASH = "f631d27da21308209519e557adfbf090e5353ebbf7b998b440465224ef421942"


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(ROOT / "src"), env.get("PYTHONPATH", "")) if item
    )
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _benchmark_results(report_text: str) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for line in report_text.splitlines():
        if not (line.startswith("| Golden |") or line.startswith("| Synthetic |")):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 7:
            continue
        label, rows, runs, minimum, median, maximum, identity = cells
        results[label.lower()] = {
            "rows": int(rows.replace(",", "")),
            "run_times_ms": [float(value.strip()) for value in runs.split(",")],
            "min_ms": float(minimum),
            "median_ms": float(median),
            "max_ms": float(maximum),
            "identical_outputs": identity,
        }
    if set(results) != {"golden", "synthetic"}:
        raise RuntimeError("could not parse benchmark reference results")
    return results


def main() -> None:
    test_command = [
        sys.executable,
        "-W",
        "error",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-v",
    ]
    tests = _run(test_command)
    test_log = tests.stdout + "\n" + tests.stderr
    match = re.search(r"Ran (\d+) tests? in ([0-9.]+)s", test_log)
    if tests.returncode or not match or "\nOK\n" not in test_log:
        raise RuntimeError(f"release tests failed or could not be parsed:\n{test_log}")

    golden_command = [sys.executable, "scripts/run_golden_integration.py"]
    golden = _run(golden_command)
    if golden.returncode or GOLDEN_HASH not in golden.stdout:
        raise RuntimeError(
            "golden integration failed:\n"
            f"stdout:\n{golden.stdout}\nstderr:\n{golden.stderr}"
        )

    benchmark_path = ROOT / "benchmarks" / "BENCHMARK_REPORT.md"
    benchmark_text = benchmark_path.read_text(encoding="utf-8")
    report = {
        "schema_version": 1,
        "test_suite": {
            "status": "passed",
            "tests": int(match.group(1)),
            "elapsed_seconds": float(match.group(2)),
            "warnings_as_errors": True,
            "command": "python -W error -m unittest discover -s tests -v",
        },
        "golden_integration": {
            "status": "passed",
            "rows": 16,
            "csv_artifacts": 5,
            "csv_bundle_sha256": GOLDEN_HASH,
            "command": "python scripts/run_golden_integration.py",
        },
        "benchmark_reference": {
            "report": "benchmarks/BENCHMARK_REPORT.md",
            "report_sha256": _sha256(benchmark_path),
            "reference_results": _benchmark_results(benchmark_text),
            "measurement_date": "2026-07-16",
            "reference_command": (
                "python benchmarks/benchmark_cli.py --runs 5 --warmup 1 "
                "--synthetic-rows 100000"
            ),
            "provenance": (
                "Frozen measured timing report; release verification does not replace "
                "or silently remeasure its machine-dependent timings. The report's "
                "normalized JSON hashes predate additive audit_schema=2 fields; the "
                "frozen CSV bundle identity remains current and verified."
            ),
        },
    }
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
