import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseAssetTests(unittest.TestCase):
    def test_machine_readable_release_verification(self):
        verification = json.loads(
            (ROOT / "release_verification.json").read_text(encoding="utf-8")
        )
        self.assertEqual(verification["test_suite"]["status"], "passed")
        self.assertTrue(verification["test_suite"]["warnings_as_errors"])
        self.assertGreaterEqual(verification["test_suite"]["tests"], 37)
        self.assertEqual(
            verification["golden_integration"]["csv_bundle_sha256"],
            "f631d27da21308209519e557adfbf090e5353ebbf7b998b440465224ef421942",
        )
        benchmark = ROOT / verification["benchmark_reference"]["report"]
        self.assertTrue(benchmark.is_file())
        self.assertEqual(
            verification["benchmark_reference"]["reference_results"]["golden"][
                "median_ms"
            ],
            584.787,
        )

    def test_license_citation_and_environment_are_explicit(self):
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertTrue(license_text.startswith("MIT License\n"))
        self.assertIn("Spatial Coverage Audit contributors", license_text)
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('license = "MIT"', pyproject)
        self.assertIn('license-files = ["LICENSE"]', pyproject)

        citation_lines = (ROOT / "CITATION.cff").read_text(
            encoding="utf-8"
        ).splitlines()
        metadata_lines = [
            line for line in citation_lines if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertFalse(any(line.lstrip().startswith("doi:") for line in metadata_lines))
        self.assertFalse(any("<" in line or ">" in line for line in metadata_lines))
        citation_text = "\n".join(metadata_lines)
        self.assertIn('family-names: "Deng"', citation_text)
        self.assertIn('family-names: "Wang"', citation_text)
        self.assertIn('family-names: "Kuang"', citation_text)
        authors_text = (ROOT / "AUTHORS.md").read_text(encoding="utf-8")
        self.assertIn("t20050522@csuft.edu.cn", authors_text)
        zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
        self.assertEqual(zenodo["version"], "0.1.0")
        self.assertEqual(len(zenodo["creators"]), 3)

        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("numpy==2.5.1", requirements)
        self.assertIn("pandas==3.0.3", requirements)
        environment = (ROOT / "environment.yml").read_text(encoding="utf-8")
        self.assertIn("python=3.14.4", environment)
        self.assertIn("numpy==2.5.1", environment)
        self.assertIn("pandas==3.0.3", environment)

    def test_one_command_golden_cli_integration(self):
        completed = subprocess.run(
            [sys.executable, "scripts/run_golden_integration.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("golden CLI integration: PASS", completed.stdout)
        self.assertIn(
            "f631d27da21308209519e557adfbf090e5353ebbf7b998b440465224ef421942",
            completed.stdout,
        )

    def test_benchmark_smoke_outputs_are_identical(self):
        completed = subprocess.run(
            [
                sys.executable,
                "benchmarks/benchmark_cli.py",
                "--runs",
                "2",
                "--warmup",
                "0",
                "--synthetic-rows",
                "600",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        report = json.loads(completed.stdout)
        self.assertEqual([case["rows"] for case in report["cases"]], [16, 600])
        self.assertEqual(
            sorted(report["cases"][0]["input_sha256"]),
            ["calibration", "predictions"],
        )
        for case in report["cases"]:
            self.assertTrue(case["output_identity"]["identical_across_runs"])
            self.assertEqual(len(case["runtime_ms"]["runs"]), 2)


if __name__ == "__main__":
    unittest.main()
