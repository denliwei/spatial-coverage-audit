# spatial-coverage-audit

Small, deterministic utilities for auditing prediction intervals under random,
grouped, domain-holdout, weighted-conformal, and target-recalibration designs.
The package is an extraction of the paper's verified audit semantics; it does
not fit a prediction model and makes no exchangeability claim for spatial
holdouts.

## Frozen semantics

- Coverage is inclusive: `lower <= y_true <= upper`.
- The interval score is the central `(1-alpha)` interval score.
- Wilson intervals use `z=1.959963984540054`.
- Split-conformal rank is `ceil((n+1)*(1-alpha))`; a rank above `n` returns
  infinity rather than clipping to the largest score.
- Target-point weighted conformal includes the target weight as mass at
  infinity. Unbounded intervals remain infinite in overall width and interval
  score; finite-only summaries are separately labelled.
- Macro metrics give each observed group equal weight. Micro metrics pool rows.
- Few-shot recalibration uses target labels and is never a zero-sample
  transport result.
- A finite-sample rank above the calibration count is a valid, explicitly
  unbounded conformal result (`valid_unbounded_finite_sample_rank`), not an
  invalid-input rejection. Invalid alpha values, dimensions, and all-nonfinite
  score vectors raise `ValueError` instead. Calibration score vectors must be
  entirely finite; rows are not silently removed. Supplied CQR score columns may be
  negative and are never clipped at zero.
- Weighted calibration rejects negative or non-finite source weights. Zero
  weights are explicitly ignored and their count is reported alongside the
  positive-weight count.
- Any declared group, method, design, repeat, or membership-ledger run key must
  be present and nonmissing; missing keys are invalid inputs rather than an
  implicit `NaN` group.

## Install and test

For the pinned reference environment used by the release checks and benchmark:

```shell
conda env create -f environment.yml
conda activate spatial-coverage-audit-0.1.0
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
```

Without conda, create a Python 3.14.4 virtual environment and install
`requirements.txt` before installing the package. The lower bounds in
`pyproject.toml` remain the package compatibility contract; the exact pins are
the reproducibility reference.

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```

For an offline, no-install run when NumPy and pandas are already present:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m spatial_coverage_audit --help
python -m unittest discover -s tests -v
```

Run the checked golden example as a real CLI subprocess with one command:

```shell
python scripts/run_golden_integration.py
```

Refresh the machine-readable test/golden/benchmark-reference record with:

```shell
python scripts/write_release_verification.py
```

This runs the suite with warnings promoted to errors and verifies the frozen
CSV identity. It references, but deliberately does not overwrite, the dated
machine-dependent timing provenance in `benchmarks/BENCHMARK_REPORT.md`.

It verifies scientific values, the five expected CSV artifacts, and their
frozen byte-level SHA-256 identity. The inputs and their design are documented
in `examples/README.md`.

## CLI

The prediction CSV needs finite truth values and interval endpoints (endpoints
may be infinite). Column names are configurable.

```powershell
spatial-coverage-audit `
  --predictions predictions.csv `
  --output-dir audit_output `
  --y-col y_true --lower-col lower --upper-col upper `
  --point-col y_pred --group-col domainID --method-col method `
  --repeat-col repeat --alpha 0.10
```

Add `--calibration calibration.csv --score-col score` to audit the exact
split-conformal quantile. Add `--calibration-weight-col weight` and
`--target-weight-col target_weight` for target-point weighted quantiles.
`--fewshot-sizes 10 20 --fewshot-repeats 100 --id-col plotID` writes a
deterministic target-calibration membership ledger and run-level quantiles.

Use `--membership-ledger membership.csv --membership-run-cols method,repeat,fold`
to validate fit/calibration/test membership. Add
`--complete-group-holdout --membership-group-col domain` to detect target-group
leakage into fitting or calibration. Duplicate evaluation IDs and cross-split
ID overlap are recorded as scientific contract failures while `run_status`
remains `completed`; malformed schemas remain run errors.

The bounded fitness engine is configured with `--fitness-min-picp`,
`--fitness-warning-picp`, the corresponding worst-group options, and warning,
failure, or unusable thresholds for unbounded-output rates. Its four states are
`no_failure_detected`, `warning`, `failure_detected`, and
`operationally_unusable`. These four action states are emitted only when the
user supplies at least one fitness criterion and the prediction identifiers,
fit/calibration/test ledger, complete-group leakage audit, and evaluation-row
alignment are all verified. Otherwise the result is `state=not_assessed` and
`mode=metric_only`.

Once scientific-fitness assessment is enabled, the default maximum usable
unbounded-output rate is zero, because any unbounded interval makes the overall
mean width and score infinite. An explicit
`--fitness-unusable-unbounded-rate` override is recorded with
`unbounded_policy.source=user_override`. `status=passed` is retained as a compatibility alias
for successful execution and must not be interpreted as scientific fitness.
Use `--include-rmse` to add RMSE to point, group, aggregate, and paired-contrast
tables without changing the frozen default output schema.

Main outputs are CSV tables plus `audit_result.json`. JSON encodes non-finite
values as `null`; the CSV tables preserve `inf` so unbounded intervals cannot
be mistaken for sharp finite intervals.
`fitness_state.json` separates `run_status` from `scientific_fitness`; a
membership issues CSV is added only when a membership ledger is supplied.
Without user thresholds or a fully verified membership contract,
`assessment_status=not_assessed`, `mode=metric_only`, and
`full_contract_verified=false`. The CLI stdout repeats those fields and never
prints `no_failure_detected` for an unassessed run.

## Python API

```python
from spatial_coverage_audit import (
    exact_split_conformal_quantile,
    FitnessCriteria,
    FitnessObservation,
    assess_fitness,
    validate_membership_ledger,
    validate_prediction_evaluation_alignment,
    summarize_intervals,
    target_point_weighted_quantiles,
)

q = exact_split_conformal_quantile([0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6], 0.10)
summary = summarize_intervals(y_true, lower, upper, alpha=0.10)
weighted = target_point_weighted_quantiles(scores, calibration_weights, target_weights, 0.10)
print(weighted.unbounded_rate)

membership_check = validate_membership_ledger(
    membership_frame, run_cols=["fold"], group_col="domain",
    complete_group_holdout=True,
)
alignment = validate_prediction_evaluation_alignment(
    predictions, membership_frame, run_cols=["fold"]
)
full_contract_verified = membership_check.valid and alignment["status"] == "passed"
fitness = assess_fitness(
    FitnessObservation(min_picp=0.89, max_unbounded_rate=0.0),
    FitnessCriteria(min_picp_failure=0.85, min_picp_warning=0.90),
    assessment_enabled=full_contract_verified,
)
print(membership_check.valid, fitness.to_dict()["state"])
```

The CLI audits supplied predictions; it does not certify conditional coverage,
repair distribution shift, or turn few-shot target calibration into a
zero-label guarantee.

## Reproducibility and release metadata

`benchmarks/benchmark_cli.py` benchmarks both the golden input and a
deterministically generated 100,000-row case. The measured reference results,
environment, runtime definition, and output hashes are recorded in
`benchmarks/BENCHMARK_REPORT.md`.

The code is released under the MIT License. Contributor-approved identities
are recorded in `AUTHORS.md` and `CITATION.cff`. The citation metadata omits a
DOI until the archival release has actually been registered.
