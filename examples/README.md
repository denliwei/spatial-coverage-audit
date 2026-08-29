# Deterministic golden example

`golden_predictions.csv` is a 16-row, two-method interval-audit input with
paired random and spatial designs, two repeats, and two domains. Two target
rows deliberately have enough target weight to make their weighted conformal
quantile unbounded. `golden_calibration.csv` contains nine finite calibration
scores with unit weights, so the exact split-conformal quantile at
`alpha = 0.10` is `0.9`.

Run the complete example and verify its checked output identity with:

```shell
python scripts/run_golden_integration.py
```

The command runs the public module CLI in a subprocess, checks all expected
artifacts and selected scientific values, and compares the CSV artifact bundle
against a frozen SHA-256 digest.
