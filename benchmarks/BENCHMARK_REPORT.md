# CLI benchmark report

Benchmark date: 2026-07-16 (Asia/Taipei)

## Scope and method

The benchmark invokes the public `python -m spatial_coverage_audit` CLI in a
fresh subprocess. Each case received one unmeasured warm-up followed by five
measured runs. Wall time includes interpreter startup, CSV reading, audit
computation, and CSV/JSON writing; deterministic synthetic-input generation is
excluded. Output identity is checked after every run by hashing the ordered CSV
artifact names and exact bytes, plus a canonical JSON result with temporary
paths reduced to basenames.

Reproduce it from the package root with:

```shell
python benchmarks/benchmark_cli.py --runs 5 --warmup 1 --synthetic-rows 100000
```

## Reference environment

- CPython 3.14.4
- NumPy 2.5.1
- pandas 3.0.3
- Windows 11 (`Windows-11-10.0.26200-SP0`)
- Intel Core Ultra X7 358H, 16 cores / 16 logical processors
- 33,873,752,064 bytes physical memory (31.55 GiB)

The dependency versions match `requirements.txt` and `environment.yml`.

## Results

| Case | Rows | Run times (ms) | Min (ms) | Median (ms) | Max (ms) | Identical outputs |
|---|---:|---|---:|---:|---:|---|
| Golden | 16 | 584.787, 587.794, 606.146, 572.437, 552.921 | 552.921 | 584.787 | 606.146 | Yes, 5/5 |
| Synthetic | 100,000 | 827.798, 822.143, 806.996, 855.270, 818.866 | 806.996 | 822.143 | 855.270 | Yes, 5/5 |

The golden case includes group aggregation, paired random-versus-spatial
contrasts, exact split conformal calibration, and target-point weighted
quantiles. The synthetic case includes 100,000 rows, three methods, five
repeats, two designs, and 20 domains.

## Input and output identities

| Case | Prediction CSV SHA-256 | Calibration CSV SHA-256 | CSV artifact bundle SHA-256 | Normalized JSON SHA-256 |
|---|---|---|---|---|
| Golden | `129877bd6337d02ce4423814ecbe3d276b01d46a8bfc56fd3b84315627af721e` | `cacfba03bc09215efe47c655ba3b21b239cd64845da8c45832bc222e979ff848` | `f631d27da21308209519e557adfbf090e5353ebbf7b998b440465224ef421942` | `90863a69bb0341fb7377ada8980605605624043cdeda3812bfa9a816d2dc5339` |
| Synthetic | `44f6b0e4914ce1abf464c806caeeac25f5698eb13725cc6ad1c13f045221ce6a` | Not used | `57a9c77cb2d405cbd92bd4ca9999cd2f59c7b5b80b5c4f1bd6da5fe3204bf60a` | `d99b5c0f754f9058ee197ac1a5b4e685e0aaa6345b89acfd1477d7e1183f30f9` |

The absolute `audit_result.json` paths are deliberately normalized before
hashing; scientific values, input hashes, limitations, and artifact names
remain covered. Timing is machine- and load-dependent, while the identity
digests are expected to remain fixed under the pinned reference environment.
