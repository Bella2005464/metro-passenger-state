# metro-passenger-state

Reliability-aware reconstruction of metro passenger states from boarding and
alighting counts under incomplete fare sensing. It is the reference
implementation of the method in the companion paper.

Metro systems that charge fares only at entry (or where transfers occur inside
fare-control zones) observe where passengers board but not where they go: the
destination of a transfer trip and the interchange it passes through are
unobserved. This package reconstructs line-level origin–destination flow from
the observable counts and reports a validated reliability for every
interface-level estimate, so downstream decisions can accept, escalate, or
observe rather than trust point estimates blindly.

## The three layers

| Module | Layer | What it delivers |
| --- | --- | --- |
| `passenger_state.network` | network | curated 14-line membership, weighted track graph, logit route choice over downstream interchanges |
| `passenger_state.reconstruction` | Layer 1–2 | closed-form line OD from stable destination shares; conjugate Dirichlet/Beta posteriors with between-day over-dispersion (`phi`/`psi`) |
| `passenger_state.reliability` | Layer 3 | per-interchange transfer-flow posteriors marginalized over the unidentifiable route-choice dispersion `eta`; mass-conserving coupling projection and network assembly |
| `passenger_state.validation` | validation | shared-mechanism and mechanism-divergent synthetic worlds, coverage metrics, Spearman rank statistics |
| `passenger_state.data` | data | DuckDB canonicalization of the raw MetroFlow CSVs into date-partitioned parquet, split policy, readers |

## Installation

Requires Python 3.10+.

```bash
pip install .
```

NumPy, DuckDB and PyArrow are installed as dependencies; there is no compiled
or GPU component. For development, use `pip install -e .`.

## Data

The pipeline consumes the open **MetroFlow** release (Sun et al., *Scientific
Data*, 2025) for the Shanghai metro: `stationInfo.csv`,
`metroData_InOutFlow.csv` (per-slot station entry/exit counts) and
`metroData_ODFlow.csv` (per-slot non-diagonal OD flow). The demo fixture under
`tests/fixtures/metroflow_tiny/` contains a synthetic 3-station dataset in
exactly this format, so the raw-CSV path can be exercised without downloading
or redistributing the full release.

`passenger_state.data.canonicalize` streams the three CSVs into a
date-partitioned parquet dataset (station crosswalk, observed flow, OD labels
by train/validation/test split, the derived self-loop residual, and the
designated stress dates), built in a staging directory and moved into place
atomically.

## Demo

```bash
python scripts/run_demo.py
```

The demo canonicalizes the bundled fixture and runs the full pipeline: the
closed-form line reconstruction (Layer 1), the between-day over-dispersion
calibration `phi` fitted on simulated training days and shown to restore
interval coverage (Layer 2), and on a small 7-station example network the
eta-marginalized transfer posterior, the gravity mechanism-divergence check,
and the mass-conserving coupling projection (Layer 3). The script ends by
printing the four check results.

## Tests

```bash
python -m unittest tests.test_smoke
```

Nine small assertions covering mass conservation, the Layer-1 row constraint,
moment matching at both extremes, eta-marginalization with non-uniform
weights, the split policy, and a canonicalize/read-back roundtrip on the
fixture; runs in a few seconds.

## Reproducibility scope

This repository provides the reference implementation, a runnable end-to-end
demonstration, and automated tests for the core reconstruction and reliability
calculations. The full MetroFlow records are not redistributed. Paper-scale
experiments require obtaining the source data separately and canonicalizing
them with `passenger_state.data`.

## Citation

Citation metadata for the software are provided in `CITATION.cff`. When using
the software in academic work, please cite both the software release and the
companion paper.

## License

MIT; see [LICENSE](LICENSE).
