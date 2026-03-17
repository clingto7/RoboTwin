# Work log

- Read context before coding:
  - `AGENTS.md`
  - `report/2026-03-12_0_pi0_pi05_data_selection_frame_metrics.md`
  - `report/coop/2026-03-12_0_pi0_pi05_data_selection_frame_metrics.md`
  - `report/2026-03-08_0_data_selection_baseline_integration.md`
  - `report/coop/2026-03-08_0_data_selection_baseline_integration.md`

# Code change (corrected design)

- Reverted `policy/pi0/process_data_pi0.sh` to simple passthrough wrapper behavior.
- Implemented new argument in `policy/pi0/scripts/process_data.py`:
  - `--selector-process-all-subsets [true|false]` (default `false`; `--selector-process-all-subsets` alone also means true)
- New processing behavior in Python:
  - One call to `select_with_data_selection(...)` generates all subsets from one run and writes one shared `selection_summary.csv`.
  - If `--selector-process-all-subsets true`, script iterates all generated subsets and runs `data_transform(...)` for each subset.
  - If false, keeps existing single-subset behavior via `--selector-subset-index`.
- Output naming for all-subset mode:
  - Uses per-seed tags (`sel<seed>`) so each processed dataset is separate and traceable for later metric-vs-result comparison.

# Validation

- LSP diagnostics (error level) passed:
  - `policy/pi0/scripts/process_data.py`
- Syntax checks passed:
  - `python3 -m py_compile policy/pi0/scripts/process_data.py`
  - `bash -n policy/pi0/process_data_pi0.sh`
- CLI help check passed and includes new arg:
  - `uv run python scripts/process_data.py --help`

# Usage example

- Process all subsets from one shared selection run:
  - `uv run bash process_data_pi0.sh click_bell demo_randomized 100 --selector-strategy random --selector-metric-level frame --pull-randomized-500 --selector-n-subsets 5 --selector-process-all-subsets true`
