## Handoff for teammates

### Goal completed

- Moved all-subset processing control into pi0 Python script so subset generation + metrics CSV remain in one shared run context.

### Changed file

- `policy/pi0/scripts/process_data.py`
- `policy/pi0/process_data_pi0.sh` (reverted to passthrough)

### New argument

- `--selector-process-all-subsets true|false`
  - Implemented in `policy/pi0/scripts/process_data.py`.
  - Default is `false` (old single-subset behavior).
  - `true` means: process all subsets generated in the same selection run.

### Behavior details

- Selection and metrics are computed once per command:
  - `select_with_data_selection(...)` still writes `selection_summary.csv` + `selection_*.json` for all subsets.
- Single subset mode (`false`):
  - unchanged, still uses `--selector-subset-index`.
- All subset mode (`true`):
  - iterates all generated `(seed, selected_episode_ids)` pairs from that same run,
  - runs `data_transform(...)` for each,
  - writes separate processed outputs with seed-based tags (`sel<seed>`) to avoid overwrite and keep experiment-metric mapping clear.

### Validation

- LSP clean (errors): `policy/pi0/scripts/process_data.py`
- `python3 -m py_compile policy/pi0/scripts/process_data.py` passed.
- `bash -n policy/pi0/process_data_pi0.sh` passed.
- `uv run python scripts/process_data.py --help` passed and shows `--selector-process-all-subsets`.

### Command example

- `uv run bash process_data_pi0.sh click_bell demo_randomized 100 --selector-strategy random --selector-metric-level frame --pull-randomized-500 --selector-n-subsets 5 --selector-process-all-subsets true`
