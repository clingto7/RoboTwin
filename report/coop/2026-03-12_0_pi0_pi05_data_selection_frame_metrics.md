## Handoff for teammates

### What was done

- Integrated `pi0` and `pi05` preprocess pipelines with `script/data_selection` (no longer limited to first-N sequential episode selection).
- Added support for registered selector workflow directly inside `process_data.py`:
  - select subsets via `--selector-strategy`
  - generate multi-subset runs via `--selector-n-subsets`
  - choose concrete subset via `--selector-subset-index`
- Added frame-level metric option for subset evaluation:
  - `--selector-metric-level frame`
  - frame source camera defaults to `head_camera` and falls back safely.

### Key files

- Data selection core:
  - `script/data_selection/feature_selector.py`
  - `script/data_selection/__init__.py`
  - `script/select_training_data.py`
- pi preprocess integration:
  - `policy/pi0/scripts/process_data.py`
  - `policy/pi05/scripts/process_data.py`
  - `policy/pi0/process_data_pi0.sh`
  - `policy/pi05/process_data_pi05.sh`

### Behavior changes

- New selector registered: `random_subsets` (alias behavior for random selection; old selectors preserved).
- New metrics added in `METRIC_REGISTRY`:
  - `feature_mean`, `feature_std`, `feature_var`
- Frame-level feature extractor added:
  - `build_frame_embeddings(...)`
- `sample_frames=1` edge case fixed in frame sampling helper.

### 500-episode pull support

- `pi0/pi05` process_data now supports pulling randomized 500 dataset when local episodes are missing:
  - `--pull-randomized-500`
  - `--hf-repo-id` (default `TianxingChen/RoboTwin2.0`)
  - `--hf-embodiment` (default `aloha-agilex`)

### Verified

- LSP diagnostics (error-level) clean for modified Python files.
- `py_compile` pass for all modified Python files.
- CLI help checks pass for updated scripts.
- Runtime smoke checks pass for:
  - frame-level feature extraction path
  - pi0 selector-integrated process_data
  - pi05 selector-integrated process_data

### Useful command templates

- Generate 5 random subsets of 100 from 500 (frame metrics):
  - `uv run python script/select_training_data.py <task> demo_randomized 100 --strategy random_subsets --n-subsets 5 --metric-level frame --frame-camera head_camera --output-dir <out>`

- pi0 process_data using integrated selector flow (choose subset index):
  - `uv run python policy/pi0/scripts/process_data.py <task> demo_randomized 100 --selector-strategy random_subsets --selector-n-subsets 5 --selector-subset-index 0 --selector-metric-level frame --selector-output-dir <out> --subset-tag sel0`

- pi05 process_data same pattern:
  - `uv run python policy/pi05/scripts/process_data.py <task> demo_randomized 100 --selector-strategy random_subsets --selector-n-subsets 5 --selector-subset-index 0 --selector-metric-level frame --selector-output-dir <out> --subset-tag sel0`
