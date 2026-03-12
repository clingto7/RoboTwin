# Work log

- Reviewed prior session docs before coding:
  - `report/coop/2026-03-08_0_data_selection_baseline_integration.md`
  - `report/2026-03-08_0_data_selection_baseline_integration.md`
  - `know-how.md`
- Verified current gap: `pi0/pi05` process_data still used default first-N sequential episodes and had no data_selection integration.

# Code changes

## 1) data_selection module extensions

- Updated `script/data_selection/feature_selector.py`:
  - Added frame-level feature extraction API: `build_frame_embeddings(...)`.
  - Added metric registration helper: `register_metric(...)`.
  - Added selector registration helper: `register_selector(...)`.
  - Added selector registration entry: `random_subsets` (keeps old selectors intact).
  - Added frame-level summary metrics:
    - `feature_mean`
    - `feature_std`
    - `feature_var`
  - Kept existing metrics/selectors:
    - metrics: `cosine_distance`, `l2_distance`, `variance_score`
    - selectors: `random`, `greedy_maxdist`, `greedy_maxvar`, `kmeans`
  - Added camera fallback selection (`head_camera`/`cam_high`/middle camera fallback).
  - Fixed frame sampling edge case when `sample_frames=1` to avoid divide-by-zero.

- Updated `script/data_selection/__init__.py` exports:
  - `build_frame_embeddings`, `register_metric`, `register_selector`

## 2) selection CLI updates

- Updated `script/select_training_data.py`:
  - New options:
    - `--metric-level {episode,frame}`
    - `--feature-batch-size`
    - `--frame-camera`
  - Supports frame-level subset metric computation:
    - episode selection still performed via registered selector on episode embeddings
    - subset metric can be computed on all frame-level features for selected episodes
  - Added CSV output fields:
    - `metric_level`, `feature_mean`, `feature_std`, `feature_var`

## 3) pi0 / pi05 process_data integration

- Rewrote:
  - `policy/pi0/scripts/process_data.py`
  - `policy/pi05/scripts/process_data.py`

- New capabilities:
  - Optional direct selection-file mode:
    - `--episode-ids-file`
  - Optional integrated data_selection mode:
    - `--selector-strategy` (supports all registered selectors, including `random_subsets`)
    - `--selector-n-subsets` (for 5-subset workflows)
    - `--selector-subset-index` (choose concrete subset)
    - `--selector-metric-level {episode,frame}`
    - `--selector-metric`
    - `--selector-model`
    - `--selector-sample-frames`
    - `--selector-batch-size`
    - `--selector-frame-camera`
    - `--selector-output-dir`
    - `--selector-seed`
    - `--selector-device`
  - Optional 500 randomized dataset pull:
    - `--pull-randomized-500`
    - `--hf-repo-id` (default `TianxingChen/RoboTwin2.0`)
    - `--hf-embodiment` (default `aloha-agilex`)
  - `--subset-tag` support for output directory naming
  - Robust instruction handling when `instructions/episode*.json` is absent (falls back to empty instruction list)

- Updated wrappers to forward optional args:
  - `policy/pi0/process_data_pi0.sh`
  - `policy/pi05/process_data_pi05.sh`

# Validation

- LSP diagnostics (errors only) on modified files: clean
  - `script/data_selection/feature_selector.py`
  - `script/select_training_data.py`
  - `policy/pi0/scripts/process_data.py`
  - `policy/pi05/scripts/process_data.py`

- Syntax compile passed:
  - `python3 -m py_compile script/data_selection/__init__.py script/data_selection/feature_selector.py script/select_training_data.py policy/pi0/scripts/process_data.py policy/pi05/scripts/process_data.py`

- CLI smoke checks passed:
  - `uv run python script/select_training_data.py --help`
  - `uv run python policy/pi0/scripts/process_data.py --help`
  - `uv run python policy/pi05/scripts/process_data.py --help`

- Runtime smoke checks passed:
  - Frame-level embedding + metric extraction:
    - `uv run python -c "... build_frame_embeddings(...) ..."`
  - pi0 integrated selector path:
    - `uv run python policy/pi0/scripts/process_data.py click_bell demo_clean 1 --selector-strategy random_subsets ...`
  - pi05 integrated selector path:
    - `uv run python policy/pi05/scripts/process_data.py click_bell demo_clean 1 --selector-strategy random_subsets ...`

# Notes

- Smoke tests produced temporary artifacts under:
  - `policy/pi0/selection_outputs_smoke/`
  - `policy/pi05/selection_outputs_smoke/`
  - `processed_data/`
