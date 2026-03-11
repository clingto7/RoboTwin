## Work log

- Completed environment init checks and dependency setup using UV:
  - `python3 -V`
  - `uv --version`
  - `uv venv --python 3.10 .venv`
  - `uv pip install -r script/requirements.txt`
  - `python script/update_embodiment_config_path.py`
- Reconciled current repo state and confirmed new baseline components already existed and were wired:
  - `script/data_selection/feature_selector.py`
  - `script/select_training_data.py`
  - `script/run_act_selection_experiments.py`
  - `policy/ACT/process_data.py`
  - `policy/DP/process_data.py`
  - `policy/DP3/scripts/process_data.py`
- Cleaned implementation details without changing behavior:
  - Removed unused imports/locals and string-format noise in modified scripts.
  - Kept backward compatibility: default first-N behavior still works when no selection file is provided.

## Validation

- LSP diagnostics: clean for modified Python files.
- Syntax validation:
  - `python3 -m py_compile` on all modified Python files passed.
- CLI argument validation in UV env (all passed):
  - `uv run python script/select_training_data.py --help`
  - `uv run python script/run_act_selection_experiments.py --help`
  - `uv run python policy/ACT/process_data.py --help`
  - `uv run python policy/DP/process_data.py --help`
  - `uv run python policy/DP3/scripts/process_data.py --help`

## Current constraints

- No local episode files found under `data/**/data/episode*.hdf5` in this workspace state.
- Because of missing local dataset episodes, end-to-end selection/training/eval runs were not started in this work slice.

## Next actions

1. Prepare or collect task dataset episodes.
2. Run `script/select_training_data.py` to generate multi-subset selection files and `selection_summary.csv`.
3. Run `script/run_act_selection_experiments.py` for subset training/evaluation, then verify `_result.txt` and save artifacts.
4. Continue correlation analysis and final model selection according to experiment outcomes.

## Follow-up in same session: official dataset pull-down

- Confirmed official RoboTwin dataset source is on HuggingFace:
  - `TianxingChen/RoboTwin2.0` (contains `dataset/<task>/<embodiment>_{clean|randomized}_{N}.zip`)
- Queried remote file list and archive sizes with `huggingface_hub`.
- Downloaded and unpacked one practical subset for immediate experiments:
  - Remote archive: `dataset/click_bell/aloha-agilex_clean_50.zip`
  - Local target: `data/click_bell/demo_clean/`
  - Verified: 50 `episode*.hdf5` and 50 `episode*.mp4`
- Verified episode schema compatibility:
  - Keys include `/joint_action`, `/observation`, `/pointcloud`
  - `/joint_action/vector` shape observed: `(81, 14)` on `episode0.hdf5`

## Runtime bug fix discovered during real run

- `script/select_training_data.py` failed at runtime due to import path mismatch:
  - old import in selector core expected `dinov2.extract_feature...`
  - actual repo path is submodule directory `dinov2/extract_feature/...`
- Fixed in `script/data_selection/feature_selector.py` by injecting submodule root into `sys.path` and importing from `extract_feature.extract_image_features`.

## Real execution proof after data pull and fix

- Command:
  - `uv run python script/select_training_data.py click_bell demo_clean 10 --sample-frames 4 --n-subsets 2 --strategy greedy_maxdist --metric cosine_distance --output-dir experiments/data_selection/selections_click_bell --device cpu`
- Outputs generated:
  - `experiments/data_selection/selections_click_bell/selection_summary.csv`
  - `experiments/data_selection/selections_click_bell/selection_greedy_maxdist_seed42.json`
  - `experiments/data_selection/selections_click_bell/selection_greedy_maxdist_seed43.json`
  - `experiments/data_selection/selections_click_bell/best_selection.txt`

## Follow-up continuation: 50 vs 500 and training/eval progress

- Verified official dataset scale directly from HuggingFace file index:
  - `zip_total=460`, `clean50=230`, `randomized500=230`.
  - Therefore, official RoboTwin provides both 50 and 500 variants per task/embodiment set.
- Downloaded and mapped `click_bell` randomized dataset:
  - source: `dataset/click_bell/aloha-agilex_randomized_500.zip`
  - target: `data/click_bell/demo_randomized/`
  - verified counts: `500` hdf5 + `500` mp4.

### Selection baseline executed on both data scales

- 50-episode set (`demo_clean`):
  - output: `experiments/data_selection/click_bell_demo_clean/selection_summary.csv`
  - best cosine distance observed: `0.316876...` (seed 44, n_select=20).
- 500-episode set (`demo_randomized`):
  - output: `experiments/data_selection/click_bell_demo_randomized/selection_summary.csv`
  - best cosine distance observed: `0.658939...` (seed 45, n_select=50).

### Training/eval continuation

- Processed selected randomized subset into ACT format:
  - `uv run python policy/ACT/process_data.py click_bell demo_randomized 50 --episode-ids-file .../selection_greedy_maxdist_seed45.json --subset-tag sel45`
- Trained ACT model on selected randomized subset (5 epochs):
  - ckpt dir: `policy/ACT/act_ckpt/act-click_bell/demo_randomized-50-sel45/`
  - artifacts include `policy_last.ckpt`, `policy_best.ckpt`.
- Started deployment eval for full loop:
  - clean subset run (`demo_clean-sel0`) and randomized subset run (`demo_randomized-sel45`) both produced many episode videos,
  - but `_result.txt` is still missing because long eval did not finish within current execution window.

### Final continuation completion

- Completed full 100-episode eval for randomized selected model:
  - output dir: `eval_result/click_bell/ACT/demo_randomized/demo_randomized-sel45/2026-03-08 15:33:53/`
  - verified files include `episode99.mp4` and `_result.txt`.
- `_result.txt` verified content:
  - `Instruction Type: unseen`
  - final success rate value: `0.0`

### Environment fixes required during continuation

- Added runtime dependencies needed by ACT/deploy path:
  - `einops`
  - `dm-control`
  - pinned `setuptools<81` to restore `pkg_resources` required by sapien import path.
