## Handoff for teammates

This session focused on stabilizing and validating the new DINOv2-based data-selection baseline integration that had already been partially implemented in the repo.

### What is ready

- Core baseline module exists and is usable:
  - `script/data_selection/feature_selector.py`
  - Registry-based metrics: `cosine_distance`, `l2_distance`, `variance_score`
  - Registry-based selectors: `random`, `greedy_maxdist`, `greedy_maxvar`, `kmeans`
- Selection CLI exists and is usable:
  - `script/select_training_data.py`
  - Supports subset count, seed sweep, strategy, metric, output dir
- Experiment runner exists:
  - `script/run_act_selection_experiments.py`
  - Orchestrates selection -> ACT process_data -> ACT train -> eval -> CSV summary
- ACT/DP/DP3 process_data scripts support optional selection file and subset tag:
  - `policy/ACT/process_data.py`
  - `policy/DP/process_data.py`
  - `policy/DP3/scripts/process_data.py`
  - Shell wrappers updated accordingly

### What was changed in this session

- Code hygiene fixes only (no behavior change):
  - Removed unused imports/locals and minor formatting issues across touched scripts.
- Validation:
  - LSP diagnostics clean on modified Python files.
  - `py_compile` pass on modified Python files.
  - `--help` CLI smoke checks passed under `uv run`.

### Important constraints observed

- No local episode files detected at `data/**/data/episode*.hdf5` during this session.
- End-to-end subset training/evaluation could not be executed without dataset episodes.

### Suggested immediate continuation

1. Collect or place dataset episodes for target task/config.
2. Run `script/select_training_data.py` to produce selection JSONs + summary CSV.
3. Run `script/run_act_selection_experiments.py` for multi-subset ACT experiments.
4. Verify each eval output has `_result.txt`, then aggregate metric-performance correlation.
5. Pick best-correlated metric and train/store final model.

### Follow-up completed (data source + pull)

- Verified official source: HuggingFace dataset `TianxingChen/RoboTwin2.0`.
- Downloaded and unpacked `dataset/click_bell/aloha-agilex_clean_50.zip` to local train layout:
  - `data/click_bell/demo_clean/data/episode*.hdf5` (50)
  - `data/click_bell/demo_clean/video/episode*.mp4` (50)
- Runtime import bug fixed in selector core:
  - `script/data_selection/feature_selector.py` now resolves `dinov2/extract_feature/extract_image_features.py` via submodule root path.
- Confirmed selector real run works:
  - `uv run python script/select_training_data.py click_bell demo_clean 10 --sample-frames 4 --n-subsets 2 --strategy greedy_maxdist --metric cosine_distance --output-dir experiments/data_selection/selections_click_bell --device cpu`
  - artifacts written under `experiments/data_selection/selections_click_bell/`.

### New continuation status (important)

- Official data scale confirmed from HF index:
  - total dataset zips: 460
  - clean50 zips: 230
  - randomized500 zips: 230
- Added `click_bell` randomized500 data locally:
  - `data/click_bell/demo_randomized/data/episode*.hdf5` (500)
  - `data/click_bell/demo_randomized/video/episode*.mp4` (500)
- Selection baseline now has outputs for both scales:
  - `experiments/data_selection/click_bell_demo_clean/selection_summary.csv`
  - `experiments/data_selection/click_bell_demo_randomized/selection_summary.csv`
- ACT randomized selected-subset training completed:
  - ckpt: `policy/ACT/act_ckpt/act-click_bell/demo_randomized-50-sel45/`
- Eval started but not yet fully completed to `_result.txt`:
  - clean run produced partial videos up to many episodes under `eval_result/click_bell/ACT/demo_clean/demo_clean-sel0/...`
  - randomized run produced partial videos under `eval_result/click_bell/ACT/demo_randomized/demo_randomized-sel45/...`
  - both are long-running and were cut by execution window before writing final `_result.txt`.

### Final continuation completion

- Randomized selected model eval completed full 100 episodes:
  - `eval_result/click_bell/ACT/demo_randomized/demo_randomized-sel45/2026-03-08 15:33:53/_result.txt`
  - `episode99.mp4` present in same run directory.
- `_result.txt` contains final success-rate value `0.0`.

### Dependency adjustments made for pipeline continuity

- Added: `einops`, `dm-control`
- Pinned: `setuptools<81` (to provide `pkg_resources` for sapien import chain)
