# Work log

## Context checked

- `AGENTS.md`
- Recent teammate reports:
  - `report/2026-03-13_0_pi0_process_data_all_subsets_switch.md`
  - `report/coop/2026-03-13_0_pi0_process_data_all_subsets_switch.md`
  - `report/2026-03-12_0_pi0_pi05_data_selection_frame_metrics.md`
  - `report/coop/2026-03-12_0_pi0_pi05_data_selection_frame_metrics.md`

## Implemented

- Added feature-based subset generation path for pi0 data selection:
  - strategy options: `feature_std`, `feature_var`
  - frame-level constrained path for these strategies
  - deterministic gradient-style 5-subset planning API for future criteria extension

- Core changes:
  - `script/data_selection/feature_selector.py`
    - added `EpisodeMetricScoreResult`
    - added `build_episode_metric_scores(...)`
    - added `plan_metric_gradient_subsets(...)`
    - registered selectors `feature_std` / `feature_var`
    - optimized `build_episode_metric_scores(...)` to load DINO model once per run
  - `script/data_selection/__init__.py`
    - exported new APIs
  - `script/select_training_data.py`
    - integrated gradient subset flow for `feature_std` / `feature_var`
  - `policy/pi0/scripts/process_data.py`
    - integrated gradient subset flow for `feature_std` / `feature_var`
    - kept existing random/greedy/kmeans paths unchanged
    - added `strategy_metric_value` column in summary rows

## Validation

- LSP diagnostics (error level): clean
  - `script/data_selection/feature_selector.py`
  - `script/data_selection/__init__.py`
  - `script/select_training_data.py`
  - `policy/pi0/scripts/process_data.py`

- Syntax compile:
  - `uv run python -m py_compile script/data_selection/__init__.py script/data_selection/feature_selector.py script/select_training_data.py policy/pi0/scripts/process_data.py`

- Runtime smoke (new optimized function):
  - `uv run python -c "... build_episode_metric_scores(...) ..."`

## Full runs (non-overwrite outputs)

- feature_std 5 subsets x 100 (all subsets processed):
  - command:
    - `uv run bash process_data_pi0.sh click_bell demo_randomized 100 --selector-strategy feature_std --selector-metric-level frame --selector-metric feature_std --pull-randomized-500 --selector-n-subsets 5 --selector-process-all-subsets true --selector-output-dir selection_outputs_feature_std_20260316_0 --subset-tag fstdgrad0`
  - outputs:
    - `policy/pi0/selection_outputs_feature_std_20260316_0/`
    - `policy/pi0/processed_data/click_bell-demo_randomized-100-fstdgrad0-sel42` .. `sel46`

- feature_var 5 subsets x 100 (all subsets processed):
  - command:
    - `uv run bash process_data_pi0.sh click_bell demo_randomized 100 --selector-strategy feature_var --selector-metric-level frame --selector-metric feature_var --pull-randomized-500 --selector-n-subsets 5 --selector-process-all-subsets true --selector-output-dir selection_outputs_feature_var_20260316_0 --subset-tag fvargrad0`
  - outputs:
    - `policy/pi0/selection_outputs_feature_var_20260316_0/`
    - `policy/pi0/processed_data/click_bell-demo_randomized-100-fvargrad0-sel42` .. `sel46`

## Observations from generated outputs

- `feature_std` strategy produced strictly increasing subset `feature_std`:
  - `[2.2096, 2.2526, 2.2760, 2.2998, 2.3422]`
- `feature_var` strategy produced strictly increasing subset `feature_var`:
  - `[4.8825, 5.0744, 5.1802, 5.2890, 5.4859]`
- Existing random run (`policy/pi0/selection_outputs/selection_summary.csv`) had narrow `feature_std` spread:
  - range `0.0074`
- New std-gradient run spread is much larger:
  - range `0.1326`

This matches the target: subset features are both different from full-feature and different from each other with a clear gradient.
