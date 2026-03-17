## Handoff for teammates

### Goal

- Add non-random subset generation for pi0 so 500->5x100 subsets can follow feature gradients (`feature_std`, `feature_var`) for later metric-vs-training analysis.

### What changed

- `script/data_selection/feature_selector.py`
  - New API: `build_episode_metric_scores(...)`
  - New API: `plan_metric_gradient_subsets(...)`
  - New dataclass: `EpisodeMetricScoreResult`
  - New selectors registered: `feature_std`, `feature_var`
  - Optimization: score-builder now loads DINO model once per run (not once per episode)

- `script/data_selection/__init__.py`
  - Exported `build_episode_metric_scores`, `plan_metric_gradient_subsets`

- `script/select_training_data.py`
  - If strategy is `feature_std` or `feature_var`, uses gradient planner instead of per-seed independent random-like selection

- `policy/pi0/scripts/process_data.py`
  - Same gradient path integrated for pi0 CLI
  - Enforces frame-level for these two strategies
  - Summary rows include `strategy_metric_value`

### Behavior notes

- For `feature_std` / `feature_var`:
  - subsets are formed from sorted metric bands, not independent random picks
  - subsets have increasing metric trend by subset index (seed42 -> seed46)
- Existing strategies (`random`, `greedy_*`, `kmeans`) keep prior flow.

### Commands executed (non-overwrite)

- Std gradient full processing:
  - `uv run bash process_data_pi0.sh click_bell demo_randomized 100 --selector-strategy feature_std --selector-metric-level frame --selector-metric feature_std --pull-randomized-500 --selector-n-subsets 5 --selector-process-all-subsets true --selector-output-dir selection_outputs_feature_std_20260316_0 --subset-tag fstdgrad0`
- Var gradient full processing:
  - `uv run bash process_data_pi0.sh click_bell demo_randomized 100 --selector-strategy feature_var --selector-metric-level frame --selector-metric feature_var --pull-randomized-500 --selector-n-subsets 5 --selector-process-all-subsets true --selector-output-dir selection_outputs_feature_var_20260316_0 --subset-tag fvargrad0`

### Output locations

- Selection outputs:
  - `policy/pi0/selection_outputs_feature_std_20260316_0/`
  - `policy/pi0/selection_outputs_feature_var_20260316_0/`
- Processed datasets:
  - `policy/pi0/processed_data/click_bell-demo_randomized-100-fstdgrad0-sel42` .. `sel46`
  - `policy/pi0/processed_data/click_bell-demo_randomized-100-fvargrad0-sel42` .. `sel46`

### Quick result snapshot

- Std summary (`selection_outputs_feature_std_20260316_0/selection_summary.csv`):
  - feature_std: `2.2096 < 2.2526 < 2.2760 < 2.2998 < 2.3422`
- Var summary (`selection_outputs_feature_var_20260316_0/selection_summary.csv`):
  - feature_var: `4.8825 < 5.0744 < 5.1802 < 5.2890 < 5.4859`
- Random baseline spread was much smaller on std (from `policy/pi0/selection_outputs/selection_summary.csv`).
