# 2026-03-07 Work Report: init + VLA docs + ACT train/deploy demo

## Objective

- Finish project init on laptop.
- Supplement `AGENTS.md` with actionable init/run guidance.
- Create beginner docs for VLA/policy understanding.
- Run a real train/deploy path and keep deployment video artifacts.

## Work done

1. Environment and dependencies

- Created UV env: `.venv` with Python 3.10.
- Installed base deps from `script/requirements.txt`.
- Downloaded and unpacked assets via `script/_download_assets.sh`.
- Ran `python script/update_embodiment_config_path.py` successfully after assets were present.
- Installed cuRobo from source (`envs/curobo`) to satisfy planner dependency.

2. Repo doc/config updates

- Updated `AGENTS.md` with:
  - `INIT` command list,
  - policy run order (ACT/DP),
  - laptop resource control rules,
  - note that `collect_data.sh` references missing `script/.update_path.sh`.
- Added `task_config/laptop_demo.yml` for low-cost validation (`episode_num=2`, `render_freq=0`).
- Added beginner doc: `docs/vla_policy_beginner_guide.md`.
- Added runbook doc: `docs/run_robo_twin_laptop.md`.

3. Real run execution

- Data collection:
  - `python script/collect_data.py click_bell laptop_demo`
  - Output data under `data/click_bell/laptop_demo/data/`.
  - Collection videos generated under `data/click_bell/laptop_demo/video/`.
- ACT preprocess:
  - 在 `policy/ACT` 目录内执行：`bash process_data.sh click_bell laptop_demo 2`
- ACT training (quick laptop mode):
  - 在 `policy/ACT` 目录内执行：`python imitate_episodes.py ... --num_epochs 10 --batch_size 2`
  - Checkpoints created in `policy/ACT/act_ckpt/act-click_bell/laptop_demo-2/`.
- ACT deploy/eval run:
  - `bash policy/ACT/eval.sh click_bell laptop_demo laptop_demo 2 0 0`
  - Deployment videos written to `eval_result/click_bell/ACT/laptop_demo/laptop_demo/2026-03-07 01:57:39/`.

## Verification artifacts

- Collection video example:
  - `data/click_bell/laptop_demo/video/episode0.mp4`
- Deployment video example:
  - `eval_result/click_bell/ACT/laptop_demo/laptop_demo/2026-03-07 01:57:39/episode0.mp4`
- Checkpoint files:
  - `policy/ACT/act_ckpt/act-click_bell/laptop_demo-2/policy_best.ckpt`
  - `policy/ACT/act_ckpt/act-click_bell/laptop_demo-2/policy_last.ckpt`

## Notes

- Short training (`2 demos`, `10 epochs`) is for pipeline validation, not final policy quality.
- `script/eval_policy.py` 默认 `test_num=100`，长时评测可能被中断；本次已生成多段部署视频，但不代表完整100条评测已结束。
