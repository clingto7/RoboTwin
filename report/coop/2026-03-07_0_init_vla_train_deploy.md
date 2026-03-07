# Coop Handoff: init + docs + ACT pipeline run

## Changed files

- `AGENTS.md`
- `task_config/laptop_demo.yml`
- `docs/vla_policy_beginner_guide.md`
- `docs/run_robo_twin_laptop.md`
- `report/2026-03-07_0_init_vla_train_deploy.md`
- `report/coop/2026-03-07_0_init_vla_train_deploy.md`

## Runtime artifacts created

- Data: `data/click_bell/laptop_demo/`
- ACT ckpt: `policy/ACT/act_ckpt/act-click_bell/laptop_demo-2/`
- Deploy videos: `eval_result/click_bell/ACT/laptop_demo/laptop_demo/2026-03-07 01:57:39/episode*.mp4`

## Execution summary

1. UV env + dependency install completed.
2. Assets downloaded and embodiment paths updated.
3. cuRobo built/installed from `envs/curobo`.
4. Collected `click_bell` demo data using `laptop_demo` config (2 episodes).
5. ACT preprocess + quick training (10 epochs) completed.
6. ACT eval/deploy command executed and deployment mp4 files generated.
7. Eval default is long-run (`test_num=100` in `script/eval_policy.py`); this run validated deploy video generation but may be partial evaluation.

## Known caveats

- `collect_data.sh` references missing `script/.update_path.sh`; direct python path was used: `python script/collect_data.py ...`.
- Quick train setting is only for smoke test; quality is expected to be low.
