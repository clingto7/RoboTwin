# RoboTwin Laptop Runbook (Init -> Train -> Deploy -> Video)

## 0) Environment init

```bash
python3 -V
uv --version
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -r script/requirements.txt
```

## 1) Download assets and configure embodiment paths

```bash
source .venv/bin/activate
bash script/_download_assets.sh
python script/update_embodiment_config_path.py
```

## 2) Optional dependency for planner (required by RoboTwin env runtime)

```bash
git clone https://github.com/NVlabs/curobo.git envs/curobo
source .venv/bin/activate
uv pip install ninja
uv pip install -e envs/curobo --no-build-isolation
```

## 3) Minimal task config for laptop validation

Use `task_config/laptop_demo.yml` (already created in this run):

- `episode_num: 2`
- `render_freq: 0`
- `eval_video_log: true`

## 4) Collect tiny dataset

```bash
source .venv/bin/activate
python script/collect_data.py click_bell laptop_demo
```

Expected output data path:

- `data/click_bell/laptop_demo/data/episode0.hdf5`
- `data/click_bell/laptop_demo/data/episode1.hdf5`

Expected collection videos:

- `data/click_bell/laptop_demo/video/episode0.mp4`
- `data/click_bell/laptop_demo/video/episode1.mp4`

## 5) ACT preprocess + train

```bash
cd policy/ACT
source ../../.venv/bin/activate
bash process_data.sh click_bell laptop_demo 2

# laptop-friendly quick train
python imitate_episodes.py \
  --task_name sim-click_bell-laptop_demo-2 \
  --ckpt_dir ./act_ckpt/act-click_bell/laptop_demo-2 \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 50 \
  --hidden_dim 512 \
  --batch_size 2 \
  --dim_feedforward 3200 \
  --num_epochs 10 \
  --lr 1e-5 \
  --save_freq 5 \
  --state_dim 14 \
  --seed 0
```

Important:

- Run these ACT commands from `policy/ACT` exactly, because relative dataset/checkpoint paths are resolved from that directory.

Expected checkpoints:

- `policy/ACT/act_ckpt/act-click_bell/laptop_demo-2/policy_best.ckpt`
- `policy/ACT/act_ckpt/act-click_bell/laptop_demo-2/policy_last.ckpt`

## 6) Deploy/evaluate and save deployment video

```bash
cd policy/ACT
source ../../.venv/bin/activate
bash eval.sh click_bell laptop_demo laptop_demo 2 0 0
```

Notes:

- `script/eval_policy.py` uses a default `test_num=100`, so full eval is long on laptop.
- If interrupted early, you still get partial `episode*.mp4`, but `_result.txt` is not generated.
- `ffmpeg` must be available on `PATH` for video writing.

Evaluation writes videos to:

- `eval_result/click_bell/ACT/laptop_demo/laptop_demo/<timestamp>/episode*.mp4`

Example path from this run (quote because timestamp includes spaces):

- `"eval_result/click_bell/ACT/laptop_demo/laptop_demo/2026-03-07 01:57:39/episode0.mp4"`

## 7) Common issues

- `collect_data.sh` references `script/.update_path.sh` (missing in this repo).
  - Use `python script/collect_data.py ...` directly.
- If planner import fails with `curobo` missing:
  - install `envs/curobo` as above.
- If OOM:
  - reduce `episode_num`, `batch_size`, and keep `render_freq: 0`.
