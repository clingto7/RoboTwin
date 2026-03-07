# RoboTwin VLA/Policy Beginner Guide

## 1) What this project is

RoboTwin is a dual-arm manipulation simulation platform. In practice, you usually run:

1. collect demonstrations in simulator,
2. convert data to policy-specific format,
3. train policy,
4. deploy/evaluate policy and record videos.

For a laptop, ACT/DP are the best first choice. Large VLA pipelines (OpenVLA/TinyVLA/DexVLA) need much more compute and memory.

## 2) Core concepts you need first

- `Task`: one manipulation goal, e.g. `click_bell`.
- `Task config`: scene/randomization/camera/data settings in `task_config/*.yml`.
- `Demo data`: trajectories collected by expert planner from the simulator.
- `Policy`: model that maps observation -> robot action.

## 3) How the main policy families differ

### ACT (Action Chunking Transformer)

- Learns to predict a chunk of future actions (not just one step).
- Strong baseline for imitation with relatively small data.
- In RoboTwin local flow: `process_data.sh -> train.sh -> eval.sh`.

Intuition: one policy forward pass predicts a short action sequence, so control is smoother and less jittery.

### DP (Diffusion Policy)

- Treats action generation like denoising in diffusion models.
- Good at multi-modal behavior (multiple valid ways to solve a task).
- More training cost than ACT, but often robust.

Intuition: instead of one deterministic action, the model iteratively denoises toward a plausible action sequence.

### VLA (Vision-Language-Action) models

- Use large VLM backbones + action head (or adapter) for robot control.
- Can use language prompt better, but are expensive (data format + compute + memory).
- In this repo: OpenVLA-oft/TinyVLA/DexVLA pipelines are available, but not laptop-first.

Intuition: VLA adds stronger visual/language priors, but pipeline complexity rises a lot.

## 4) Why start with ACT/DP on laptop

- Fastest way to verify full loop (data -> train -> deploy -> video).
- Fewer moving parts than VLA data conversion/fine-tuning.
- Lower GPU memory pressure on RTX 4060.

Practical rule:

- first run ACT with tiny dataset (2~5 demos),
- then increase demos/epochs,
- then try DP,
- move to VLA only after stable reproducible baseline.

## 5) Mapping concepts to this repo

- Data collection: `script/collect_data.py`
- ACT preprocess/train: `policy/ACT/process_data.py`, `policy/ACT/imitate_episodes.py`
- Deployment/eval loop + video writing: `script/eval_policy.py`
- Deployment model wrapper for ACT: `policy/ACT/deploy_policy.py`

## 6) Compute tips for your machine (RTX 4060 laptop)

- Keep `render_freq: 0`.
- Keep `episode_num` small for first pass.
- For ACT training, lower `batch_size` first when OOM.
- Verify one policy and one task fully before adding randomization/other policies.
