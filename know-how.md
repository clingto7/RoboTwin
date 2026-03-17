# data_selection

## METRIC_REGISTRY

注册了使用到的metric计算函数，包括计算`cosine_distance`,`L2_distance`,`variance`

输入一个
```python
MetricFn = Callable[[torch.Tensor], float]
```

用于计算subset数据集metric的工具集合，用于评估subset数据集的质量

## SELECTOR_REGISTRY

注册了使用到的选择数据的策略，包括`random`,`greedy_maxdist`,`greedy_maxvar`,`kmeans`

```python
SelectorFn = Callable[[torch.Tensor, int, int], list[int]]
```
## utils

### EpisodeEmbeddingResult
按顺序存储episode_id和该episode对应的embedding
```python
@dataclass
class EpisodeEmbeddingResult:
    episode_ids: list[int]
    embeddings: torch.Tensor
```

### _sample_indices

hdf5数据集cam_high采样帧工具

### build_episode_embeddings

由dinov2 submodule引入图像feature提取工具，对指定的episode_path读取数据，采样图片，抽feature求平均来代表整个episode情况

### select_episode_indices
按照指定的策略根据计算的episodes 的 embeddings选择出subset所使用的策略，再计算出各个metric

# pi0

## setup

```sh
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

## curobo setup

```sh
conda deactivate
source .venv/bin/activate
# At this point, you should be in the (openpi) environment
cd ../../envs
git clone https://github.com/NVlabs/curobo.git
cd curobo
pip install -e . --no-build-isolation
cd ../../policy/pi0/
bash
```

## collect data

```sh
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Clean Data Example: bash collect_data.sh beat_block_hammer demo_clean 0
# Radomized Data Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

collected data under directory **data/${task_name}/${task_config}**

### OR download from the RoboTwin official link

[download link](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset)

wget <download-url> is OK

## in policy/pi0

```sh
mkdir processed_data && mkdir training_data
```

### convert data type

```sh
bash process_data_pi0.sh ${task_name} ${task_config} ${expert_data_num}
# bash process_data_pi0.sh beat_block_hammer demo_clean 50
# or processing randomized data: bash process_data.sh beat_block_hammer demo_randomized 50
```
you will find the `${task_name}-${task_config}-${expert_data_num}` folder under policy/pi0/processed_data

### training data

Copy all the data you wish to use for training from processed_data into `training_data/${model_name}`

### generate lerobot dataset format data for pi0

```sh
# hdf5_path: The path to the generated HDF5 data (e.g., ./training_data/${model_name}/)
# repo_id: The name of the dataset (e.g., my_repo)
bash generate.sh ${hdf5_path} ${repo_id}
#bash generate.sh ./training_data/demo_clean/ demo_clean_repo
```

you'll fin data under `${XDG_CACHE_HOME}/huggingface/lerobot/${repo_id}`

## write training config

## finetune model

# process_data_pi0




ulw 看AGENTS.md 然后看report下的文档,特别是report/coop/目录下的文档,了解一下你的同事做过的一些工作 关于pi0的process_data脚本,目前做了一些修改，
运行命令 ~/ws/RoboTwin/policy/pi0$ uv run bash process_data_pi0.sh click_bell demo_randomized 100 --selector-strategy random
 --selector-metric-level frame --pull-randomized-500 会得到当前selection_outputs里的东西还有processed_data下的数据，目前是会在select出的几个subset里面选一个进行处理，默认参数selector-subset-index为0，在这次的结果里面就是直接用第一次的seed42选择的episode作为数据集进行处理。现在做一下修改，给脚本再加一个arg，如果传入true，就处理所有subset，而不是按照subset-index



uv run bash process_data_pi0.sh click_bell demo_randomized 100 --selector-strategy random --selector-metric-level frame --pull-randomized-500 --selector-process-all-subsets true

uv run bash process_data_pi0.sh click_bell demo_randomized 100 --selector-strategy feature_std --selector-metric-level frame --selector-metric feature_std --pull-randomized-500 --selector-n-subsets 5 --selector-process-all-subsets true --selector-output-dir selection_outputs_feature_std_20260316_0 --subset-tag fstdgrad0

uv run bash process_data_pi0.sh click_bell demo_randomized 100 --selector-strategy feature_var --selector-metric-level frame --selector-metric feature_var --pull-randomized-500 --selector-n-subsets 5 --selector-process-all-subsets true --selector-output-dir selection_outputs_feature_var_20260316_0 --subset-tag fvargrad0
