# METRIC_REGISTRY

注册了使用到的metric计算函数，包括计算`cosine_distance`,`L2_distance`,`variance`

输入一个
```python
MetricFn = Callable[[torch.Tensor], float]
```

用于计算subset数据集metric的工具集合，用于评估subset数据集的质量

# SELECTOR_REGISTRY

注册了使用到的选择数据的策略，包括`random`,`greedy_maxdist`,`greedy_maxvar`,`kmeans`

```python
SelectorFn = Callable[[torch.Tensor, int, int], list[int]]
```
# utils

## EpisodeEmbeddingResult
按顺序存储episode_id和该episode对应的embedding
```python
@dataclass
class EpisodeEmbeddingResult:
    episode_ids: list[int]
    embeddings: torch.Tensor
```

## _sample_indices

hdf5数据集cam_high采样帧工具

## build_episode_embeddings

由dinov2 submodule引入图像feature提取工具，对指定的episode_path读取数据，采样图片，抽feature求平均来代表整个episode情况

## select_episode_indices
按照指定的策略根据计算的episodes 的 embeddings选择出subset所使用的策略，再计算出各个metric
