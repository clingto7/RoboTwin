import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


MetricFn = Callable[[torch.Tensor], float]
SelectorFn = Callable[[torch.Tensor, int, int], list[int]]


DEFAULT_CAMERA_CANDIDATES = (
    "head_camera",
    "cam_high",
    "middle_camera",
)


def _cosine_distance(features: torch.Tensor) -> float:
    if features.shape[0] < 2:
        return 0.0
    normed = F.normalize(features, p=2, dim=-1)
    sim = normed @ normed.T
    tri = torch.triu(1.0 - sim, diagonal=1)
    vals = tri[tri > 0]
    return float(vals.mean().item()) if vals.numel() > 0 else 0.0


def _l2_distance(features: torch.Tensor) -> float:
    if features.shape[0] < 2:
        return 0.0
    dist = torch.cdist(features, features, p=2)
    tri = torch.triu(dist, diagonal=1)
    vals = tri[tri > 0]
    return float(vals.mean().item()) if vals.numel() > 0 else 0.0


def _variance_score(features: torch.Tensor) -> float:
    if features.shape[0] < 2:
        return 0.0
    return float(features.var(dim=0).mean().item())


def _feature_mean(features: torch.Tensor) -> float:
    return float(features.mean().item())


def _feature_std(features: torch.Tensor) -> float:
    if features.numel() < 2:
        return 0.0
    return float(features.std(unbiased=False).item())


def _feature_var(features: torch.Tensor) -> float:
    if features.numel() < 2:
        return 0.0
    return float(features.var(unbiased=False).item())


METRIC_REGISTRY: dict[str, MetricFn] = {
    "cosine_distance": _cosine_distance,
    "l2_distance": _l2_distance,
    "variance_score": _variance_score,
    "feature_mean": _feature_mean,
    "feature_std": _feature_std,
    "feature_var": _feature_var,
}


def register_metric(name: str, metric_fn: MetricFn) -> None:
    if name in METRIC_REGISTRY:
        raise ValueError(f"metric already registered: {name}")
    METRIC_REGISTRY[name] = metric_fn


def _select_random(embeddings: torch.Tensor, n_select: int, seed: int) -> list[int]:
    n = embeddings.shape[0]
    if n_select >= n:
        return list(range(n))
    g = torch.Generator()
    g.manual_seed(seed)
    return sorted(torch.randperm(n, generator=g)[:n_select].tolist())


def _select_random_subsets(
    embeddings: torch.Tensor, n_select: int, seed: int
) -> list[int]:
    return _select_random(embeddings, n_select, seed)


def _select_greedy_maxdist(
    embeddings: torch.Tensor, n_select: int, seed: int
) -> list[int]:
    n = embeddings.shape[0]
    if n_select >= n:
        return list(range(n))

    normed = F.normalize(embeddings, p=2, dim=-1)
    dist = 1.0 - (normed @ normed.T)
    g = torch.Generator()
    g.manual_seed(seed)
    selected = [int(torch.randint(n, (1,), generator=g).item())]
    remaining = set(range(n)) - set(selected)

    for _ in range(n_select - 1):
        best_idx = -1
        best_min_dist = -1.0
        for cand in remaining:
            min_dist = min(float(dist[cand, s].item()) for s in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = cand
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def _select_greedy_maxvar(
    embeddings: torch.Tensor, n_select: int, seed: int
) -> list[int]:
    n = embeddings.shape[0]
    if n_select >= n:
        return list(range(n))

    normed = F.normalize(embeddings, p=2, dim=-1)
    sim = normed @ normed.T
    sim.fill_diagonal_(2.0)
    dist = 1.0 - sim
    flat_idx = int(dist.argmax().item())
    i, j = flat_idx // n, flat_idx % n
    selected = [i, j]
    remaining = set(range(n)) - {i, j}

    while len(selected) < n_select and remaining:
        best_idx = -1
        best_var = -1.0
        for cand in remaining:
            trial = embeddings[selected + [cand]]
            score = float(trial.var(dim=0).mean().item())
            if score > best_var:
                best_var = score
                best_idx = cand
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def _select_kmeans(embeddings: torch.Tensor, n_select: int, seed: int) -> list[int]:
    n = embeddings.shape[0]
    if n_select >= n:
        return list(range(n))

    normed = F.normalize(embeddings, p=2, dim=-1)
    g = torch.Generator()
    g.manual_seed(seed)
    centroids = normed[torch.randperm(n, generator=g)[:n_select]].clone()

    for _ in range(60):
        sim = normed @ centroids.T
        assign = sim.argmax(dim=1)
        new_centroids = torch.zeros_like(centroids)
        for k in range(n_select):
            mask = assign == k
            if mask.any():
                new_centroids[k] = normed[mask].mean(dim=0)
            else:
                new_centroids[k] = normed[
                    int(torch.randint(n, (1,), generator=g).item())
                ]
        centroids = F.normalize(new_centroids, p=2, dim=-1)

    sim = normed @ centroids.T
    selected = []
    used = set()
    for k in range(n_select):
        scores = sim[:, k].clone()
        for idx in used:
            scores[idx] = -2.0
        best = int(scores.argmax().item())
        selected.append(best)
        used.add(best)
    return selected


SELECTOR_REGISTRY: dict[str, SelectorFn] = {
    "random": _select_random,
    "random_subsets": _select_random_subsets,
    "greedy_maxdist": _select_greedy_maxdist,
    "greedy_maxvar": _select_greedy_maxvar,
    "kmeans": _select_kmeans,
}


def register_selector(name: str, selector_fn: SelectorFn) -> None:
    if name in SELECTOR_REGISTRY:
        raise ValueError(f"selector already registered: {name}")
    SELECTOR_REGISTRY[name] = selector_fn


@dataclass
class EpisodeEmbeddingResult:
    episode_ids: list[int]
    embeddings: torch.Tensor


def _default_camera_key(cameras: list[str]) -> str:
    for candidate in DEFAULT_CAMERA_CANDIDATES:
        if candidate in cameras:
            return candidate
    if len(cameras) >= 3:
        return sorted(cameras)[len(cameras) // 2]
    return cameras[0]


def _sample_indices(total: int, n: int) -> list[int]:
    if n <= 1:
        return [0] if total > 0 else []
    if total <= n:
        return list(range(total))
    step = (total - 1) / float(n - 1)
    return [int(round(i * step)) for i in range(n)]


def _decode_rgb(frame_bytes: bytes) -> Image.Image:
    bgr = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _extract_cls_feature_batches(
    frames: np.ndarray,
    model,
    transform,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    features: list[torch.Tensor] = []
    total = len(frames)
    for start in range(0, total, batch_size):
        batch_frames = frames[start : start + batch_size]
        tensors = torch.stack([transform(_decode_rgb(frame)) for frame in batch_frames])
        from extract_feature.extract_image_features import extract_cls_features

        cls = extract_cls_features(model, tensors, device)
        features.append(cls.cpu())
    return torch.cat(features, dim=0)


def build_episode_embeddings(
    episode_files: list[Path],
    model_name: str,
    device: torch.device,
    sample_frames: int,
    batch_size: int = 64,
    camera_name: str | None = None,
) -> EpisodeEmbeddingResult:
    repo_root = Path(__file__).resolve().parents[2]
    dinov2_root = repo_root / "dinov2"
    if str(dinov2_root) not in sys.path:
        sys.path.insert(0, str(dinov2_root))

    from extract_feature.extract_image_features import load_model, make_transform

    transform = make_transform()
    model = load_model(model_name, device)
    episode_ids: list[int] = []
    embeddings: list[torch.Tensor] = []

    for file_path in episode_files:
        stem = file_path.stem
        ep_id = int(stem.replace("episode", ""))
        with h5py.File(file_path, "r") as root:
            cameras = list(root["/observation"].keys())
            key = (
                camera_name if camera_name in cameras else _default_camera_key(cameras)
            )
            rgb_arr = root[f"/observation/{key}/rgb"][()]

        picked = _sample_indices(len(rgb_arr), sample_frames)
        cls = _extract_cls_feature_batches(
            rgb_arr[picked], model, transform, device, batch_size
        )
        episode_ids.append(ep_id)
        embeddings.append(cls.mean(dim=0).cpu())

    return EpisodeEmbeddingResult(
        episode_ids=episode_ids, embeddings=torch.stack(embeddings)
    )


def build_frame_embeddings(
    episode_files: list[Path],
    model_name: str,
    device: torch.device,
    camera_name: str | None = None,
    batch_size: int = 64,
) -> torch.Tensor:
    repo_root = Path(__file__).resolve().parents[2]
    dinov2_root = repo_root / "dinov2"
    if str(dinov2_root) not in sys.path:
        sys.path.insert(0, str(dinov2_root))

    from extract_feature.extract_image_features import load_model, make_transform

    transform = make_transform()
    model = load_model(model_name, device)
    all_features: list[torch.Tensor] = []

    for file_path in episode_files:
        with h5py.File(file_path, "r") as root:
            cameras = list(root["/observation"].keys())
            key = (
                camera_name if camera_name in cameras else _default_camera_key(cameras)
            )
            rgb_arr = root[f"/observation/{key}/rgb"][()]
        frame_features = _extract_cls_feature_batches(
            rgb_arr, model, transform, device, batch_size
        )
        all_features.append(frame_features)

    if not all_features:
        raise ValueError("No frame embeddings extracted; episode_files is empty")
    return torch.cat(all_features, dim=0)


def compute_subset_metrics(embeddings: torch.Tensor) -> dict[str, float]:
    return {name: fn(embeddings) for name, fn in METRIC_REGISTRY.items()}


def select_episode_indices(
    episode_ids: list[int],
    embeddings: torch.Tensor,
    strategy: str,
    n_select: int,
    seed: int,
) -> tuple[list[int], dict[str, float]]:
    if strategy not in SELECTOR_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy}")
    selected_local = SELECTOR_REGISTRY[strategy](embeddings, n_select, seed)
    selected_ids = [episode_ids[i] for i in selected_local]
    metrics = compute_subset_metrics(embeddings[selected_local])
    return selected_ids, metrics


def save_selection_result(
    out_path: Path,
    task_name: str,
    task_config: str,
    strategy: str,
    metric: str,
    selected_ids: list[int],
    selected_metrics: dict[str, float],
    full_metrics: dict[str, float],
) -> None:
    payload = {
        "task_name": task_name,
        "task_config": task_config,
        "strategy": strategy,
        "primary_metric": metric,
        "selected_episode_ids": selected_ids,
        "selected_metrics": selected_metrics,
        "full_metrics": full_metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
