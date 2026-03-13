import argparse
import csv
import json
import os
import sys
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = REPO_ROOT / "script"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from data_selection.feature_selector import (  # noqa: E402
    METRIC_REGISTRY,
    SELECTOR_REGISTRY,
    build_episode_embeddings,
    build_frame_embeddings,
    compute_subset_metrics,
    save_selection_result,
    select_episode_indices,
)


def load_episode_ids(episode_ids_file: str | None) -> list[int] | None:
    if episode_ids_file is None:
        return None
    with open(episode_ids_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "selected_episode_ids" in payload:
        return [int(x) for x in payload["selected_episode_ids"]]
    if isinstance(payload, list):
        return [int(x) for x in payload]
    raise ValueError("episode ids file must be a list or contain selected_episode_ids")


def load_hdf5(dataset_path: Path):
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist at: {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = {}
        for cam_name in root["/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs):
    encode_data = []
    max_len = 0
    for image in imgs:
        _, encoded_image = cv2.imencode(".jpg", image)
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    return encode_data, max_len


def _episode_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("episode*.hdf5"), key=lambda p: int(p.stem.replace("episode", "")))
    if not files:
        raise FileNotFoundError(f"No episode files found in {data_dir}")
    return files


def maybe_pull_randomized_500(
    task_name: str,
    load_dir: Path,
    should_pull: bool,
    repo_id: str,
    embodiment: str,
) -> None:
    has_local = (load_dir / "data").exists() and any((load_dir / "data").glob("episode*.hdf5"))
    if has_local or not should_pull:
        return

    from huggingface_hub import hf_hub_download

    archive_name = f"dataset/{task_name}/{embodiment}_randomized_500.zip"
    zip_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=archive_name)
    load_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(load_dir)


def select_with_data_selection(
    task_name: str,
    task_config: str,
    episode_files: list[Path],
    n_select: int,
    args,
) -> tuple[list[int], str]:
    output_dir = Path(args.selector_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.selector_device)

    emb_res = build_episode_embeddings(
        episode_files=episode_files,
        model_name=args.selector_model,
        device=device,
        sample_frames=args.selector_sample_frames,
        batch_size=args.selector_batch_size,
        camera_name=args.selector_frame_camera,
    )
    file_map = {int(p.stem.replace("episode", "")): p for p in episode_files}

    if args.selector_metric_level == "frame":
        full_features = build_frame_embeddings(
            episode_files=episode_files,
            model_name=args.selector_model,
            device=device,
            camera_name=args.selector_frame_camera,
            batch_size=args.selector_batch_size,
        )
        full_metrics = compute_subset_metrics(full_features)
    else:
        full_metrics = compute_subset_metrics(emb_res.embeddings)

    rows = []
    subsets = []
    for i in range(args.selector_n_subsets):
        run_seed = args.selector_seed + i
        selected_ids, selected_metrics = select_episode_indices(
            episode_ids=emb_res.episode_ids,
            embeddings=emb_res.embeddings,
            strategy=args.selector_strategy,
            n_select=n_select,
            seed=run_seed,
        )

        if args.selector_metric_level == "frame":
            selected_files = [file_map[ep_id] for ep_id in selected_ids]
            selected_features = build_frame_embeddings(
                episode_files=selected_files,
                model_name=args.selector_model,
                device=device,
                camera_name=args.selector_frame_camera,
                batch_size=args.selector_batch_size,
            )
            selected_metrics = compute_subset_metrics(selected_features)

        score = selected_metrics[args.selector_metric]
        selection_file = output_dir / f"selection_{args.selector_strategy}_seed{run_seed}.json"
        save_selection_result(
            out_path=selection_file,
            task_name=task_name,
            task_config=task_config,
            strategy=args.selector_strategy,
            metric=args.selector_metric,
            selected_ids=selected_ids,
            selected_metrics=selected_metrics,
            full_metrics=full_metrics,
        )
        subsets.append((run_seed, selected_ids))
        rows.append(
            {
                "selection_file": str(selection_file),
                "seed": run_seed,
                "strategy": args.selector_strategy,
                "metric_level": args.selector_metric_level,
                "primary_metric": args.selector_metric,
                "primary_metric_value": score,
                "cosine_distance": selected_metrics["cosine_distance"],
                "l2_distance": selected_metrics["l2_distance"],
                "variance_score": selected_metrics["variance_score"],
                "feature_mean": selected_metrics["feature_mean"],
                "feature_std": selected_metrics["feature_std"],
                "feature_var": selected_metrics["feature_var"],
                "selected_episode_ids": ",".join(map(str, selected_ids)),
            }
        )

    summary_csv = output_dir / "selection_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    subset_idx = args.selector_subset_index
    if subset_idx < 0 or subset_idx >= len(subsets):
        raise ValueError(f"selector_subset_index {subset_idx} out of range for {len(subsets)} subsets")
    chosen_seed, chosen_ids = subsets[subset_idx]
    subset_tag = args.subset_tag.strip() if args.subset_tag else f"sel{chosen_seed}"
    return chosen_ids, subset_tag


def data_transform(
    load_dir: Path,
    episode_num: int,
    save_path: Path,
    episode_ids: list[int] | None = None,
):
    if episode_ids is None:
        selected_ids = list(range(episode_num))
    else:
        if len(episode_ids) < episode_num:
            raise ValueError("episode ids provided are fewer than expert_data_num")
        selected_ids = episode_ids[:episode_num]

    save_path.mkdir(parents=True, exist_ok=True)
    for out_idx, episode_id in enumerate(selected_ids):
        desc_type = "seen"
        instruction_data_path = load_dir / "instructions" / f"episode{episode_id}.json"
        if instruction_data_path.exists():
            with instruction_data_path.open("r", encoding="utf-8") as f_instr:
                instruction_dict = json.load(f_instr)
            save_instructions_json = {"instructions": instruction_dict.get(desc_type, [])}
        else:
            save_instructions_json = {"instructions": []}

        episode_dir = save_path / f"episode_{out_idx}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        with (episode_dir / "instructions.json").open("w", encoding="utf-8") as f:
            json.dump(save_instructions_json, f, indent=2)

        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            image_dict,
        ) = load_hdf5(load_dir / "data" / f"episode{episode_id}.hdf5")

        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        for j in range(left_gripper_all.shape[0]):
            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )
            state = np.array(
                left_arm.tolist() + [left_gripper] + right_arm.tolist() + [right_gripper],
                dtype=np.float32,
            )

            if j != left_gripper_all.shape[0] - 1:
                qpos.append(state)

                camera_high = cv2.imdecode(
                    np.frombuffer(image_dict["head_camera"][j], np.uint8),
                    cv2.IMREAD_COLOR,
                )
                cam_high.append(cv2.resize(camera_high, (640, 480)))

                camera_right_wrist = cv2.imdecode(
                    np.frombuffer(image_dict["right_camera"][j], np.uint8),
                    cv2.IMREAD_COLOR,
                )
                cam_right_wrist.append(cv2.resize(camera_right_wrist, (640, 480)))

                camera_left_wrist = cv2.imdecode(
                    np.frombuffer(image_dict["left_camera"][j], np.uint8),
                    cv2.IMREAD_COLOR,
                )
                cam_left_wrist.append(cv2.resize(camera_left_wrist, (640, 480)))

            if j != 0:
                actions.append(state)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        hdf5path = episode_dir / f"episode_{out_idx}.hdf5"
        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            cam_high_enc, len_high = images_encoding(cam_high)
            cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            image.create_dataset("cam_right_wrist", data=cam_right_wrist_enc, dtype=f"S{len_right}")
            image.create_dataset("cam_left_wrist", data=cam_left_wrist_enc, dtype=f"S{len_left}")
        print(f"process episode {episode_id} -> episode_{out_idx} success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process episodes for pi0 training")
    parser.add_argument("task_name", type=str)
    parser.add_argument("setting", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--episode-ids-file", type=str, default=None)
    parser.add_argument("--subset-tag", type=str, default="")
    parser.add_argument(
        "--selector-strategy",
        type=str,
        default=None,
        choices=[None] + list(SELECTOR_REGISTRY.keys()),
    )
    parser.add_argument(
        "--selector-metric",
        type=str,
        default="cosine_distance",
        choices=list(METRIC_REGISTRY.keys()),
    )
    parser.add_argument("--selector-model", type=str, default="dinov2_vits14")
    parser.add_argument("--selector-sample-frames", type=int, default=8)
    parser.add_argument("--selector-batch-size", type=int, default=64)
    parser.add_argument("--selector-frame-camera", type=str, default="head_camera")
    parser.add_argument(
        "--selector-metric-level",
        type=str,
        default="frame",
        choices=["episode", "frame"],
    )
    parser.add_argument("--selector-n-subsets", type=int, default=5)
    parser.add_argument("--selector-seed", type=int, default=42)
    parser.add_argument("--selector-subset-index", type=int, default=0)
    parser.add_argument("--selector-output-dir", type=str, default="selection_outputs")
    parser.add_argument(
        "--selector-device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--pull-randomized-500", action="store_true")
    parser.add_argument("--hf-repo-id", type=str, default="TianxingChen/RoboTwin2.0")
    parser.add_argument("--hf-embodiment", type=str, default="aloha-agilex")
    args = parser.parse_args()

    task_name = args.task_name
    setting = args.setting
    expert_data_num = args.expert_data_num
    load_dir = REPO_ROOT / "data" / task_name / setting

    maybe_pull_randomized_500(
        task_name=task_name,
        load_dir=load_dir,
        should_pull=args.pull_randomized_500,
        repo_id=args.hf_repo_id,
        embodiment=args.hf_embodiment,
    )

    selected_episode_ids = load_episode_ids(args.episode_ids_file)
    subset_tag = args.subset_tag.strip()
    if selected_episode_ids is None and args.selector_strategy is not None:
        selected_episode_ids, subset_tag = select_with_data_selection(
            task_name=task_name,
            task_config=setting,
            episode_files=_episode_files(load_dir / "data"),
            n_select=expert_data_num,
            args=args,
        )

    target_name = f"{task_name}-{setting}-{expert_data_num}"
    if subset_tag:
        target_name = f"{target_name}-{subset_tag}"
    target_dir = Path("processed_data") / target_name
    data_transform(
        load_dir=load_dir,
        episode_num=expert_data_num,
        save_path=target_dir,
        episode_ids=selected_episode_ids,
    )
