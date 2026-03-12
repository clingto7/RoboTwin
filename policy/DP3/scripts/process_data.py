import os
import numpy as np
import zarr
import shutil
import argparse
import h5py
import json


def load_episode_ids(episode_ids_file):
    if episode_ids_file is None:
        return None
    with open(episode_ids_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "selected_episode_ids" in payload:
        return [int(x) for x in payload["selected_episode_ids"]]
    if isinstance(payload, list):
        return [int(x) for x in payload]
    raise ValueError("episode ids file must be a list or contain selected_episode_ids")


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        vector = root["/joint_action/vector"][()]
        pointcloud = root["/pointcloud"][()]

    return left_gripper, left_arm, right_gripper, right_arm, vector, pointcloud


def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    parser.add_argument("--episode-ids-file", type=str, default=None)
    parser.add_argument("--subset-tag", type=str, default="")
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    load_dir = "../../data/" + str(task_name) + "/" + str(task_config)
    subset_tag = args.subset_tag.strip()
    selected_episode_ids = load_episode_ids(args.episode_ids_file)

    total_count = 0

    suffix = f"-{subset_tag}" if subset_tag else ""
    save_dir = f"./data/{task_name}-{task_config}-{num}{suffix}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    point_cloud_arrays = []
    episode_ends_arrays, state_arrays, joint_action_arrays = ([], [], [])

    if selected_episode_ids is None:
        episode_order = list(range(num))
    else:
        episode_order = selected_episode_ids[:num]

    while current_ep < len(episode_order):
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")

        episode_id = episode_order[current_ep]
        load_path = os.path.join(load_dir, f"data/episode{episode_id}.hdf5")
        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            vector_all,
            pointcloud_all,
        ) = load_hdf5(load_path)

        for j in range(0, left_gripper_all.shape[0]):
            pointcloud = pointcloud_all[j]
            joint_state = vector_all[j]

            if j != left_gripper_all.shape[0] - 1:
                point_cloud_arrays.append(pointcloud)
                state_arrays.append(joint_state)
            if j != 0:
                joint_action_arrays.append(joint_state)

        current_ep += 1
        total_count += left_gripper_all.shape[0] - 1
        episode_ends_arrays.append(total_count)

    print()
    try:
        episode_ends_arrays = np.array(episode_ends_arrays)
        state_arrays = np.array(state_arrays)
        point_cloud_arrays = np.array(point_cloud_arrays)
        joint_action_arrays = np.array(joint_action_arrays)

        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        state_chunk_size = (100, state_arrays.shape[1])
        joint_chunk_size = (100, joint_action_arrays.shape[1])
        point_cloud_chunk_size = (100, point_cloud_arrays.shape[1])
        zarr_data.create_dataset(
            "point_cloud",
            data=point_cloud_arrays,
            chunks=point_cloud_chunk_size,
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "state",
            data=state_arrays,
            chunks=state_chunk_size,
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "action",
            data=joint_action_arrays,
            chunks=joint_chunk_size,
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        zarr_meta.create_dataset(
            "episode_ends",
            data=episode_ends_arrays,
            dtype="int64",
            overwrite=True,
            compressor=compressor,
        )
    except ZeroDivisionError:
        print(
            "If you get a `ZeroDivisionError: division by zero`, check that `data/pointcloud` in the task config is set to true."
        )
        raise
    except Exception as e:
        print(f"An unexpected error occurred ({type(e).__name__}): {e}")
        raise


if __name__ == "__main__":
    main()
