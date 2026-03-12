import sys

sys.path.append("./policy/ACT/")

import os
import h5py
import numpy as np
import cv2
import argparse
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
        image_dict = dict()
        for cam_name in root["/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def data_transform(path, episode_num, save_path, episode_ids=None):
    begin = 0
    floders = os.listdir(path)
    assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if episode_ids is None:
        selected_ids = list(range(episode_num))
    else:
        selected_ids = episode_ids[:episode_num]

    for i, episode_id in enumerate(selected_ids):
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = (
            load_hdf5(os.path.join(path, f"episode{episode_id}.hdf5"))
        )
        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        state = None
        for j in range(0, left_gripper_all.shape[0]):
            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            if j != left_gripper_all.shape[0] - 1:
                state = np.concatenate(
                    (left_arm, [left_gripper], right_arm, [right_gripper]), axis=0
                )  # joint

                state = state.astype(np.float32)
                qpos.append(state)

                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(
                    np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR
                )
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(
                    np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR
                )
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(
                    np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR
                )
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            if j != 0 and state is not None:
                action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        hdf5path = os.path.join(save_path, f"episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            # cam_high_enc, len_high = images_encoding(cam_high)
            # cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            # cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=np.stack(cam_high), dtype=np.uint8)
            image.create_dataset(
                "cam_right_wrist", data=np.stack(cam_right_wrist), dtype=np.uint8
            )
            image.create_dataset(
                "cam_left_wrist", data=np.stack(cam_left_wrist), dtype=np.uint8
            )

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., adjust_bottle)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--episode-ids-file", type=str, default=None)
    parser.add_argument("--subset-tag", type=str, default="")

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num

    subset_tag = args.subset_tag.strip()
    selected_episode_ids = load_episode_ids(args.episode_ids_file)

    begin = 0
    begin = data_transform(
        os.path.join("../../data/", task_name, task_config, "data"),
        expert_data_num,
        f"processed_data/sim-{task_name}/{task_config}-{expert_data_num}{('-' + subset_tag) if subset_tag else ''}",
        episode_ids=selected_episode_ids,
    )

    SIM_TASK_CONFIGS_PATH = "./SIM_TASK_CONFIGS.json"

    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
    except Exception:
        SIM_TASK_CONFIGS = {}

    dataset_key = f"sim-{task_name}-{task_config}-{expert_data_num}{('-' + subset_tag) if subset_tag else ''}"
    SIM_TASK_CONFIGS[dataset_key] = {
        "dataset_dir": f"./processed_data/sim-{task_name}/{task_config}-{expert_data_num}{('-' + subset_tag) if subset_tag else ''}",
        "num_episodes": expert_data_num,
        "episode_len": 1000,
        "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
    }

    with open(SIM_TASK_CONFIGS_PATH, "w") as f:
        json.dump(SIM_TASK_CONFIGS, f, indent=4)
