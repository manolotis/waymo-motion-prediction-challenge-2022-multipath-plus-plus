import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from model.data import get_dataloader
import os
import random
from utils.predict_utils import parse_arguments, get_config
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import bisect

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

args = parse_arguments()
config = get_config(args)

dataloader = get_dataloader(config["test"]["data_config"])


def generate_filename(scene_data, agent_index):
    scenario_id = scene_data["scenario_id"][agent_index]
    agent_id = scene_data["agent_id"][agent_index]
    agent_type = scene_data["target/agent_type"][agent_index]
    return f"scid_{scenario_id}__aid_{agent_id}__atype_{int(agent_type.item())}.npz"


model_name = config["model"]["name"]
if "name_addition" in config["model"]:
    model_name = config["model"]["name"] + "_" + config["model"]["name_addition"]

savefolder = os.path.join(config["test"]["output_config"]["out_path"], model_name)

if not os.path.exists(savefolder):
    os.makedirs(savefolder, exist_ok=True)

    # counts
    # -180: 91
    # -160: 319
    # -140: 305
    # -120: 384
    # -100: 5861
    # -80: 2615
    # -60: 2196
    # -40: 2323
    # -20: 12263
    # 0: 29809
    # 20: 4977
    # 40: 4752
    # 60: 5365
    # 80: 3516
    # 100: 107
    # 120: 106
    # 140: 1
    # 160: 10


# 3500


def map_to_floor_value(value, ranges):
    # Find the insertion point
    index = bisect.bisect_right(ranges, value) - 1
    # Get the floor value
    return ranges[max(0, index)]


ranges = list(range(-180, 170, 20))
counts = {}
for bottom_range in ranges:
    counts[bottom_range] = 0


def map_to_floor_value(value, ranges):
    # Find the insertion point
    index = bisect.bisect_right(ranges, value) - 1
    # Get the floor value
    return ranges[max(0, index)]


files_to_keep = []
files_to_remove = []
counter = 0

for data in tqdm(dataloader):
    yaw_diff = data["target/future/yaw"][:, -1, 0] - data["target/history/yaw"][:, 0, 0]
    xy_start = data["target/history/xy"][:, 0, :2]
    xy_end = data["target/future/xy"][:, -1, :2]

    yaw_diff[yaw_diff < np.deg2rad(-180)] += np.deg2rad(360)
    yaw_diff[yaw_diff > np.deg2rad(180)] -= np.deg2rad(360)

    sids = data["scenario_id"]
    aids = data["agent_id"]
    atypes = data["target/agent_type"]

    for i, y in enumerate(yaw_diff):
        c = map_to_floor_value(np.rad2deg(y), ranges)

        filename = f"scid_{sids[i]}__aid_{int(aids[i])}__atype_{int(atypes[i])}.npz"
        dist = np.linalg.norm(xy_end[i] - xy_start[i])

        if dist < 0.5:
            files_to_remove.append(filename)
            continue

        if counts[c] >= args.max_count:
            files_to_remove.append(filename)
        else:
            files_to_keep.append(filename)
            counts[c] += 1

    counter += yaw_diff.shape[0]

print("counts", counts)
print("len(files_to_remove)", len(files_to_remove))
print("len(files_to_keep)", len(files_to_keep))

if args.remove:
    print("Removing files")
    for filename in files_to_remove:
        folder = args.test_data_path
        path = os.path.join(folder, filename)
        os.remove(path)
