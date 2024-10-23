import copy
import torch
from requests.packages import target

torch.multiprocessing.set_sharing_strategy('file_system')
from model.multipathpp import MultiPathPP
from model.data import get_dataloader, dict_to_cuda, normalize
import os
import glob
import random
from utils.predict_utils import parse_arguments, get_config
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    # target/history/lstm_data
    # target/history/lstm_data_diff
    # other/history/lstm_data
    # other/history/lstm_data_diff
    # target/history/mcg_input_data
    # other/history/mcg_input_data
    # road_network_embeddings

    # if features == ("xy", "yaw", "speed", "width", "length", "valid"):
    #     normalizarion_means = {
    #         "target/history/lstm_data": np.array(
    #             [-2.9633283615112305, 0.005309064872562885, -0.003220283193513751, 6.059159278869629,
    #              1.9252972602844238, 4.271720886230469, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    #         "target/history/lstm_data_diff": np.array(
    #             [0.5990215539932251, -0.0018718164646998048, 0.0006288147415034473, 0.0017819292843341827, 0.0, 0.0,
    #              0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    #         "other/history/lstm_data": np.array(
    #             [5.601348876953125, 1.4943491220474243, -0.013019951991736889, 1.44475519657135, 1.072572946548462,
    #              2.4158480167388916, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    #         "other/history/lstm_data_diff": np.array(
    #             [0.025991378352046013, -0.0008657555445097387, 9.549396054353565e-05, 0.001465122913941741, 0.0, 0.0,
    #              0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    #         "target/history/mcg_input_data": np.array(
    #             [-2.9633283615112305, 0.005309064872562885, -0.003220283193513751, 6.059159278869629,
    #              1.9252972602844238, 4.271720886230469, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #              0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    #         "other/history/mcg_input_data": np.array(
    #             [5.601348876953125, 1.4943491220474243, -0.013019951991736889, 1.44475519657135, 1.072572946548462,
    #              2.4158480167388916, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             dtype=np.float32),
    #         "road_network_embeddings": np.array(
    #             [77.35582733154297, 0.12082172930240631, 0.05486442521214485, 0.004187341313809156,
    #              -0.0015162595082074404, 2.011558771133423, 0.9601883888244629, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    #     }
    #     normalizarion_stds = {
    #         "target/history/lstm_data": np.array(
    #             [3.738459825515747, 0.11283490061759949, 0.10153655707836151, 5.553133487701416, 0.5482628345489502,
    #              1.6044323444366455, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    #         "target/history/lstm_data_diff": np.array(
    #             [0.5629324316978455, 0.03495170176029205, 0.04547161981463432, 0.5762772560119629, 1.0, 1.0, 1.0, 1.0,
    #              1.0, 1.0, 1.0], dtype=np.float32),
    #         "other/history/lstm_data": np.array(
    #             [33.899658203125, 25.64937973022461, 1.3623465299606323, 3.8417460918426514, 1.0777146816253662,
    #              2.4492409229278564, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    #         "other/history/lstm_data_diff": np.array(
    #             [0.36061710119247437, 0.1885228455066681, 0.08698483556509018, 0.43648791313171387, 1.0, 1.0, 1.0, 1.0,
    #              1.0, 1.0, 1.0], dtype=np.float32),
    #         "target/history/mcg_input_data": np.array(
    #             [3.738459825515747, 0.11283490061759949, 0.10153655707836151, 5.553133487701416, 0.5482628345489502,
    #              1.6044323444366455, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #             dtype=np.float32),
    #         "other/history/mcg_input_data": np.array(
    #             [33.899658203125, 25.64937973022461, 1.3623465299606323, 3.8417460918426514, 1.0777146816253662,
    #              2.4492409229278564, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #             dtype=np.float32),
    #         "road_network_embeddings": np.array(
    #             [36.71162414550781, 0.761500358581543, 0.6328969597816467, 0.7438802719116211, 0.6675100326538086,
    #              0.9678668975830078, 1.1907216310501099, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    #              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    #     }

all_target_x = []
future_target_x = []
all_target_y = []
future_target_y = []
all_target_speed = []
all_yaw_diff = []
counter = 0

all_target_lstm_data = []
all_target_lstm_data_diff = []
all_other_lstm_data = []
all_other_lstm_data_diff = []
all_target_mcg_input_data = []
all_other_mcg_input_data = []
all_road_network_embeddings = []

all_keys = [
    "all_target_lstm_data",
    "all_target_lstm_data_diff",
    "all_other_lstm_data",
    "all_other_lstm_data_diff",
    "all_target_mcg_input_data",
    "all_other_mcg_input_data",
    "all_road_network_embeddings",

]

for data in tqdm(dataloader):

    yaw_diff = data["target/future/yaw"][:, -1, 0] - data["target/history/yaw"][:, 0, 0]

    yaw_diff[yaw_diff < np.deg2rad(-180)] += np.deg2rad(360)
    yaw_diff[yaw_diff > np.deg2rad(180)] -= np.deg2rad(360)

    all_yaw_diff.extend(np.rad2deg(yaw_diff))

    for key, value in data.items():
        # print(key)
        # try:
        #     print("\t shape: ", value.shape)
        # except AttributeError:
        #     print("\t type: ", type(value))

        # should_continue = "target" in ""
        if key == "target/history/xy" or key == "target/future/xy":
            # if key == "target/history/xy" :
            all_target_x.extend(value[:, :, 0].flatten())
            all_target_y.extend(value[:, :, 1].flatten())

            if "future" in key:
                future_target_x.extend(value[:, :, 0].flatten())
                future_target_y.extend(value[:, :, 1].flatten())

        if key == "target/history/speed" or key == "target/future/speed":
            # if key == "target/history/speed":
            all_target_speed.extend(value[:, :, 0].flatten())

        if key == "target/history/lstm_data":
            all_target_lstm_data.extend(value[:, :, :6].tolist())

        if key == "target/history/lstm_data_diff":
            all_target_lstm_data_diff.extend(value[:, :, :4].tolist())

        if key == "other/history/lstm_data":
            all_other_lstm_data.extend(value[:, :, :6].tolist())

        if key == "other/history/lstm_data_diff":
            all_other_lstm_data_diff.extend(value[:, :, :4].tolist())

        if key == "target/history/mcg_input_data":
            all_target_mcg_input_data.extend(value[:, :, :6].tolist())

        if key == "other/history/mcg_input_data":
            all_other_mcg_input_data.extend(value[:, :, :6].tolist())

        if key == "road_network_embeddings":
            all_road_network_embeddings.extend(value[:, :, :7].tolist())

    counter += yaw_diff.shape[0]
    if counter > args.max_count:
        break

variables = {
    "all_target_x": all_target_x,
    "all_target_y": all_target_y,
    "future_target_x": future_target_x,
    "future_target_y": future_target_y,
    "all_target_speed": all_target_speed,

}

for k, v in variables.items():
    print(k)
    print(f"\tshape", np.array(v).shape)
    print(f"\tmean", np.array(v).mean())
    print(f"\tstd", np.array(v).std())

variables_to_aggregate = {
    "all_target_lstm_data": np.array(all_target_lstm_data),
    "all_target_lstm_data_diff": np.array(all_target_lstm_data_diff),
    "all_other_lstm_data": np.array(all_other_lstm_data),
    "all_other_lstm_data_diff": np.array(all_other_lstm_data_diff),
    "all_target_mcg_input_data": np.array(all_target_mcg_input_data),
    "all_other_mcg_input_data": np.array(all_other_mcg_input_data),
    "all_road_network_embeddings": np.array(all_road_network_embeddings),
}

for k, v in variables_to_aggregate.items():
    print(f"{k}")
    print(f"\tshape", v.shape)
    print(f"\tmean", v.mean(axis=(0, 1)))
    print(f"\tstd", v.std(axis=(0, 1)))

print('np array of all_target_lstm_data shape', np.array(all_target_lstm_data).shape)

#
# plt.scatter(all_target_x, all_target_y, c=all_target_speed, cmap='coolwarm', s=1, alpha=0.3, vmin=0, vmax=24)
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.colorbar(label='speed [m/s]')
# plt.tight_layout()
# plt.show()
