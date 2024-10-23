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

# Waymo Values
# MEAN_X = 1.4715e+01
# MEAN_Y = 4.3008e-03
# STD_XY = 10.

# CARLA Behavior Agent values (in Town05)
MEAN_X = 20.424562
MEAN_Y = 0.0039684023
STD_XY = (20.241842 + 14.278944) / 2.0


def get_last_checkpoint(path):
    print("Looking for last checkpoint in ", path)
    list_of_files = glob.glob(f'{path}/*')
    print("List of files: ", list_of_files)
    if len(list_of_files) == 0:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def get_best_checkpoint(path):
    print("Looking for best checkpoint in ", path)
    list_of_files = glob.glob(f'{path}/*')
    print("List of files: ", list_of_files)
    for f in list_of_files:
        if "best" in f and "old" not in f:
            return f
    return None
# /home/manolotis/sandbox/multipathpp/code/trained_models/final_RoP_Cov_Single__aa8678f__from_trained
# /home/manolotis/sandbox/multipathpp/code/trained_models/final_RoP_Cov_Single__aa8678f_from_trained
try:
    models_path = os.path.join(config["model"]["path"], config["model"]["name"])
    # checkpoint_path = get_last_checkpoint(models_path)
    checkpoint_path = get_best_checkpoint(models_path)
    test_dataloader = get_dataloader(config["test"]["data_config"])
    model = MultiPathPP(config["model"])
    model.cuda()
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
        num_steps = torch.load(checkpoint_path)["num_steps"]
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("LOADED ", checkpoint_path, "with N params=", params)
    else:
        raise ValueError("Checkpoint was None")
    model.eval()


except Exception as e:
    # ToDo: deal with this
    raise e


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

for data in tqdm(test_dataloader):
    # for key in data.keys():
    #     print(key, type(data[key]))
    #     try:
    #         print("\t", data[key][0].shape)
    #     except TypeError:
    #         print("\t", data[key])
    #     except AttributeError:
    #         print("\t", data[key])

    if config["test"]["normalize"]:
        data_original = copy.deepcopy(data)
        data = normalize(data, config, split="test")
    else:
        data_original = data

    # print("AFTER NORM")
    # for key in data.keys():
    #     print(key, type(data[key]))
    #     try:
    #         print("\t", data[key][0].shape)
    #     except TypeError:
    #         print("\t", data[key])
    #     except AttributeError:
    #         print("\t", data[key])

    dict_to_cuda(data)
    probs, coordinates, covariance_matrices, loss_coeff = model(data)
    probs = probs.detach().cpu()
    covariance_matrices = covariance_matrices.detach().cpu()

    coordinates = coordinates * STD_XY + torch.Tensor([MEAN_X, MEAN_Y]).cuda()
    # ToDo: use new normalization values
    coordinates = coordinates.detach().cpu()

    for agent_index, agent_id in enumerate(data["agent_id"]):
        filename = generate_filename(data, agent_index)
        savedata = {
            "scenario_id": data["scenario_id"][agent_index],
            "agent_id": data["agent_id"][agent_index],
            "agent_type": data["target/agent_type"][agent_index].flatten(),
            "coordinates": coordinates[agent_index],
            "probabilities": probs[agent_index],
            "target/history/xy": data_original["target/history/xy"][agent_index],
            "other/history/xy": data_original["other/history/xy"][agent_index],
            "target/future/xy": data_original["target/future/xy"][agent_index],
            "other/future/xy": data_original["other/future/xy"][agent_index],
            "target/history/valid": data_original["target/history/valid"][agent_index],
            "other/history/valid": data_original["other/history/valid"][agent_index],
            "target/future/valid": data_original["target/future/valid"][agent_index],
            "other/future/valid": data_original["other/future/valid"][agent_index],
            "covariance_matrix": covariance_matrices[agent_index]
        }
        #
        # if agent_index < 3:
        #     segments = data_original["road_network_segments"]
        #
        #     plt.scatter(segments[:, 0, 0], segments[:, 0, 1], color="black", s=0.1)
        #
        #     future = savedata["target/future/xy"].squeeze()
        #     history = savedata["target/history/xy"].squeeze()
        #
        #     print("target history", history)
        #
        #     valid_future = savedata["target/future/valid"].squeeze() > 0
        #     valid_history = savedata["target/history/valid"].squeeze() > 0
        #
        #     plt.scatter(history[valid_history, 0], history[valid_history, 1], alpha=1)
        #     plt.scatter(future[valid_future, 0], future[valid_future, 1], alpha=1, s=16)
        #
        #     all_x = np.concatenate([
        #         savedata["coordinates"][..., 0].flatten(),
        #         history[valid_history, 0],
        #         future[valid_future, 0]]
        #     )
        #
        #     all_y = np.concatenate([
        #         savedata["coordinates"][..., 1].flatten(),
        #         history[valid_history, 1],
        #         future[valid_future, 1]]
        #     )
        #
        #     for i, mode in enumerate(savedata["coordinates"]):
        #         plt.scatter(mode[:, 0], mode[:, 1], label=f"p={savedata['probabilities'][i]:.3f}", s=15,
        #                     alpha=0.3)
        #
        #     plt.scatter(history[valid_history, 0], history[valid_history, 1], label="history", c="blue", s=20, alpha=1)
        #     plt.scatter(future[valid_future, 0], future[valid_future, 1], label="future", c="orange", s=5, alpha=1)
        #
        #     padding = 20
        #     xlim = (all_x.min() - padding, all_x.max() + padding)
        #     ylim = (all_y.min() - padding, all_y.max() + padding)
        #
        #     plt.xlim(xlim)
        #     plt.ylim(ylim)
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.show()

        np.savez_compressed(os.path.join(savefolder, filename), **savedata)
