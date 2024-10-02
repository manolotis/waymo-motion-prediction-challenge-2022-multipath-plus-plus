import copy
import torch

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

TO_PREDICT = set([
    ("f677b0c413db52aa", 107, 11) # scid_f677b0c413db52aa__aid_107__atype_1__t_11
])


def get_last_checkpoint(path):
    list_of_files = glob.glob(f'{path}/*')
    if len(list_of_files) == 0:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def get_best_checkpoint(path):
    list_of_files = glob.glob(f'{path}/*')
    for f in list_of_files:
        if "best" in f:
            return f
    return None


try:
    models_path = os.path.join(config["model"]["path"], config["model"]["name"])
    # checkpoint_path = get_last_checkpoint(models_path)
    checkpoint_path = get_best_checkpoint(models_path)
    test_dataloader = get_dataloader(config["test"]["data_config"], None)
    # test_dataloader = get_dataloader(config["test"]["data_config"], TO_PREDICT)
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
    timestep = scene_data["timestep"][agent_index]
    return f"scid_{scenario_id}__aid_{agent_id}__atype_{int(agent_type.item())}__t_{timestep}.npz"


model_name = config["model"]["name"]
if "name_addition" in config["model"]:
    model_name = config["model"]["name"] + "_" + config["model"]["name_addition"]

savefolder = os.path.join(config["test"]["output_config"]["out_path"], model_name)

if not os.path.exists(savefolder):
    os.makedirs(savefolder, exist_ok=True)

for data in tqdm(test_dataloader):

    if (len(data["scenario_id"]) != 1):
        raise ValueError("Batch size should be 1") # ToDo: allow higher for efficiency

    sid = data["scenario_id"][0]
    aid = int(data["agent_id"][0])
    t = int(data["timestep"][0])

    tup = (sid, aid, t)
    # if tup not in TO_PREDICT:
    #     continue

    print("Found: ", tup)

    data["target/width"][0] = -1
    data["target/length"][0] = -1

    for key in data.keys():
        print("-----------------------------")
        print(key, type(data[key]))
        if "other" not in key and "mcg" not in key and "future" not in key:
            print(data[key])
        try:
            print("\t data[key][0].shape", data[key][0].shape)
            print("\t data[key].shape", data[key].shape)
        except TypeError:
            print("\t", data[key])
        except AttributeError:
            print("\t", data[key])

    # exit()

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
    try:
        probs, coordinates, covariance_matrices, loss_coeff = model(data)
    except RuntimeError as e:
        print("There was a runtime error. Skipping... ")
        raise e
        continue

    probs = torch.nn.functional.softmax(probs, dim=1)

    # Step 1: sort probs
    index_sort = torch.argsort(probs, dim=1, descending=True)  # Sort along dimension 1

    # Step 2: Sort probs
    probs_sorted = torch.gather(probs, 1, index_sort)

    # Step 3: Sort coordinates (shape: [8, 6, 80, 2])
    # We expand index_sort to match the shape of coordinates
    index_sort_broadcast_coords = index_sort.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, coordinates.size(2),
                                                                                coordinates.size(3))
    # Sort coordinates based on the indices from probs
    coordinates_sorted = torch.gather(coordinates, 1, index_sort_broadcast_coords)

    #  Step 4: Sort covariance_matrices (shape: [8, 6, 80, 2, 2])
    # Expand index_sort for covariance_matrices
    index_sort_broadcast_cov = index_sort.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1,
                                                                                           covariance_matrices.size(2),
                                                                                           covariance_matrices.size(3),
                                                                                           covariance_matrices.size(4))

    # Sort covariance_matrices based on the indices from probs
    covariance_matrices_sorted = torch.gather(covariance_matrices, 1, index_sort_broadcast_cov)

    probs = probs_sorted
    coordinates = coordinates_sorted
    covariance_matrices = covariance_matrices_sorted

    probs = probs.detach().cpu()

    covariance_matrices = covariance_matrices.detach().cpu()

    coordinates = coordinates * 10. + torch.Tensor([1.4715e+01, 4.3008e-03]).cuda()
    coordinates = coordinates.detach().cpu()

    for agent_index, agent_id in enumerate(data["agent_id"]):
        agent_id_filter = data["agent_id"][agent_index]

        # if agent_id_filter != 130:
        # # if agent_id_filter != 128 and agent_id_filter != 130:
        #     continue # ToDo: remove

        filename = generate_filename(data, agent_index)
        savedata = {
            "timestep": data["timestep"][agent_index],
            "scenario_id": data["scenario_id"][agent_index],
            "agent_id": data["agent_id"][agent_index],
            "agent_type": data["target/agent_type"][agent_index].flatten(),
            "coordinates": coordinates[agent_index],
            "probabilities": probs[agent_index],
            "target/history/xy": data_original["target/history/xy"][agent_index],
            "other/history/xy": data_original["other/history/xy"][agent_index],

            "target/history/valid": data_original["target/history/valid"][agent_index],
            "other/history/valid": data_original["other/history/valid"][agent_index],

            "covariance_matrix": covariance_matrices[agent_index]
        }

        if "target/future/xy" in data_original:
            savedata["target/future/xy"] = data_original["target/future/xy"][agent_index]
            savedata["other/future/xy"] = data_original["other/future/xy"][agent_index]
            savedata["target/future/valid"] = data_original["target/future/valid"][agent_index]
            savedata["other/future/valid"] = data_original["other/future/valid"][agent_index]


        # # print("segments shape", data["road_network_segments"].shape)
        # # print("segments embeddings", data["road_network_embeddings"].shape)
        # # # print("segments types", data["road_network_segments_types"].shape)
        # # print("coordinates", coordinates.shape)
        # # print("data original taget history xy", data_original["target/history/xy"].shape)
        # segments = data["road_network_segments"].cpu()
        # embeddings = data["road_network_embeddings"].cpu()
        # # print(segments.shape)
        # # print(embeddings[:, 0, -20:].shape)
        # segment_types = np.argmax(embeddings[:, 0, -20:], axis=1)  # assuming you have n-by-m arr
        #
        # plt.scatter(segments[:, 0, 0], segments[:, 0, 1], color="black", s=0.3)
        #
        # future = (data["target/future/xy"][agent_index].squeeze()).cpu()
        # history = (data["target/history/xy"][agent_index].squeeze()).cpu()
        # # history_other = data["other/history/xy"].squeeze()
        #
        # valid_future = (data["target/future/valid"][agent_index].squeeze() > 0).cpu()
        # valid_history = (data["target/history/valid"][agent_index].squeeze() > 0).cpu()
        #
        # # print("future", future.shape)
        # # print("history", history.shape)
        # #
        # # print("valid_future", valid_future.shape)
        # # print("valid_history", valid_history.shape)
        # #
        # # plt.scatter(history[valid_history, 0], history[valid_history, 1], label="history", s=6, alpha=0.5)
        # # plt.scatter(future[valid_future, 0], future[valid_future, 1], label="history", s=6, alpha=0.5)
        # #
        # # num_modes, num_timesteps, num_features = coordinates[agent_index].shape
        # # for i in range(num_modes):
        # #     plt.scatter(coordinates[agent_index][i, :, 0], coordinates[agent_index][i, :, 1], s=6)
        # #
        # # plt.xlabel("x [m]")
        # # plt.ylabel("y [m]")
        # # plt.tight_layout()
        # #
        # # # plt.show()
        # #
        # # exit()

        # np.savez_compressed(os.path.join(savefolder, filename), **savedata)

        pred_coordinates = savedata["coordinates"].detach().cpu()


        probs = savedata["probabilities"].detach().cpu()
        print("pred coordinates", pred_coordinates.shape)
        print("probs ", probs.shape)

        pred_coordinates = np.expand_dims(pred_coordinates, axis=0)
        probs = np.expand_dims(probs, axis=0)

        segments = data["road_network_segments"].detach().cpu()
        embeddings = data["road_network_embeddings"].detach().cpu()
        segment_types = np.argmax(embeddings[:, 0, -20:], axis=1)  # assuming you have n-by-m arr

        is_solid = segment_types == 7
        unique, counts = np.unique(segment_types, return_counts=True)

        is_none = segment_types == 0
        is_laneCenterFreeway = segment_types == 1
        is_laneCenterSurface = segment_types == 2
        is_laneCenterBike = segment_types == 3

        is_laneCenter = is_laneCenterSurface | is_laneCenterFreeway
        is_laneBoundary = (segment_types >= 6) & (segment_types <= 13)
        is_roadEdgeBoundary = segment_types == 15
        is_roadEdgeMedian = segment_types == 16
        is_roadEdge = is_roadEdgeBoundary | is_roadEdgeMedian
        is_other = segment_types >= 17  # stop sign, crosswalk, speedbump

        plt.scatter(segments[is_none, 0, 0], segments[is_none, 0, 1], color="red", marker="x", s=0.7)
        plt.scatter(segments[is_laneBoundary, 0, 0], segments[is_laneBoundary, 0, 1], color="grey", s=0.7)
        plt.scatter(segments[is_roadEdge, 0, 0], segments[is_roadEdge, 0, 1], color="black", s=0.7)
        plt.scatter(segments[is_laneCenter, 0, 0], segments[is_laneCenter, 0, 1], color="green", s=0.7)
        plt.scatter(segments[is_laneCenterBike, 0, 0], segments[is_laneCenterBike, 0, 1], color="purple", s=0.7)
        plt.scatter(segments[is_other, 0, 0], segments[is_other, 0, 1], color="red", s=5)

        # plt.scatter(segments[:, 0, 0], segments[:, 0, 1], color="grey", s=0.3)
        plt.scatter(segments[is_solid, 0, 0], segments[is_solid, 0, 1], color="black", s=1)

        history = data["target/history/xy"].squeeze()

        keys_to_print = [
            "shift",
            "yaw",
            "target/history/valid",
            "target/history/xy",
            "target/history/yaw",
            "target/history/speed",
            "target/history/yaw_sin",
            "target/history/yaw_cos",
            "target/history/length",
            "target/history/width",

            # "target/history/xy_diff",
            # "target/history/yaw_diff",
            # "target/history/speed_diff",
            # "target/history/yaw_sin_diff",
            # "target/history/yaw_cos_diff",
        ]
        for k in keys_to_print:
            print(k)
            print(np.round(data[k], decimals=2))

        print("")


        history_other = data["other/history/xy"].squeeze()

        valid_history = (data["target/history/valid"].squeeze() > 0)

        plt.scatter(history[valid_history, 0], history[valid_history, 1], label="history", s=6, alpha=0.5)
        plt.scatter(history[valid_history, 0][-1], history[valid_history, 1][-1], label="current",
                    s=15)  # current position in different color

        for other_idx in range(history_other.shape[0]):
            hist = history_other[other_idx]
            hist_valid = data["other/history/valid"][other_idx].squeeze() > 0
            plt.scatter(hist[hist_valid, 0], hist[hist_valid, 1], s=6, alpha=1, color="red")

        if pred_coordinates is not None:
            num_agents, num_modes, num_timesteps, num_features = pred_coordinates.shape
            # num_modes, num_timesteps, num_features = pred_coordinates.shape
            for i in range(num_modes):
                if probs is not None:
                    plt.scatter(pred_coordinates[0, i, :, 0], pred_coordinates[0, i, :, 1], alpha=0.3, s=6,
                                label=f"p={probs[0, i]:.3f}")
                else:
                    plt.scatter(pred_coordinates[0, i, :, 0], pred_coordinates[0, i, :, 1], alpha=0.3, s=6)

        # plt.scatter(history_other[valid_history_other, 0], history_other[valid_history_other, 1], label="history", s=3, alpha=0.5)
        # plt.scatter(future[valid_future, 0], future[valid_future, 1], label="future", alpha=0.5)

        padding = 3

        figtitle = f"{data['scenario_id'][0]} | Agent {agent_id} | Type {data['target/agent_type'][0]}"

        # plt.xlim(xlim[0] - padding, xlim[1] + padding)
        # plt.ylim(ylim[0] - padding, ylim[1] + padding)

        plt.title(figtitle)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.legend(loc="upper right")
        plt.tight_layout()

        # plt.gca().invert_yaxis()
        # savepath = os.path.join(CARLA_EXAMPLES_DIR, figtitle)
        # savepath = os.path.join(PROJECT_DIR, f"viz/png/last/no_lanes__aid_{agent_id}__t_{timestep}.png")
        # savepath = os.path.join(PROJECT_DIR, f"viz/png/last/aid_{agent_id}__t_{timestep}.png")
        # plt.savefig(savepath)
        plt.show()
        plt.close()
