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
    timestep = scene_data["timestep"][agent_index]
    return f"scid_{scenario_id}__aid_{agent_id}__atype_{int(agent_type.item())}__t_{timestep}.npz"


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

    probs = torch.nn.functional.softmax(probs, dim=1)

    # print("probs shape", probs.shape)
    # print("coordinates shape", coordinates.shape)
    # print("covariance_matrices shape", covariance_matrices.shape)

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
    index_sort_broadcast_cov = index_sort.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, covariance_matrices.size(2), covariance_matrices.size(3), covariance_matrices.size(4))

    # Sort covariance_matrices based on the indices from probs
    covariance_matrices_sorted = torch.gather(covariance_matrices, 1, index_sort_broadcast_cov)

    # print("probs_sorted shape", probs_sorted.shape)
    # print("coordinates_sorted shape", coordinates_sorted.shape)
    # print("covariance_matrices shape", covariance_matrices.shape)

    # exit()
    # loss_coeff = loss_coeff[index_sort]

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
            "target/future/xy": data_original["target/future/xy"][agent_index],
            "other/future/xy": data_original["other/future/xy"][agent_index],
            "target/history/valid": data_original["target/history/valid"][agent_index],
            "other/history/valid": data_original["other/history/valid"][agent_index],
            "target/future/valid": data_original["target/future/valid"][agent_index],
            "other/future/valid": data_original["other/future/valid"][agent_index],
            "covariance_matrix": covariance_matrices[agent_index]
        }
        np.savez_compressed(os.path.join(savefolder, filename), **savedata)
