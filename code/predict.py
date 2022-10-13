import torch

torch.multiprocessing.set_sharing_strategy('file_system')
import subprocess
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


def get_last_file(path):
    list_of_files = glob.glob(f'{path}/*')
    if len(list_of_files) == 0:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


try:
    models_path = os.path.join(config["model"]["path"], config["model"]["name"])
    last_checkpoint = get_last_file(models_path)
    test_dataloader = get_dataloader(config["test"]["data_config"])
    model = MultiPathPP(config["model"])
    model.cuda()
    if last_checkpoint is not None:
        model.load_state_dict(torch.load(last_checkpoint)["model_state_dict"])
        num_steps = torch.load(last_checkpoint)["num_steps"]
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("LOADED ", last_checkpoint, "with N params=", params)
    model.eval()


except Exception as e:
    # ToDo: deal with this
    raise e


def generate_filename(scene_data, agent_index):
    scenario_id = scene_data["scenario_id"][agent_index]
    agent_id = scene_data["agent_id"][agent_index]
    agent_type = scene_data["target/agent_type"][agent_index]
    return f"scid_{scenario_id}__aid_{agent_id}__atype_{agent_type.item()}.npz"


savefolder = config["test"]["output_config"]["out_path"]

if not os.path.exists(savefolder):
    os.mkdir(savefolder)

for data in tqdm(test_dataloader):
    if config["test"]["normalize"]:
        data = normalize(data, config, split="test")
    dict_to_cuda(data)
    probs, coordinates, _, _ = model(data)
    probs = probs.detach().cpu()

    coordinates = coordinates * 10. + torch.Tensor([1.4715e+01, 4.3008e-03]).cuda()
    coordinates = coordinates.detach().cpu()

    # print("probs.shape", probs.shape)
    # print("coordinates.shape", coordinates.shape)

    for agent_index, agent_id in enumerate(data["agent_id"]):
        filename = generate_filename(data, agent_index)
        savedata = {
            "scenario_id": data["scenario_id"][agent_index],
            "agent_id": data["agent_id"][agent_index],
            "agent_type": data["target/agent_type"][agent_index],
            "coordinates": coordinates[agent_index],
            "probabilities": probs[agent_index]
        }
        np.savez_compressed(os.path.join(savefolder, filename), **savedata)

    # print(data.keys())
    # for key in data.keys():
    #     print(f"len {key}, {len(data[key])}")
    # exit()
