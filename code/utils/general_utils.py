import numpy as np
import os
from tqdm import tqdm


def generate_filename(scene_data):
    scenario_id = scene_data["scenario_id"]
    agent_id = scene_data["agent_id"]
    agent_type = scene_data["target/agent_type"]
    timestep = scene_data["timestep"]
    return f"scid_{scenario_id}__aid_{agent_id}__atype_{agent_type.item()}__t_{timestep}.npz"


def get_scene_filename(prediction):
    scenario_id = prediction["scenario_id"]
    agent_id = prediction["agent_id"]
    agent_type = prediction["agent_type"]
    timestep = prediction["timestep"]

    data_filename = f"scid_{scenario_id}__aid_{agent_id}__atype_{int(agent_type)}__t_{timestep}.npz"
    return data_filename

def get_grouped_scene_filename(scenario_id, agent_id, agent_type,min_t,max_t):
    data_filename = f"scid_{scenario_id}__aid_{agent_id}__atype_{int(agent_type)}__t_{min_t}-{max_t}.npz"
    return data_filename


def load_scene_data(prediction, data_path):
    data_filename = get_scene_filename(prediction)
    scene_data = np.load(os.path.join(data_path, data_filename))
    return scene_data


def load_predictions_and_group(predictions_path, MAX=None):
    # loads predictions and groups them by scene and agent

    prediction_files = sorted([os.path.join(predictions_path, file) for file in os.listdir(predictions_path)])
    predictions = [np.load(file) for file in prediction_files]
    if MAX is not None:
        predictions = predictions[:MAX] if MAX is not None else predictions

    grouped_predictions = {}
    scene_agent_type_map = {} # key is pair of (scene_id, agent_id) to value agent_type

    for prediction in tqdm(predictions):

        # scene_data = load_scene_data(prediction, data_path) # Not needed for now. This loads the scene for other things (e.g. get road graph data)
        # prediction has the following keys: ['timestep', 'scenario_id', 'agent_id', 'agent_type', 'coordinates', 'probabilities',
        # 'target/history/xy', 'other/history/xy', 'target/future/xy', 'other/future/xy', 'target/history/valid', 'other/history/valid',
        # 'target/future/valid', 'other/future/valid', 'covariance_matrix']

        scenario_id = prediction['scenario_id'].item()
        timestep = prediction['timestep'].item()
        agent_id = prediction['agent_id'].item()
        agent_type = prediction['agent_type'].item()

        if scenario_id not in grouped_predictions:
            grouped_predictions[scenario_id] = {}

        if agent_id not in grouped_predictions[scenario_id]:
            grouped_predictions[scenario_id][agent_id] = {}

        grouped_predictions[scenario_id][agent_id][timestep] = prediction
        scene_agent_type_map[(scenario_id, agent_id)] = agent_type

    return grouped_predictions, scene_agent_type_map