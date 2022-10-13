# ToDo: config file
import os
import numpy as np
import matplotlib.pyplot as plt

predictions_path = "/home/manolotis/sandbox/robustness_benchmark/multipathPP/predictions/"
data_path = "/home/manolotis/sandbox/robustness_benchmark/multipathPP/data/prerendered/test/"
MAX = 20

prediction_files = [os.path.join(predictions_path, file) for file in os.listdir(predictions_path)]
predictions = [np.load(file) for file in prediction_files]
predictions = predictions[:MAX] if MAX is not None else predictions


def load_scene_data(prediction):
    scenario_id = prediction["scenario_id"]
    agent_id = prediction["agent_id"]
    agent_type = prediction["agent_type"]

    data_filename = f"scid_{scenario_id}__aid_{agent_id}__atype_{int(agent_type)}.npz"
    scene_data = np.load(os.path.join(data_path, data_filename))
    return scene_data


for prediction in predictions:
    scene_data = load_scene_data(prediction)
    segments=scene_data["road_network_segments"]
    # plt.scatter(segments[:, 0, 0], segments[:, 0, 1], color="black", s=0.1)

    plt.scatter(scene_data["target/history/xy"][...,0], scene_data["target/history/xy"][...,1])
    plt.scatter(scene_data["target/future/xy"][...,0], scene_data["target/future/xy"][...,1])
    # plt.scatter()


    for i, mode in enumerate(prediction["coordinates"]):
        plt.scatter(mode[:, 0], mode[:, 1], label=f"p={prediction['probabilities'][i]:.4f}")
    plt.legend()
    plt.show()
