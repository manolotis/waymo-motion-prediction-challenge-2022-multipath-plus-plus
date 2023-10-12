# ToDo: config file
import os
import numpy as np
import matplotlib.pyplot as plt

predictions_path = "/home/manolotis/sandbox/robustness_benchmark/multipathPP/predictions/"
data_path = "/home/manolotis/sandbox/robustness_benchmark/multipathPP/data/prerendered/test/"
MAX = 10

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
    segments = scene_data["road_network_segments"]
    plt.scatter(segments[:, 0, 0], segments[:, 0, 1], color="black", s=0.1)

    future = prediction["target/future/xy"].squeeze()
    history = prediction["target/history/xy"].squeeze()

    valid_future = prediction["target/future/valid"].squeeze() > 0
    valid_history = prediction["target/history/valid"].squeeze() > 0

    plt.scatter(history[valid_history, 0], history[valid_history, 1], label="history", alpha=0.5)
    plt.scatter(future[valid_future, 0], future[valid_future, 1], label="future", alpha=0.5)

    all_x = np.concatenate([
        prediction["coordinates"][..., 0].flatten(),
        history[valid_history, 0],
        future[valid_future, 0]]
    )

    all_y = np.concatenate([
        prediction["coordinates"][..., 1].flatten(),
        history[valid_history, 1],
        future[valid_future, 1]]
    )

    for i, mode in enumerate(prediction["coordinates"]):
        plt.scatter(mode[:, 0], mode[:, 1], label=f"p={prediction['probabilities'][i]:.3f}", s=15, alpha=0.5)

    padding = 10
    xlim = (all_x.min() - padding, all_x.max() + padding)
    ylim = (all_y.min() - padding, all_y.max() + padding)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.tight_layout()
    plt.show()
