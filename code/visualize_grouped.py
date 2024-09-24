"""
Visualizes grouped predictions according to some filtering of their evaluations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils.general_utils import get_scene_filename, generate_filename, load_scene_data
from utils.general_utils import load_predictions_and_group
from tqdm import tqdm
import multiprocessing

evaluations_path = "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/evals/"

evaluation_files = sorted([os.path.join(evaluations_path, file) for file in os.listdir(evaluations_path)])
evals = [np.load(file) for file in evaluation_files]
file2eval = {}
for f, e in zip(evaluation_files, evals):
    file2eval[f] = e
n_evals = len(evals)

MEAN_TOP1_RANK_SWITCHES = 9
MEAN_TOP1_RANK_SWITCHES_COUNTER = 6.5
MEAN_TOP1_ADE_MEAN = 1.25

MULTIPLIER = 2
MEAN_TOP1_RANK_SWITCHES_THRESHOLD = MULTIPLIER * MEAN_TOP1_RANK_SWITCHES
MEAN_TOP1_RANK_SWITCHES_COUNTER_THRESHOLD = MULTIPLIER * MEAN_TOP1_RANK_SWITCHES_COUNTER
MEAN_TOP1_ADE_MEAN_THRESHOLD = MULTIPLIER * MEAN_TOP1_ADE_MEAN


def get_files_to_visualize(file2eval):
    files_to_visualize = []

    for file, eval in file2eval.items():
        if not (eval["top1_rank_switches"] > MEAN_TOP1_RANK_SWITCHES_THRESHOLD and eval[
            "top1_rank_switches_ade_mean"] > MEAN_TOP1_ADE_MEAN_THRESHOLD and eval[
                    "top1_rank_switches_counter"] > MEAN_TOP1_RANK_SWITCHES_COUNTER_THRESHOLD):
            continue
        files_to_visualize.append(file)

    return files_to_visualize


files_to_visualize = get_files_to_visualize(file2eval)

predictions_path = "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/predictions/final_RoP_Cov_Single__18c3cff/"
data_path = "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/data/prerendered/validation/"


scenes_agents_to_visualize_strings = []

for f in files_to_visualize:
    filename = f.split("/")[-1].replace(".npz", "")
    scid, aid, atype, t = filename.split("__")
    # scid, aid, atype = scid.split("_")[-1], aid.split("_")[-1], atype.split("_")[-1]
    scenes_agents_to_visualize_strings.append(f"{scid}__{aid}")
    print(scid, aid, atype, t)
    # print(f"{scid}__{aid}")


#####################################################
prediction_files = sorted([os.path.join(predictions_path, file) for file in os.listdir(predictions_path)])
predictions = []
limits = {} # min-max limits to visualize for (scene_id, agent_id) combinations

for file in prediction_files:
    f = file.split("/")[-1].replace(".npz", "")
    scid, aid, atype, t = f.split("__")

    substring_filter = f"{scid}__{aid}"
    if substring_filter not in scenes_agents_to_visualize_strings:
        continue

    # Temporary filters. ToDO: remove
    # if scid != "scid_2e74092aca2c90d3" and aid != 12:
    #     continue
    # if not (t == "t_11" or t == "t_40"):
    #     continue
    #############################

    pred = np.load(file)
    predictions.append(pred)

    # Get min-max limits for each scene,agent prediction
    future = pred["target/future/xy"].squeeze()
    history = pred["target/history/xy"].squeeze()
    valid_future = pred["target/future/valid"].squeeze() > 0
    valid_history = pred["target/history/valid"].squeeze() > 0


    all_x = np.concatenate([
        pred["coordinates"][..., 0].flatten(),
        history[valid_history, 0],
        future[valid_future, 0]]
    )

    all_y = np.concatenate([
        pred["coordinates"][..., 1].flatten(),
        history[valid_history, 1],
        future[valid_future, 1]]
    )

    min_x = min(all_x)
    min_y = min(all_y)
    max_x = max(all_x)
    max_y = max(all_y)

    scid_aid_key = (scid.split("_")[-1], int(aid.split("_")[-1]))

    if not scid_aid_key in limits:
        limits[scid_aid_key] = {
            "x": [9999999999, -9999999999],
            "y": [9999999999, -9999999999],
        }


    limits[scid_aid_key]["x"][0] = min(min_x, limits[scid_aid_key]["x"][0])
    limits[scid_aid_key]["x"][1] = max(max_x, limits[scid_aid_key]["x"][1])
    limits[scid_aid_key]["y"][0] = min(min_y, limits[scid_aid_key]["y"][0])
    limits[scid_aid_key]["y"][1] = max(max_y, limits[scid_aid_key]["y"][1])


# for prediction in predictions:

def process_and_plot_grouped_prediction(prediction):
    scene_data = load_scene_data(prediction, data_path)
    segments = scene_data["road_network_segments"]
    embeddings = scene_data["road_network_embeddings"]
    # print(segments.shape)
    segment_types = np.argmax(embeddings[:,0,-20:], axis=1) # assuming you have n-by-m arr

    is_laneCenterFreeway = segment_types == 1
    is_laneCenterSurface = segment_types == 2
    is_laneCenterBike = segment_types == 3

    is_laneCenter = is_laneCenterSurface | is_laneCenterFreeway

    # is_brokenSingleWhite = segment_types == 6
    # is_solidSingleWhite = segment_types == 7
    # is_solidDoubleYellow = segment_types == 12
    is_laneBoundary = (segment_types >= 6) & (segment_types <= 13)

    is_roadEdgeBoundary = segment_types == 15
    is_roadEdgeMedian = segment_types == 16
    is_roadEdge = is_roadEdgeBoundary | is_roadEdgeMedian
    is_other = segment_types >= 17 # stop sign, crosswalk, speedbump

    # print("segments", segments.shape)
    # print("types", segment_types.shape)
    # print("is_roadEdgeBoundary", is_roadEdgeBoundary.shape)
    # unique, counts = np.unique(segment_types, return_counts=True)
    # print("unique", unique)
    # print("counts", counts)
    # print()
    # print("types", segment_types)
    # print(embeddings.shape)
    # print(embeddings[-5:,0,-20:])
    # print(embeddings[:,0,-18].sum())
    plt.scatter(segments[is_laneBoundary, 0, 0], segments[is_laneBoundary, 0, 1], color="grey", s=0.7)
    plt.scatter(segments[is_roadEdge, 0, 0], segments[is_roadEdge, 0, 1], color="black", s=0.7)
    plt.scatter(segments[is_laneCenter, 0, 0], segments[is_laneCenter, 0, 1], color="green", s=0.7)
    plt.scatter(segments[is_laneCenterBike, 0, 0], segments[is_laneCenterBike, 0, 1], color="purple", s=0.7)
    plt.scatter(segments[is_other, 0, 0], segments[is_other, 0, 1], color="red", s=5)
    # plt.scatter(segments[:, 0, 0], segments[:, 0, 1], color="black", s=0.3)
    # plt.scatter(segments[:, 0, 0], segments[:, 0, 1], c=segment_types, s=0.1)
    # exit()

    future = prediction["target/future/xy"].squeeze()
    history = prediction["target/history/xy"].squeeze()

    valid_future = prediction["target/future/valid"].squeeze() > 0
    valid_history = prediction["target/history/valid"].squeeze() > 0


    plt.scatter(history[valid_history, 0], history[valid_history, 1], label="history", alpha=0.5)
    plt.scatter(future[valid_future, 0], future[valid_future, 1], label="future", alpha=0.5)


    for i, mode in enumerate(prediction["coordinates"]):
        a = 0.7 if i == 0 else 0.2
        plt.scatter(mode[:, 0], mode[:, 1], label=f"p={prediction['probabilities'][i]:.3f}", s=15, alpha=a)

    scene_filename = get_scene_filename(prediction)
    scid = prediction["scenario_id"].item()
    aid = prediction["agent_id"].item()
    t = prediction["timestep"].item()

    savename = scene_filename.replace(".npz", ".png")

    padding = 2
    # xlim = (all_x.min() - padding, all_x.max() + padding)
    # ylim = (all_y.min() - padding, all_y.max() + padding)
    # print(limits)
    xlim = (limits[(scid, aid)]["x"][0] - padding, limits[(scid, aid)]["x"][1] + padding)
    ylim = (limits[(scid, aid)]["y"][0] - padding, limits[(scid, aid)]["y"][1] + padding)
    # xlim = limits[(scid, aid)]["x"]
    # ylim = limits[(scid, aid)]["y"]

    plt.xlim(xlim)
    plt.ylim(ylim)
    figtitle = f"Scene {scid}, Agent {aid}, Timestep {t}"
    plt.title(figtitle)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(os.path.join("/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/viz/img", savename))
    # plt.show()
    plt.close()
    # exit()


for prediction in tqdm(predictions):
    process_and_plot_grouped_prediction(prediction)




# Multiprocessing is not much faster than single process...
# p = multiprocessing.Pool(12)
# processes = []
# for prediction in tqdm(predictions):
#     process_and_plot_grouped_prediction(prediction)
#     processes.append(
#         p.apply_async(
#             process_and_plot_grouped_prediction,
#             kwds= dict(
#                 prediction=prediction
#             )
#         )
#     )
#
# for r in tqdm(processes):
#     r.get()




"""
Num evaluations:  1367
rank_switches_counter
	-min 1
	-max 52
	-mean 20.776883686905634
	-std 10.118787870934018
top1_rank_switches
	-min 1
	-max 59
	-mean 9.078273591806877
	-std 8.474870324211425
top1_rank_switches_ade_sum
	-min 0.03246651217341423
	-max 152.71599712967873
	-mean 8.832034516273968
	-std 11.42639031564148
top1_rank_switches_ade_mean
	-min 0.02009209245443344
	-max 13.834274291992188
	-mean 1.2494108333540557
	-std 1.1608659760625861
top1_rank_switches_counter
	-min 1
	-max 30
	-mean 6.485003657644477
	-std 4.896697495315631
counter_top1_rank_switches 152
counter_ade_mean 76
counter_both 14
"""
