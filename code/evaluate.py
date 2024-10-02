from multipathPP.code.utils.general_utils import get_grouped_scene_filename
from utils.general_utils import get_scene_filename, generate_filename, load_scene_data, get_grouped_scene_filename, \
    load_predictions_and_group
from scipy.optimize import linear_sum_assignment
import numpy as np
import os
from tqdm import tqdm

# MODE = "standard"
# MODE = "simplified_rg"
# MODE = "simplified_rg_no_others"
MODE = "carla"

mode2paths = {
    "standard": {
        "predictions_path": "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/predictions/final_RoP_Cov_Single__18c3cff/",
        "data_path": "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/data/prerendered/validation/",
        "savefolder": "../evals/",
    },
    "simplified_rg": {
        "predictions_path": "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/predictions_simplified_rg/final_RoP_Cov_Single__18c3cff/",
        "data_path": "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/data/prerendered/validation_simplified_rg/",
        "savefolder": "../evals_simplified_rg/",
    },
    "simplified_rg_no_others": {
        "predictions_path": "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/predictions_simplified_rg_no_others/final_RoP_Cov_Single__18c3cff/",
        "data_path": "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/data/prerendered/validation_simplified_rg_no_others/",
        "savefolder": "../evals_simplified_rg_no_others/",
    },
    "carla": {
        "predictions_path": "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/predictions_carla/final_RoP_Cov_Single__18c3cff/",
        "data_path": "/home/manolotis/sandbox/c4r/data/last/",
        "savefolder": "../evals_carla/",
    }
}

predictions_path = mode2paths[MODE]["predictions_path"]
data_path = mode2paths[MODE]["data_path"]
savefolder = mode2paths[MODE]["savefolder"]

MAX = 100000000000000


def new_scenario_or_agent_id(last_sid, last_aid, current_sid, current_aid):
    if last_sid is None or last_aid is None:
        return False

    if last_sid != current_sid or last_aid != current_aid:
        return True

    return False


def new_scenario(last_sid, current_sid):
    if last_sid is None:
        return False

    if last_sid != current_sid:
        return True

    return False


def new_agent(last_aid, current_aid):
    if last_aid is None:
        return False

    if last_aid != current_aid:
        return True

    return False


def get_trajectory_matching(pred1, pred2):
    # Out of 2 sets of predictions with K trajectories each (for MultiPath++ it's 6), get their matching using the
    # Hungarian algorithm.
    # E.g. pred1 are the predictions made at time t, and pred2 at time t+1. We match the predicted trajectories from the
    # predictions made at 2 subsequent timesteps.
    pass


def get_ade_between_trajectories(traj1, traj2):
    diff = traj2 - traj1
    euclidian_dist = np.linalg.norm(diff, axis=1)
    ade = np.mean(euclidian_dist)

    return ade


def get_fde_between_trajectories(traj1, traj2):
    p1 = traj1[-1:]
    p2 = traj2[-1:]

    diff = p2 - p1
    euclidian_dist = np.linalg.norm(diff, axis=1)
    fde = np.mean(euclidian_dist)

    return fde


def get_ade_matrix(coords1, coords2):
    # given trajectory predictions of shape (n, 80, 2), return a n x n matrix where each entry is the ADE between the
    # trajectories
    n, t, f = coords1.shape
    m = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            traj1 = coords1[i]
            traj2 = coords2[j]
            # ToDo: Get only the ade between overlapping timesteps
            ade = get_ade_between_trajectories(traj1, traj2)
            m[i, j] = ade

    return m


def get_fde_matrix(coords1, coords2):
    # given trajectory predictions of shape (n, 80, 2), return a n x n matrix where each entry is the ADE between the
    # trajectories
    n, t, f = coords1.shape
    m = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            traj1 = coords1[i]
            traj2 = coords2[j]
            # ToDo: Get only the fde between overlapping timesteps
            fde = get_fde_between_trajectories(traj1, traj2)
            m[i, j] = fde

    return m


def run():
    print("Getting grouped predictions...")
    grouped_predictions, scene_agent_type_map = load_predictions_and_group(predictions_path, MAX)

    for scene_id, scene_predictions in grouped_predictions.items():
        for agent_id, agent_predictions in scene_predictions.items():
            # agent_predictions should be a dictionary of (timestep, prediction(s)) pairs

            timesteps = list(agent_predictions.keys())
            min_timestep, max_timestep = min(timesteps), max(timesteps)

            rank_switches_counter = 0
            top1_rank_switches = 0  # ToDO: make TopK
            top1_rank_switches_ade_sum = 0  # ToDO: make TopK
            top1_rank_switches_ades = []  # ToDO: make TopK
            top1_rank_switches_counter = 0  # counts the number of times a rank switch occurs, instead of how many ranks are switched. Used for average ADE computation

            for t in range(min_timestep + 1, max_timestep + 1):
                # compute matching between predictions at subsequent steps
                t_prev = t - 1
                t_curr = t

                try:
                    pred_prev = agent_predictions[t_prev]
                    coords_prev = agent_predictions[t_prev]['coordinates']  # shape (6, 80, 2)
                    probs_prev = agent_predictions[t_prev]['probabilities']  # shape (6,)
                    pred_curr = agent_predictions[t_curr]
                except KeyError:
                    print("[!] Key error", t_curr, scene_id, agent_id)
                    continue
                coords_curr = agent_predictions[t_curr]['coordinates']  # shape (6, 80, 2)
                probs_curr = agent_predictions[t_curr]['probabilities']  # shape (6,)

                # except KeyError:
                #     print("Keyerror: ", scene_id, agent_id, t, "continuing")
                #     continue

                ade_matrix = get_ade_matrix(coords_prev, coords_curr)
                fde_matrix = get_fde_matrix(coords_prev, coords_curr)

                row_ind, col_ind = linear_sum_assignment(ade_matrix)

                # print(ade_matrix)
                # print("col_ind from ADE", col_ind)

                # ToDo: make topK
                # Important: this assumes predictions are order from most to least likely
                if col_ind[0] != 0:
                    # This means the new trajectory matched to the old one is no longer the most likely.
                    # Increase rank switch count and sum ADE
                    previous_trajectory = coords_prev[0]
                    new_trajectory = coords_curr[col_ind[0]]
                    top1_rank_switches_counter += 1  # there was a rank switch
                    top1_rank_switches += col_ind[0]  # we sum how many ranks it moved
                    ade = get_ade_between_trajectories(previous_trajectory, new_trajectory)
                    top1_rank_switches_ade_sum += ade
                    top1_rank_switches_ades.append(ade)

                    # print("timestep: ", t, "col_ind", col_ind)

                if np.any(col_ind != [0, 1, 2, 3, 4, 5]):
                    rank_switches_counter += 1
                    # print("timestep: ", t, "col_ind", col_ind)

                # print("row_ind from ADE", row_ind)
                # row_ind, col_ind = linear_sum_assignment(fde_matrix)
                # print("col_ind from FDE", col_ind)
                # print("row_ind from FDE", row_ind)

                # exit()

                # print(coords_curr.shape)
                # print(probs_curr.shape)
                # print(probs_curr)

            # print(min_timestep, max_timestep, agent_predictions)
            # print("Rank switches counter", rank_switches_counter)
            # print("Top1 Rank switches", top1_rank_switches)
            # print("Top1 Rank switches counter", top1_rank_switches_counter)
            #
            # print("All ADEs of top1 rank switches", top1_rank_switches_ades)
            # print("Avg ADE of top1 rank switches", top1_rank_switches_ade_sum / top1_rank_switches_counter)

            # toDo: make K
            # if top1_rank_switches_counter == 0:
            #     print("skipping")
            #     continue

            eval = {
                'rank_switches_counter': rank_switches_counter,
                'top1_rank_switches': top1_rank_switches,
                'top1_rank_switches_ade_sum': top1_rank_switches_ade_sum,
                'top1_rank_switches_ades': top1_rank_switches_ades,
                'top1_rank_switches_ade_mean': np.mean(top1_rank_switches_ades),
                'top1_rank_switches_counter': top1_rank_switches_counter,
            }

            agent_type = scene_agent_type_map[(scene_id, agent_id)]
            filename = get_grouped_scene_filename(scene_id, agent_id, agent_type, min_timestep, max_timestep)

            savepath = os.path.join(savefolder, filename)
            # print(savepath)

            np.savez_compressed(savepath, **eval)
            # print("saved to ")
            #
            # exit()
            # pass


if __name__ == "__main__":
    run()
