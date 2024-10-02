from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt

from .utils import (
    filter_valid, get_filter_valid_roadnetwork_keys, get_filter_valid_anget_history,
    get_normalize_data)


class Renderer(ABC):
    @abstractmethod
    def render(self, data):
        pass


class SegmentFilteringPolicy:
    def __init__(self, config):
        self._config = config

    def _select_n_closest_segments(self, segments, types):
        distances = np.linalg.norm(segments, axis=-1).min(axis=-1)
        n_closest_segments_ids = np.argpartition(
            distances, self._config["n_closest_segments"])[:self._config["n_closest_segments"]]
        return segments[n_closest_segments_ids], types[n_closest_segments_ids].flatten()

    def _select_segments_within_radius(self, segments, types):
        distances = np.linalg.norm(segments, axis=-1).min(axis=-1)
        closest_segments_selector = distances < self._config["segments_filtering_radius"]
        return segments[closest_segments_selector], types[closest_segments_selector].flatten()

    def filter(self, segments, types):
        if self._config["policy"] == "n_closest_segments":
            return self._select_n_closest_segments(segments, types)
        if self._config["policy"] == "within_radius":
            return self._select_segments_within_radius(segments, types)
        raise Exception(f"Unknown segment filtering policy {self._config['policy']}")


class TargetAgentFilteringPolicy:
    def __init__(self, config):
        self._config = config

    def _get_only_interesting_agents(self, data, i):
        return data["state/tracks_to_predict"][i] > 0

    def _get_only_fully_available_agents(self, data, i):
        full_validity = np.concatenate([
            data["state/past/valid"], data["state/current/valid"], data["state/future/valid"]],
            axis=-1)
        n_timestamps = full_validity.shape[-1]
        n_valid_timestamps = full_validity.sum(axis=-1)
        return n_valid_timestamps[i] == n_timestamps
        # return np.ones_like(n_valid_timestamps) * (n_valid_timestamps == n_timestamps)

    def _get_interesting_and_fully_available_agents(self, data, i):
        interesting = self._get_only_interesting_agents(data, i)
        fully_valid = self._get_only_fully_available_agents(data, i)
        return interesting and fully_valid

    def _get_fully_available_agents_without_interesting(self, data, i):
        interesting = self._get_only_interesting_agents(data, i)
        fully_valid = self._get_only_fully_available_agents(data, i)
        return fully_valid and not interesting

    def allow(self, data, i):
        if self._config["policy"] == "interesting":
            return self._get_only_interesting_agents(data, i)
        if self._config["policy"] == "fully_available":
            return self._get_only_fully_available_agents(data, i)
        if self._config["policy"] == "interesting_and_fully_available":
            return self._get_interesting_and_fully_available_agents(data, i)
        if self._config["policy"] == "fully_available_agents_without_interesting":
            return self._get_fully_available_agents_without_interesting(data, i)
        raise Exception(f"Unknown agent filtering policy {self._config['policy']}")


class MultiPathPPRenderer(Renderer):
    def __init__(self, config):
        self._config = config
        self.n_segment_types = 20
        self._segment_filter = SegmentFilteringPolicy(self._config["segment_filtering"])
        self._target_agent_filter = TargetAgentFilteringPolicy(self._config["agent_filtering"])

    def _select_agents_with_any_validity(self, data):
        return data["state/current/valid"].sum(axis=-1) + \
            data["state/future/valid"].sum(axis=-1) + data["state/past/valid"].sum(axis=-1)

    def _preprocess_data(self, data):

        # node_type = data["roadgraph_samples/type"]
        #
        # # Keep only lane centers and broken/solid single white lines
        # # values_to_keep = [1, 2, 3, 6, 7]
        #
        # # Keep only lane centers
        # values_to_keep = [1, 2, 3]
        #
        # nodes_to_keep = node_type == 1
        # for v in values_to_keep[1:]:
        #     nodes_to_keep = nodes_to_keep | (node_type == v)
        #
        # nodes_to_keep = nodes_to_keep.flatten()
        # data["roadgraph_samples/valid"][~nodes_to_keep] = 0

        valid_roadnetwork_selector = data["roadgraph_samples/valid"]
        for key in get_filter_valid_roadnetwork_keys():
            data[key] = filter_valid(data[key], valid_roadnetwork_selector)
        agents_with_any_validity_selector = self._select_agents_with_any_validity(data)
        for key in get_filter_valid_anget_history():
            data[key] = filter_valid(data[key], agents_with_any_validity_selector)

    def _prepare_roadnetwork_info(self, data):
        # Returns np.array of shape [N, 2, 2]
        # 0 dim: N - number of segments
        # 1 dim: the start and the end of a segment
        # 2 dim: (x, y)
        # and
        # ndarray of segment types
        node_xyz = data["roadgraph_samples/xyz"][:, :2]
        node_id = data["roadgraph_samples/id"].flatten()
        node_type = data["roadgraph_samples/type"]
        result = []
        segment_types = []
        for polyline_id in np.unique(node_id):
            polyline_nodes = node_xyz[node_id == polyline_id]
            polyline_type = node_type[node_id == polyline_id][0]
            if len(polyline_nodes) == 1:
                polyline_nodes = np.array([polyline_nodes[0], polyline_nodes[0]])
            if "drop_segments" in self._config:
                selector = np.arange(len(polyline_nodes), step=self._config["drop_segments"])
                if len(polyline_nodes) <= self._config["drop_segments"]:
                    selector = np.array([0, len(polyline_nodes) - 1])
                selector[-1] = len(polyline_nodes) - 1
                polyline_nodes = polyline_nodes[selector]
            polyline_start_end = np.array(
                [polyline_nodes[:-1], polyline_nodes[1:]]).transpose(1, 0, 2)
            result.append(polyline_start_end)

            segment_types.extend([polyline_type] * len(polyline_start_end))
        result = np.concatenate(result, axis=0)
        assert len(segment_types) == len(result), \
            f"Number of segments {len(result)} doen't match the number of types {len(segment_types)}"
        return {
            "segments": result,
            "segment_types": np.array(segment_types)}

    def _split_past_and_future(self, data, key, timestep):

        all_data = np.concatenate(
            [data[f"state/past/{key}"], data[f"state/current/{key}"], data[f"state/future/{key}"]], axis=1)[..., None]

        history = all_data[:, timestep - 11:timestep, :]

        # future must be 80 timesteps. Make sure it is.
        n_agents, n_steps, n_features = all_data[:, timestep:, :].shape
        filler_data = np.full((n_agents, 80, n_features), -1)
        future = filler_data
        future[:, :n_steps, :] = all_data[:, timestep:, :]

        #
        # history = np.concatenate(
        #     [data[f"state/past/{key}"], data[f"state/current/{key}"]], axis=1)[..., None]
        # future = data[f"state/future/{key}"][..., None]
        return history, future

    def _prepare_agent_history(self, data, timestep):
        # (n_agents, 11, 2)
        preprocessed_data = {}

        # print("shape before", all_xy.shape)
        # print("type before", type(all_xy))

        # preprocessed_data["history/xy"] = np.array([
        #     np.concatenate([data["state/past/x"], data["state/current/x"]], axis=1),
        #     np.concatenate([data["state/past/y"], data["state/current/y"]], axis=1)
        # ]).transpose(1, 2, 0)
        #
        # # (n_agents, 80, 2)
        # preprocessed_data["future/xy"] = np.array(
        #     [data["state/future/x"], data["state/future/y"]]).transpose(1, 2, 0)
        # # (n_agents, 11, 1)

        # print(data["state/past/x"].shape, data["state/current/x"].shape, data["state/future/x"].shape)
        # print()

        all_xy = np.array([
            np.concatenate([data["state/past/x"], data["state/current/x"], data["state/future/x"]], axis=1),
            np.concatenate([data["state/past/y"], data["state/current/y"], data["state/future/y"]], axis=1),
        ]).transpose(1, 2, 0)

        if timestep < 11:
            raise ValueError
        preprocessed_data["history/xy"] = all_xy[:, timestep - 11:timestep, :]
        preprocessed_data["future/xy"] = all_xy[:, timestep:, :]

        # future must be 80 timesteps. Make sure it is.
        n_agents, n_steps, n_features = all_xy[:, timestep:, :].shape
        filler_xy = np.full((n_agents, 80, n_features), -1.0)
        preprocessed_data["future/xy"] = filler_xy
        preprocessed_data["future/xy"][:, :n_steps, :] = all_xy[:, timestep:, :]

        #
        # # print("shape after", preprocessed_data["future/xy"].shape)
        # # print("type after", type(preprocessed_data["history/xy"]))

        for key in ["speed", "bbox_yaw", "valid"]:
            preprocessed_data[f"history/{key}"], preprocessed_data[f"future/{key}"] = \
                self._split_past_and_future(data, key, timestep)

        for key in ["state/id", "state/is_sdc", "state/type", "state/current/width",
                    "state/current/length"]:
            preprocessed_data[key.split('/')[-1]] = data[key]
        preprocessed_data["scenario_id"] = data["scenario/id"]

        return preprocessed_data

    def _transfrom_to_agent_coordinate_system(self, coordinates, shift, yaw):
        # coordinates
        # dim 0: number of agents / number of segments for road network
        # dim 1: number of history points / (start_point, end_point) for segments
        # dim 2: x, y
        yaw = -yaw
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array(((c, -s), (s, c))).reshape(2, 2)
        transformed = np.matmul((coordinates - shift), R.T)
        return transformed

    def _filter_closest_segments(self, segments, types):
        # This method works only with road segments in agent-related coordinate system
        assert len(segments.shape) == 3
        assert segments.shape[1] == segments.shape[2] == 2
        assert len(segments) == len(types), \
            f"n_segments={len(segments)} must match len_types={len(types)}"
        return self._segment_filter.filter(segments, types)

    def _compute_closest_point_of_segment(self, segments):
        # This method works only with road segments in agent-related coordinate system
        assert len(segments.shape) == 3
        assert segments.shape[1] == segments.shape[2] == 2
        A, B = segments[:, 0, :], segments[:, 1, :]
        M = B - A
        t = (-A * M).sum(axis=-1) / ((M * M).sum(axis=-1) + 1e-6)
        clipped_t = np.clip(t, 0, 1)[:, None]
        closest_points = A + clipped_t * M
        return closest_points

    def _generate_segment_embeddings(self, segments, types):
        # This method works only with road segments in agent-related coordinate system
        # previously filtered
        closest_points = self._compute_closest_point_of_segment(segments)
        r_norm = np.linalg.norm(closest_points, axis=-1, keepdims=True)
        r_unit_vector = closest_points / (r_norm + 1e-6)
        segment_end_minus_start = segments[:, 1, :] - segments[:, 0, :]
        segment_end_minus_start_norm = np.linalg.norm(
            segment_end_minus_start, axis=-1, keepdims=True)
        segment_unit_vector = segment_end_minus_start / (segment_end_minus_start_norm + 1e-6)
        segment_end_minus_r_norm = np.linalg.norm(
            segments[:, 1, :] - closest_points, axis=-1, keepdims=True)
        segment_type_ohe = np.eye(self.n_segment_types)[types]
        resulting_embeddings = np.concatenate([
            r_norm, r_unit_vector, segment_unit_vector, segment_end_minus_start_norm,
            segment_end_minus_r_norm, segment_type_ohe], axis=-1)
        return resulting_embeddings[:, None, :]

    def _normalize_tensor(self, tensor, mean, std):
        if not self._config["normalize"]:
            return tensor
        raise Exception("Normalizing here is really not what you want. Please use normalization from model.data")
        return (tensor - mean) / (std + 1e-6)

    def _normalize(self, tensor, i, key):
        target_data = tensor[i][None,]
        other_data = np.delete(tensor, i, axis=0)
        if not self._config["normalize"]:
            return target_data, other_data
        target_data = self._normalize_tensor(target_data, **get_normalize_data()["target"][key])
        other_data = self._normalize_tensor(other_data, **get_normalize_data()["other"][key])
        return target_data, other_data

    def _get_trajectory_class(self, data):
        valid = np.concatenate(
            [data["target/history/valid"][0, -1:, 0], data["target/future/valid"][0, :, 0]])
        future_xy = np.concatenate(
            [data["target/history/xy"][0, -1:, :], data["target/future/xy"][0, :, :]])
        future_yaw = np.concatenate(
            [data["target/history/yaw"][0, -1:, 0], data["target/future/yaw"][0, :, 0]])
        future_speed = np.concatenate(
            [data["target/history/speed"][0, -1:, 0], data["target/future/speed"][0, :, 0]])

        kMaxSpeedForStationary = 2.0  # (m/s)
        kMaxDisplacementForStationary = 5.0  # (m)
        kMaxLateralDisplacementForStraight = 5.0  # (m)
        kMinLongitudinalDisplacementForUTurn = -5.0  # (m)
        kMaxAbsHeadingDiffForStraight = np.pi / 6.0  # (rad)
        first_valid_index, last_valid_index = 0, None
        for i in range(1, len(valid)):
            if valid[i] == 1:
                last_valid_index = i
        if valid[first_valid_index] == 0 or last_valid_index is None:
            return None

        xy_delta = future_xy[last_valid_index] - future_xy[first_valid_index]
        final_displacement = np.linalg.norm(xy_delta)
        heading_delta = future_yaw[last_valid_index] - future_yaw[first_valid_index]
        max_speed = max(future_speed[last_valid_index], future_speed[first_valid_index])

        if max_speed < kMaxSpeedForStationary and \
                final_displacement < kMaxDisplacementForStationary:
            return "stationary"
        if np.abs(heading_delta) < kMaxAbsHeadingDiffForStraight:
            if np.abs(xy_delta[1]) < kMaxLateralDisplacementForStraight:
                return "straight"
            return "straing_right" if xy_delta[1] < 0 else "straight_left"
        if heading_delta < -kMaxAbsHeadingDiffForStraight and xy_delta[1]:
            return "right_u_turn" if xy_delta[0] < kMinLongitudinalDisplacementForUTurn \
                else "right_turn"
        if xy_delta[0] < kMinLongitudinalDisplacementForUTurn:
            return "left_u_turn"
        return "left_turn"

    def render(self, data):
        array_of_scene_data_dicts = []
        self._preprocess_data(data)

        try:
            road_network_info = self._prepare_roadnetwork_info(data)
        except ValueError:
            return None

        num_steps = 80
        current_step = 11
        for timestep in range(current_step, current_step + num_steps):

            agent_history_info = self._prepare_agent_history(data, timestep)

            for i in range(agent_history_info["history/xy"].shape[0]):
                if not self._target_agent_filter.allow(data, i):
                    continue
                current_agent_scene_shift = agent_history_info["history/xy"][i][-1]

                if self._config["noisy_heading"]:
                    agent_history_info["history/bbox_yaw"][i][-1] += np.pi / 2

                current_agent_scene_yaw = np.copy(agent_history_info["history/bbox_yaw"][i][-1])
                current_scene_road_network_coordinates = self._transfrom_to_agent_coordinate_system(
                    road_network_info["segments"], current_agent_scene_shift, current_agent_scene_yaw)
                current_scene_road_network_coordinates, current_scene_road_network_types = \
                    self._filter_closest_segments(
                        current_scene_road_network_coordinates, road_network_info["segment_types"])
                current_scene_road_network_coordinates = self._normalize_tensor(
                    current_scene_road_network_coordinates,
                    **get_normalize_data()["road_network_segments"])
                road_segments_embeddings = self._generate_segment_embeddings(
                    current_scene_road_network_coordinates, current_scene_road_network_types)
                current_scene_agents_coordinates_history = self._transfrom_to_agent_coordinate_system(
                    agent_history_info["history/xy"], current_agent_scene_shift,
                    current_agent_scene_yaw)
                current_scene_agents_coordinates_future = self._transfrom_to_agent_coordinate_system(
                    agent_history_info["future/xy"], current_agent_scene_shift, current_agent_scene_yaw)
                current_scene_agents_yaws_history = \
                    agent_history_info["history/bbox_yaw"] - current_agent_scene_yaw
                current_scene_agents_yaws_future = \
                    agent_history_info["future/bbox_yaw"] - current_agent_scene_yaw
                (current_scene_target_agent_coordinates_history,
                 current_scene_other_agents_coordinates_history) = self._normalize(
                    current_scene_agents_coordinates_history, i, "xy")
                current_scene_target_agent_yaws_history, current_scene_other_agents_yaws_history = \
                    self._normalize(current_scene_agents_yaws_history, i, "yaw")
                current_scene_target_agent_speed_history, current_scene_other_agents_speed_history = \
                    self._normalize(agent_history_info["history/speed"], i, "speed")

                other_agent_type = np.delete(agent_history_info["type"], i, axis=0).astype(int)
                other_is_sdc = np.delete(agent_history_info["is_sdc"], i, axis=0).astype(int)
                other_width = np.delete(agent_history_info["width"], i)
                other_length = np.delete(agent_history_info["length"], i)
                other_future_xy = np.delete(current_scene_agents_coordinates_future, i, axis=0)
                other_future_yaw = np.delete(current_scene_agents_yaws_future, i, axis=0)
                other_future_speed = np.delete(agent_history_info["future/speed"], i, axis=0)
                other_future_valid = np.delete(agent_history_info["future/valid"], i, axis=0)
                other_history_xy = current_scene_other_agents_coordinates_history
                other_history_yaw = current_scene_other_agents_yaws_history
                other_history_speed = current_scene_other_agents_speed_history
                other_history_valid = np.delete(agent_history_info["history/valid"], i, axis=0)

                ###################
                # To remove information of others
                # print("other_agent_type.shape", other_agent_type.shape)
                # print("other_is_sdc.shape", other_is_sdc.shape)
                # print("other_width.shape", other_width.shape)
                # print("other_length.shape", other_length.shape)
                # print("other_future_xy.shape", other_future_xy.shape)
                # print("other_future_yaw.shape", other_future_yaw.shape)
                # print("other_future_speed.shape", other_future_speed.shape)
                # print("other_future_valid.shape", other_future_valid.shape)
                # print("other_history_xy.shape", other_history_xy.shape)
                # print("other_history_yaw.shape", other_history_yaw.shape)
                # print("other_history_speed.shape", other_history_speed.shape)
                # print("other_history_valid.shape", other_history_valid.shape)

                # other_agent_type[:] = -1
                # other_is_sdc[:] = -1
                # other_width[:] = -1
                # other_length[:] = -1
                # other_future_xy[:, :, :] = -1
                # other_future_yaw[:, :, :] = -1
                # other_future_speed[:, :, :] = -1
                # other_future_valid[:, :, :] = 0
                # other_history_xy[:, :, :] = -1
                # other_history_yaw[:, :, :] = -1
                # other_history_speed[:, :, :] = -1
                # other_history_valid[:, :, :] = 0

                # exit()

                ###################



                scene_data = {
                    "shift": current_agent_scene_shift[None,],
                    "yaw": current_agent_scene_yaw,
                    "scenario_id": agent_history_info["scenario_id"].item().decode("utf-8"),
                    "agent_id": int(agent_history_info["id"][i]),
                    "target/agent_type": np.array([int(agent_history_info["type"][i])]).reshape(1),

                    "target/is_sdc": np.array(int(agent_history_info["is_sdc"][i])).reshape(1),
                    "target/width": agent_history_info["width"][i].item(),
                    "target/length": agent_history_info["length"][i].item(),

                    "target/future/xy": current_scene_agents_coordinates_future[i][None,],
                    "target/future/yaw": current_scene_agents_yaws_future[i][None,],
                    "target/future/speed": agent_history_info["future/speed"][i][None,],
                    "target/future/valid": agent_history_info["future/valid"][i][None,],
                    "target/history/xy": current_scene_target_agent_coordinates_history,
                    "target/history/yaw": current_scene_target_agent_yaws_history,
                    "target/history/speed": current_scene_target_agent_speed_history,
                    "target/history/valid": agent_history_info["history/valid"][i][None,],

                    "other/agent_type": other_agent_type,
                    "other/is_sdc": other_is_sdc,
                    "other/width": other_width,
                    "other/length": other_length,
                    "other/future/xy": other_future_xy,
                    "other/future/yaw": other_future_yaw,
                    "other/future/speed": other_future_speed,
                    "other/future/valid": other_future_valid,
                    "other/history/xy": other_history_xy,
                    "other/history/yaw": other_history_yaw,
                    "other/history/speed": other_history_speed,
                    "other/history/valid": other_history_valid,

                    "road_network_embeddings": road_segments_embeddings,
                    "road_network_segments": current_scene_road_network_coordinates,
                    "road_network_segments_types": current_scene_road_network_types,
                    "timestep": timestep

                }

                scene_data["trajectory_bucket"] = self._get_trajectory_class(scene_data)
                if self._config["noisy_heading"]:
                    scene_data["yaw_original"] = current_agent_scene_yaw - np.pi / 2
                array_of_scene_data_dicts.append(scene_data)
        return array_of_scene_data_dicts
