"""
Given a tensor of trajectories, each of these functions will generate a
tensor of matching shape with one statistic (scalar value) for each of
the trajectories.
"""

import numpy as np
import h5py
from scipy.spatial.distance import jensenshannon as JSD
from . import stop_points as sp

def min_max_distance_traveled(M1, M2, G, tol=1e-6, mode="min"):
    net_distance = {}

    for name, tensor in zip(("M1", "M2", "G"), (M1, M2, G)):
        net_distance[name] = []

        for agent in tensor:
            agent_distance = []
            # agent shape: (..., T, D)

            # step-wise distances
            transitions = np.linalg.norm(
                np.diff(agent, axis=-2), axis=-1
            )  # shape: (..., T-1)

            # ignore numerical noise
            transitions = np.where(transitions > tol, transitions, 0.0)

            # total distance traveled per trajectory
            total_distance = transitions.sum(axis=-1)  # shape: (...)

            if mode == "min":
                agent_distance.append(np.nanmin(total_distance, axis=-1))
            elif mode == "max":
                agent_distance.append(np.nanmax(total_distance, axis=-1))
            else:
                raise ValueError("mode must be 'min' or 'max'")

            net_distance[name].append(np.array(agent_distance))

    return net_distance


def distance_traveled(M1, M2, G):
    net_distance = {}

    for name, tensor in zip(("M1", "M2", "G"), (M1, M2, G)):
        net_distance[name] = []
        for agent in tensor:
            distances = np.diff(agent, axis=1)
            net_distance[name].append(np.linalg.norm(distances, axis=-1).sum(axis=1))

    return net_distance


def radius_of_gyration(M1, M2, G):
    net_rg = {}

    for name, tensor in zip(("M1", "M2", "G"), (M1, M2, G)):
        net_rg[name] = []
        for agent in tensor:
            agent_stats = []
            mean_pos = agent.mean(axis=1, keepdims=True)
            sq_disp = np.sum((agent - mean_pos) ** 2, axis=-1)
            rg = np.sqrt(sq_disp.mean(axis=1))
            net_rg[name].append(rg)

    return net_rg


def duration_of_movement(M1, M2, G):
    time_moving = {}

    for name, tensor in zip(("M1", "M2", "G"), (M1, M2, G)):
        time_moving[name] = []
        for agent in tensor:
            deltas = np.diff(agent, axis=1)
            dists = np.linalg.norm(deltas, axis=-1)
            moving = (dists > 0).sum(axis=1)
            time_moving[name].append(moving)

    return time_moving

def unique_stop_points(M1, M2, G):
    M1 = sp.detect_stops(M1)
    M2 = sp.detect_stops(M2)
    G = sp.detect_stops(G)

    unique_stops = {}

    for group, data in (("M1", M1), ("M2", M2), ("G", G)):
        unique_stops[group] = []
        for trains, tests in zip(M1, data):
            train = np.vstack(trains)

            agent = []
            for test in tests:
                agent.append(sp.unique_stops(train, test, 1e-4).shape[0])
            unique_stops[group].append(np.array(agent))

    return unique_stops
