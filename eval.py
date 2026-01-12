"""
Evaluates trajectories based on a series of metrics and prints out a table of
results.
"""

import h5py
import numpy as np
from pprint import pprint

import evaluation.trajectory_stats as stats
import evaluation.metrics as metrics
import evaluation.table as table


def trim(M1, M2, G, start=0, end=-1):
    def forward_fill_inf(x, axis=1):
        x = np.array(x, copy=True)
        mask = np.isinf(x).any(axis=-1)

        # Forward fill along axis
        idx = np.where(~mask, np.arange(x.shape[axis]), 0)
        idx = np.maximum.accumulate(idx, axis=axis)
        x = np.take_along_axis(x, idx[..., None], axis=axis)

        # Back-fill leading infs
        first_valid_idx = (~mask).argmax(axis=axis)[:, None, None]
        first_valid_vals = np.take_along_axis(x, first_valid_idx, axis=axis)

        pos = np.arange(x.shape[axis])[None, :, None]
        x = np.where(pos < first_valid_idx, first_valid_vals, x)

        return x

    trimmed = []
    for group, data in (("M1", M1), ("M2", M2), ("G", G)):
        trimmed_agents = [forward_fill_inf(agent)[:, start:end, :] for agent in data]
        trimmed.append(trimmed_agents)

    return tuple(trimmed)


def load(path):
    M1, M2, G = [], [], []
    with h5py.File(path, "r") as f:
        for key, target in (("M1", M1), ("M2", M2), ("G", G)):
            matches = [match for match in f.keys() if match.startswith(f"{key}_")]
            target += [f[match][()] for match in matches]

    return M1, M2, G

def compute_scores(statistics, mode="agent", comparison="MAE"):
    if mode == "agent":
        stats_by_agent = metrics.mean_by_agent(statistics)
        if comparison == "MAPE":
            scores = metrics.MAPE(stats_by_agent)
        elif comparison == "MAE":
            scores = metrics.MAE(stats_by_agent)
        return scores
    elif mode == "population":
        stats_by_trajectory = metrics.mean_by_trajectory(statistics)
        scores = metrics.JSD(stats_by_trajectory)
        return scores


def eval():
    datasets = [
        ("ua", None), 
        ("geolife", (4500, 7000))
    ]
    modes = ["agent", "population"]
    methods = ["baseline"]
    statistics = [
        ("Distance Traveled", stats.distance_traveled),
        ("Shortest Trip Distance", lambda m1, m2, g:
            stats.min_max_distance_traveled(m1, m2, g, mode="min")),
        ("Longest Trip Distance", lambda m1, m2, g:
            stats.min_max_distance_traveled(m1, m2, g, mode="max")),
        ("Radius of Gyration", stats.radius_of_gyration),
        ("Duration of Movement", stats.duration_of_movement),
        ("New Locations Visited", stats.unique_stop_points)
    ]

    output_table = {}
    for dataset, bounds in datasets:
        print("Working on", dataset)
        dataset_table = []
        for method in methods:
            print("Method", method)
            M1, M2, G = load(f"outputs/{method}_{dataset}.h5")
            if bounds:
                print("Trimming and refitting")
                M1, M2, G = trim(M1, M2, G, start=bounds[0], end=bounds[1])

            for mode in modes:
                row_dict = {
                    "M2": {"Evaluation Method": mode, "Generation Method": method},
                    "G": {"Evaluation Method": mode, "Generation Method": method},
                }

                for stat_name, stat_function in statistics:
                    stat = stat_function(M1, M2, G)
                    scores = compute_scores(stat, mode=mode)
                    for source, score in scores.items():
                        row_dict[source][stat_name] = score

                # flatten into a table
                for source, row in row_dict.items():
                    row["Source"] = source
                    dataset_table.append(row)

        output_table[dataset] = dataset_table
    return output_table

def ablation():
    datasets = [
        ("ua", None), 
        ("geolife", (4500, 7000))
    ]
    resolutions = [
        "1e-1", "1e-2", "1e-3", "1e-4", "1e-6"
    ]
    modes = ["agent", "population"]
    methods = ["hrg"]
    statistics = [
        ("Distance Traveled", stats.distance_traveled),
        ("Shortest Trip Distance", lambda m1, m2, g:
            stats.min_max_distance_traveled(m1, m2, g, mode="min")),
        ("Longest Trip Distance", lambda m1, m2, g:
            stats.min_max_distance_traveled(m1, m2, g, mode="max")),
        ("Radius of Gyration", stats.radius_of_gyration),
        ("Duration of Movement", stats.duration_of_movement),
        ("New Locations Visited", stats.unique_stop_points)
    ]

    output_table = {}
    for dataset, bounds in datasets:
        print("Working on", dataset)
        dataset_table = []
        for method in methods:
            print("Method", method)
            for resolution in resolutions:
                print("Resolution", resolution)
                M1, M2, G = load(f"outputs/{method}_{dataset}_epsilon={resolution}_alpha=1e-1.h5")
                if bounds:
                    print("Trimming and refitting")
                    M1, M2, G = trim(M1, M2, G, start=bounds[0], end=bounds[1])

                for mode in modes:
                    row_dict = {
                        "M2": {"Evaluation Method": mode, "Generation Method":
                               method, "Epsilon": resolution},
                        "G": {"Evaluation Method": mode, "Generation Method":
                               method, "Epsilon": resolution},
                    }

                    for stat_name, stat_function in statistics:
                        stat = stat_function(M1, M2, G)
                        scores = compute_scores(stat, mode=mode)
                        for source, score in scores.items():
                            row_dict[source][stat_name] = score

                    # flatten into a table
                    for source, row in row_dict.items():
                        row["Source"] = source
                        dataset_table.append(row)

        output_table[dataset] = dataset_table
    return output_table

if __name__ == "__main__":
    from pprint import pprint
    output_table = ablation()
    pprint(output_table)

