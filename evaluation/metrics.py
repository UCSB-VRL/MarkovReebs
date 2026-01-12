import warnings
import numpy as np
from scipy.spatial.distance import jensenshannon


def compute_hist(value_matrix, minimum=0., maximum=100., bins=100):
    return np.histogram(value_matrix.flatten(), bins=bins, range=(minimum, maximum))


def mean_by_agent(statistics):
    # statistics is a dict of tensors
    # each statistic is a list of agents each an array of statistics
    mean_statistics = {}
    for group, data in statistics.items():
        mean_statistics[group] = []
        for agent in data:
            if len(agent.shape) < 1:
                # some agents may be empty
                continue
            mean_statistics[group].append(np.mean(agent, axis=0))
        mean_statistics[group] = np.array(mean_statistics[group])

    return mean_statistics

def mean_by_trajectory(statistics):
    mean_statistics = {}
    for group, data in statistics.items():
        mean_statistics[group] = []
        for agent in data:
            if len(agent.shape) < 1:
                # some agents may be empty
                continue
            mean_statistics[group].append(agent.flatten())
        mean_statistics[group] = np.concatenate(mean_statistics[group]).flatten()

    return mean_statistics

def JSD(statistics, baseline="M1"):
    minimum = min(np.nanmin(matrix) for matrix in statistics.values())
    maximum = max(np.nanmax(matrix) for matrix in statistics.values())

    M1 = statistics.pop(baseline, None)
    baseline_hist = compute_hist(M1, minimum, maximum)

    results = {}
    for group, statistic in statistics.items():
        statistic = statistic[~np.isnan(statistic) & ~np.isinf(statistic)]
        group_hist = compute_hist(statistic, minimum, maximum)
        results[group] = jensenshannon(baseline_hist[0], group_hist[0])
    return results

def MAPE(statistics, baseline="M1"):
    M1 = statistics.pop(baseline, None)

    results = {}
    for group, statistic in statistics.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"divide by zero encountered in divide")
            warnings.filterwarnings("ignore", r"invalid value encountered in divide")
            APE = np.ma.masked_invalid(np.abs((statistic - M1) / M1))
        results[group] = np.mean(APE)

    return results


def MAE(statistics, baseline="M1"):
    M1 = statistics.pop(baseline, None)

    results = {}
    for group, statistic in statistics.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"divide by zero encountered in divide")
            warnings.filterwarnings("ignore", r"invalid value encountered in divide")
            MAE = np.ma.masked_invalid(np.abs(statistic - M1))
        results[group] = np.mean(MAE)

    return results
