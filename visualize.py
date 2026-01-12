"""
Generates all figures for plotting
"""

# %% Parameters
files = ["outputs/srg_ua.h5", "outputs/hrg_geolife.h5"]

# %% Setup
import importlib
import io

import altair as alt
import h5py
import numpy as np

import visualizations.heatmap as heatmap
import visualizations.histogram as histogram


def load(path):
    M1, M2, G = [], [], []
    with h5py.File(path, "r") as f:
        for key, target in (("M1", M1), ("M2", M2), ("G", G)):
            matches = [match for match in f.keys() if match.startswith(f"{key}_")]
            target += [f[match][()] for match in matches]

    return M1, M2, G


# %% Load data
M1, M2, G = load(files[0])
# _, _, G2 = load(files[1])

M1s = np.vstack(M1)
M2s = np.vstack(M2)
Gs = np.vstack(G)
# %%
importlib.reload(heatmap)
crop = None

# cmaps = [
#     [[255, 255, 255], [255, 50, 50]],
#     [[255, 255, 255], [50, 255, 50]],
#     [[255, 255, 255], [50, 50, 255]],
# ]

cmaps = [
    [[255, 255, 255], [150, 25, 25]],
    [[255, 255, 255], [25, 150, 25]],
    [[255, 255, 255], [25, 25, 150]],
]

M1chart = heatmap.finite_visit_location(M1s, crop=crop, cmap=cmaps[0])
M2chart = heatmap.finite_visit_location(M2s, crop=crop, cmap=cmaps[1])
Gchart = heatmap.finite_visit_location(Gs, crop=crop, cmap=cmaps[2])

chart = (
    alt.hconcat(M1chart, M2chart, Gchart)
    .resolve_scale(x="shared", y="shared", color="independent")
    .configure_legend(disable=True)
)

# chart = (
#     alt.layer(M1chart, M2chart, Gchart)
#     .resolve_scale(color="independent")
#     .configure_legend(disable=True)
# )

chart.save("outputs/heatmap.html")

cmaps = [
    [[255, 100, 100], [200, 100, 100]],
    [[100, 255, 100], [100, 200, 100]],
    [[100, 100, 255], [100, 100, 200]],
]

M1chart = heatmap.finite_visit_location_on_map(M1s, crop=crop, cmap=cmaps[0])
M2chart = heatmap.finite_visit_location_on_map(M2s, crop=crop, cmap=cmaps[1])
Gchart = heatmap.finite_visit_location_on_map(Gs, crop=crop, cmap=cmaps[2])

M1chart.save("outputs/M1.html")
M2chart.save("outputs/M2.html")
Gchart.save("outputs/G.html")

# %% Generate histograms
importlib.reload(histogram)

geolife_histogram = None
ua_histogram = None

with h5py.File("outputs/time_histogram.h5", "r") as f:
    geolife_histogram = f["geolife"][:]
    ua_histogram = f["ua"][:]

chart = histogram.plot_time_histogram(ua_histogram, bandwidth=0.25)
chart.save("outputs/ua_histogram.html")
chart = histogram.plot_time_histogram(geolife_histogram, bandwidth=0.25, correction=8)
chart.save("outputs/geolife_histogram.html")
