import numpy as np
import pickle
import h5py

import altair as alt
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d

def plot_time_histogram(data, bandwidth=0, fill_color='lightblue', 
                   stroke_color='steelblue', correction=0):
    data = np.array(data)
    
    if correction != 0:
        shift = int(len(data) * correction / 24)
        data = np.roll(data, shift)
    
    if bandwidth > 0:
        sigma = bandwidth * len(data) / 24
        pad = int(3 * sigma)
        wrapped = np.concatenate([data[-pad:], data, data[:pad]])
        smoothed = gaussian_filter1d(wrapped, sigma=sigma)
        data = smoothed[pad:-pad]
    
    x_interp = np.linspace(0, 24, 200)
    y_interp = np.interp(x_interp, np.linspace(0, 24, len(data)), data)
    
    df = pd.DataFrame({'hour': x_interp, 'density': y_interp})
    
    return alt.Chart(df).mark_area(
        line={'color': stroke_color},
        color=fill_color,
        opacity=0.7
    ).encode(
        x=alt.X('hour:Q', scale=alt.Scale(domain=[0, 24]), title='Hour of Day'),
        y=alt.Y('density:Q', title='Density', axis=None)
    ).properties(
        width=600, height=400
    ).configure_axis(
        grid=False
    ).configure_view(
        stroke=None
    )

def compute_marg_histogram(
    marg_path
):
    marg = None
    with open(marg_path, "rb") as f:
        marg = pickle.load(f)

    frequencies = np.zeros(marg.L)

    for node, data in marg.nodes(data=True):
        frequencies[data["time"]] += 1

    return frequencies

if __name__ == "__main__":
    marg_path = "../outputs/MARG_geolife.pkl"

    with h5py.File('../outputs/time_histogram.h5', 'w') as f:
        f.create_dataset('geolife',
                         data=compute_marg_histogram("../outputs/MARG_geolife.pkl"))
        f.create_dataset('ua',
                         data=compute_marg_histogram("../outputs/MARG_ua.pkl"))
