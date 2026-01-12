import altair as alt
# needed to work with very large datasets
alt.data_transformers.enable("vegafusion")

import folium

import numpy as np
import pandas as pd

def rgb_to_hex(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

"""
We assume that the data is a tensor with any number of dimensions, as long as
the final dimension is 2. 
"""
def finite_visit_location(data, crop=None, cmap=[(0, 0, 0), (255, 255, 255)]):
    data_reshaped = data.reshape((-1, 2))
    finite_rows = data_reshaped[np.isfinite(data_reshaped).all(axis=1)]

    # convert RGB to hex
    cmap = [rgb_to_hex(color) for color in cmap]
    
    if crop is not None:
        lat_min, lat_max = crop[0]
        lon_min, lon_max = crop[1]
        mask = ((finite_rows[:, 0] >= lat_min) & (finite_rows[:, 0] <= lat_max) &
                (finite_rows[:, 1] >= lon_min) & (finite_rows[:, 1] <= lon_max))
        finite_rows = finite_rows[mask]
        lat_domain, lon_domain = crop[0], crop[1]
        lat_bin = alt.Bin(maxbins=32, extent=crop[0])
        lon_bin = alt.Bin(maxbins=32, extent=crop[1])
        lat_scale = alt.Scale(zero=False, domain=lat_domain, nice=False)
        lon_scale = alt.Scale(zero=False, domain=lon_domain, nice=False)
    else:
        lat_domain, lon_domain = None, None
        lat_bin = alt.Bin(maxbins=32)
        lon_bin = alt.Bin(maxbins=32)
        lat_scale = alt.Scale(zero=False)
        lon_scale = alt.Scale(zero=False)
    
    df = pd.DataFrame(finite_rows, columns=['Latitude', 'Longitude'])
    chart = alt.Chart(df).mark_rect(opacity=0.5).encode(
        x=alt.X('Longitude:Q', bin=lon_bin, scale=lon_scale, axis=None),
        y=alt.Y('Latitude:Q', bin=lat_bin, scale=lat_scale, axis=None),
        color=alt.Color('count()', scale=alt.Scale(range=cmap, type='log'), 
                        # legend=None
                        )
    ).properties(width=300, height=300)
    return chart

def finite_visit_location_on_map(data, crop=None, bins=32, cmap=[[0]*3] * 2):
    data_reshaped = data.reshape((-1, 2))
    finite_rows = data_reshaped[np.isfinite(data_reshaped).all(axis=1)]
    
    if crop is not None:
        lat_min, lat_max = crop[0]
        lon_min, lon_max = crop[1]
        mask = ((finite_rows[:, 0] >= lat_min) & (finite_rows[:, 0] <= lat_max) &
                (finite_rows[:, 1] >= lon_min) & (finite_rows[:, 1] <= lon_max))
        finite_rows = finite_rows[mask]
    
    # Create 2D histogram
    lat_edges = np.linspace(finite_rows[:, 0].min(), finite_rows[:, 0].max(), bins + 1)
    lon_edges = np.linspace(finite_rows[:, 1].min(), finite_rows[:, 1].max(), bins + 1)
    counts, _, _ = np.histogram2d(finite_rows[:, 0], finite_rows[:, 1], bins=[lat_edges, lon_edges])
    
    # Create folium map
    center_lat = finite_rows[:, 0].mean()
    center_lon = finite_rows[:, 1].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15,
                   tiles='CartoDB Positron')
    
    # Normalize counts for color mapping (log scale)
    log_counts = np.log10(counts + 1)
    max_log = log_counts.max()
    
    # Add rectangles
    for i in range(len(lat_edges) - 1):
        for j in range(len(lon_edges) - 1):
            if counts[i, j] > 0:
                intensity = log_counts[i, j] / max_log

                def bilinear_color(I, J, c00, c11):
                    """Bilinear interp: c00=(0,0) → c11=(1,1)"""
                    c01 = c00  # Top-left = Top-right for simple gradient
                    c10 = c11  # Bottom-left = Bottom-right for simple gradient
                    color = c00 * (1-I)*(1-J) + c10 * I*(1-J) + c01 * (1-I)*J + c11 * I*J
                    return (color.astype(int)).tolist()

                x, y = i/len(lat_edges), j/len(lon_edges)
                start_color = np.array(cmap[0])
                end_color = np.array(cmap[1])
                rgb = bilinear_color(x, y, start_color, end_color)
                fill = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

                folium.Rectangle(
                    bounds=[[lat_edges[i], lon_edges[j]], [lat_edges[i+1], lon_edges[j+1]]],
                    color='none',
                    fill=True,
                    fillColor=fill,
                    fillOpacity=intensity
                ).add_to(m)
    
    return m
