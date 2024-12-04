import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
from sklearn.cluster import KMeans
from bokeh.io import show
from bokeh.plotting import gmap
from bokeh.models import GMapOptions
from bokeh.models import ColumnDataSource
from bokeh.transform import linear_cmap
from bokeh.palettes import Turbo256  # A palette with 256 colors

bokeh_width, bokeh_height = 500,400
# Load the data
data_invaders = pd.read_csv('/Users/mfleury/POSTDOC/LIBRAIRY/Space_Invaders/1_input/dataframes/space_invaders.csv')

api_key = 'AIzaSyCqE4Sxap0crRpiy6RYYSoyhyvqMkShG4U'

def plot_map_space_invaders():
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    # latitudide as y, longitude as x
    sns.scatterplot(x='Longitude', y='Latitude', data=data_invaders, ax=ax, color='blue', s=0.5)
    plt.title('Space Invaders')
    plt.show()


def plot(lat, lng, zoom=10, map_type='roadmap', data_invaders=data_invaders):
    # Rename columns to lat/lon for consistency
    data_invaders = data_invaders.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})

    # Setup Google Maps options
    gmap_options = GMapOptions(lat=lat, lng=lng, map_type=map_type, zoom=zoom)

    # Initialize GMap plot
    p = gmap(api_key, gmap_options, title='Pays de Gex', width=800, height=600)

    # Create a ColumnDataSource from the data
    source = ColumnDataSource(data_invaders)

    # Define a color mapper using a larger palette (Turbo256) for many clusters
    cluster_min = data_invaders['cluster'].min()
    cluster_max = data_invaders['cluster'].max()
    color_mapper = linear_cmap(field_name='cluster', palette=Turbo256, low=cluster_min, high=cluster_max)

    # Plot the data with a continuous color map based on cluster
    p.circle('lon', 'lat', size=10, alpha=0.6, color=color_mapper, source=source)

    # Show the plot
    show(p)
    return p


def main():
    # plot_map_space_invaders()

    # we want to clusterise the data
    # 1. Select the features
    X = data_invaders[['Latitude', 'Longitude']]
    # 2. Create the model
    kmeans = KMeans(n_clusters=20)
    # 3. Fit the model
    kmeans.fit(X)
    # 4. Predict the clusters
    data_invaders['cluster'] = kmeans.predict(X)
    # 5. Plot the clusters
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x='Longitude', y='Latitude', data=data_invaders, ax=ax, hue='cluster', palette='tab10', s=10)
    ax.set_xlim(2.25, 2.45)
    ax.set_ylim(48.8, 48.9)
    plt.title('Space Invaders')
    # plt.show()

    p = plot(48.8, 2.35, zoom=12, map_type='roadmap', data_invaders=data_invaders)






if __name__ == '__main__':
    main()