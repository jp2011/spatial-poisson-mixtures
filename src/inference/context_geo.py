import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numba import jit
import geopandas as gpd
import pysal as ps


class GridContextGeo:
    """Class representing the context of the model and the experiment"""

    def __init__(self, *,
                 interpolation='weighted',
                 year='12017-122017',
                 resolution=400,
                 crime_type='burglary',
                 model_name='burglary_raw_4',
                 cov_collection_unit='lsoa',
                 covariates_type = 'raw'):

        if not covariates_type:
            file_name = f"grid-{cov_collection_unit}-{interpolation}-spatial-counts-covariates-{year}-{resolution}.gpk"
        else:
            file_name = f"grid-{covariates_type}-{cov_collection_unit}-{interpolation}-spatial-counts-covariates-{year}-{resolution}.gpk"

        full_path = Path("data") / "processed" / file_name
        all_data_gdf = gpd.read_file(full_path)

        self.resolution = resolution

        centroids = all_data_gdf.centroid
        centroids_df = pd.DataFrame({'x': centroids.x, 'y': centroids.y})
        centroids_ordered = centroids_df.sort_values(by=['x', 'y'], ascending=True)
        all_data_gdf = all_data_gdf.loc[centroids_ordered.index]
        self.full_dataset = all_data_gdf

        # Useful for Moran's I statistic
        self.spatial_weights = ps.lib.weights.Kernel.from_dataframe(all_data_gdf, function='gaussian', bandwidth=2*resolution)

        self.sp_geom = all_data_gdf.geometry.reset_index(drop=True)

        x_range_ticks = all_data_gdf.centroid.x.sort_values(ascending=True).unique()
        y_range_ticks = all_data_gdf.centroid.y.sort_values(ascending=True).unique()
        x_ticks_all, y_ticks_all = np.meshgrid(x_range_ticks, y_range_ticks)

        xy_ticks_all = np.stack((x_ticks_all.flatten(), y_ticks_all.flatten()), axis=1)
        xy_ticks_all = xy_ticks_all[np.lexsort((y_ticks_all.flatten(), x_ticks_all.flatten()))]
        full_grid_index = pd.MultiIndex.from_arrays([xy_ticks_all[:, 0], xy_ticks_all[:, 1]], names=('x', 'y'))

        # build the mask and the count vectors
        self.mask = pd.Series(index=full_grid_index, data=np.zeros(shape=(x_range_ticks.shape[0] * y_range_ticks.shape[0]), dtype=bool))
        for (x, y) in zip(centroids.x, centroids.y):
            self.mask.loc[x, y] = True

        # COUNTS
        crime_type_col_name = crime_type
        self.counts = all_data_gdf[crime_type_col_name].values  # training (in-sample) counts
        self.test_counts = all_data_gdf[f"test.{crime_type_col_name}"].values  # test (out-of-sample) counts

        # useful to keep the original cells centroids
        self.x_cells_ticks = centroids.x.values
        self.y_cells_ticks = centroids.y.values

        # also export ranges with unity distance
        self.x_ticks = (x_range_ticks - np.min(x_range_ticks)) / resolution
        self.y_ticks = (y_range_ticks - np.min(y_range_ticks)) / resolution
        self.dim_x = self.x_ticks.shape[0]
        self.dim_y = self.y_ticks.shape[0]

        # LSOA and MSOA memmbership
        self.lsoas = all_data_gdf['LSOA11CD'].values
        self.msoas = all_data_gdf['MSOA11CD'].values
        self.lads = all_data_gdf['LAD11CD'].values
        self.wards = all_data_gdf['WARD'].values

        # COVARIATES
        model_config_file_path = Path(os.getcwd()) / 'models' / 'config' / f"{model_name}.yml"
        with open(model_config_file_path, 'r') as stream:
            try:
                covariates_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        covariates_names = covariates_config['covariates']
        covariates_names_to_log = covariates_config['covariates_to_log']

        if len(covariates_names) != 0:

            # Log and normalise the covariates before expanding to full grid
            if covariates_names_to_log and len(covariates_names_to_log) > 0:
                all_data_gdf[covariates_names_to_log] = np.log(
                    all_data_gdf[covariates_names_to_log])

            column_means = np.mean(all_data_gdf[covariates_names], axis=0)
            column_sds = np.std(all_data_gdf[covariates_names], axis=0)
            normalised_covariates = (all_data_gdf[covariates_names] - column_means) / column_sds

            if('intercept' in covariates_names):
                normalised_covariates['intercept'] = 1
            all_data_gdf[covariates_names] = normalised_covariates

            self.covariates = all_data_gdf[covariates_names].values
            self.covariates_names = covariates_names

            self.J = len(covariates_names)
        else:
            self.covariates = None
            self.covariates_names = None

            self.J = 0

        self.N = self.counts.shape[0] # number of cells

    def compute_moran(self, observations):
        moran_statistic = ps.explore.esda.moran.Moran(observations, self.spatial_weights)
        return moran_statistic.I

    def plot_realisations(self, realisations, plot_no, plot_title=""):
        gdf = gpd.GeoDataFrame(pd.DataFrame({'data': realisations}))
        gdf = gdf.set_geometry(self.sp_geom)

        ax = plt.subplot(plot_no)
        ax.set_title(plot_title)
        ax.set_xlabel("Easting")
        ax.set_ylabel("Northing")
        return gdf.plot(column='data', ax=ax, legend=True, cmap='RdYlBu_r')


@jit(nopython=True)
def gp_inflate_duplicate(gp_matrix_block, block_membership, N, K):
    """Create a grid such that the cells inherit values from the block to which they belong"""
    gp_matrix_grid = np.zeros((N, K))
    for n in range(N):
        gp_matrix_grid[n, :] = gp_matrix_block[block_membership[n], :]
    return gp_matrix_grid


@jit(nopython=True)
def gp_deflate_sum(gp_matrix_full, block_membership, N, B, K):
    """Reduced a full-size grid to blocks such that the values at the cell level are summed on a per-block basis"""
    output = np.zeros((B, K))
    for n in range(N):
        b = block_membership[n]
        output[b, :] = output[b, :] + gp_matrix_full[n, :]
    return output