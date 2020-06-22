import logging
import os
import pickle
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import zsampler
from dotenv import load_dotenv, find_dotenv

from src.inference.context_geo import GridContextGeo
from src.inference.hmc import HMCSampler
from src.inference.priors import BetaPriorWithIntercept
from src.experiment.model_management import build_block_mixture_flat_uid
from src.experiment.visualize import plot_traceplots


class FlatRegionMixture:

    def __init__(self, *, uid=None,
                 grid_context=None,
                 K=1,
                 thinning=5,
                 n_info_interval=1000,
                 block_type="msoa"):
        self.uid = uid
        self.context = grid_context
        self.K = K
        self.NN = self.context.mask.shape[0]
        self.thinning = thinning
        self.n_info_interval = n_info_interval
        N = grid_context.counts.shape[0]

        # do a random assignment to mixtures
        initial_Z = np.zeros((N, K), dtype=int)
        initial_Z[np.arange(N), np.random.choice(K, N)] = 1

        # Create an (N x 1) vector which gives the corresponding block for each cell.
        if block_type == "lad":
            block_assignment = np.asarray(grid_context.lads)
        elif block_type == "msoa":
            block_assignment = np.asarray(grid_context.msoas)
        elif block_type == "ward":
            block_assignment = np.asarray(grid_context.wards)
        elif block_type == "lsoa":
            block_assignment = np.asarray(grid_context.lsoas)
        else:
            block_assignment = np.repeat(1, N) # a single block

        unique_block_labels = np.unique(block_assignment)
        self.block_assignment_numeric = np.zeros(block_assignment.shape[0], dtype=np.int)
        for idx_cell, block_label in enumerate(block_assignment):
            self.block_assignment_numeric[idx_cell] = np.where(unique_block_labels == block_label)[0]
        self.block_assignment = block_assignment

        self.B = np.max(self.block_assignment_numeric) + 1

        # Create B x K matrix which counts number of cells in a block b with assignment k.
        self.block_label_counts = np.zeros(shape=(self.B, K), dtype=np.int64)
        for i in range(N):
            self.block_label_counts[self.block_assignment_numeric[i], :] += initial_Z[i, :]

        # Beta prior
        self.beta_prior = BetaPriorWithIntercept(a=1, b=0.01)
        self.Z = initial_Z

        self.Z_samples = []
        self.alpha_samples = []
        self.beta_samples = []

        self.logger = logging.getLogger(__name__)

    def loglik(self, estimand):
        """ Compute log p(y | beta, Z) + log p(beta)"""
        J = self.context.J
        K = self.K
        covariates = self.context.covariates
        counts = self.context.counts

        beta = estimand[:(J * K)]
        beta_matrix = beta.reshape((J, K), order='F')

        fixed_effects = np.sum(np.multiply(self.Z, np.dot(covariates, beta_matrix)), axis=1)

        poisson_part = np.sum(np.multiply(counts, fixed_effects) - np.exp(fixed_effects))
        self.logger.debug(f"Poisson part: {poisson_part}")
        beta_part = self.beta_prior.log_pdf(beta, J)

        output = poisson_part + beta_part
        return output

    def nabla_loglik(self, estimand):
        """ Compute the gradient of log p(y | beta, Z) + log p(beta) with respect to beta"""

        J = self.context.J
        K = self.K
        covariates = self.context.covariates
        counts = self.context.counts

        beta = estimand[:(J * K)]
        beta_matrix = beta.reshape((J, K), order='F')
        fixed_effects = np.sum(np.multiply(self.Z, np.dot(covariates, beta_matrix)), axis=1)

        # nabla beta
        nabla_beta_matrix = np.zeros(beta_matrix.shape)
        nabla_beta_matrix += np.dot(covariates.T, self.Z * counts[:, np.newaxis])

        temp = np.exp(fixed_effects)
        nabla_beta_matrix += (- np.dot(covariates.T, self.Z * temp[:, np.newaxis]))
        nabla_beta = nabla_beta_matrix.flatten('F')
        nabla_beta += self.beta_prior.nabla_beta_log_pdf(beta, J)

        output = nabla_beta
        return output

    def plot_traces(self, hmc_samples):

        samples_array = np.asarray(hmc_samples)

        S = samples_array.shape[0]
        J = self.context.J
        N = self.context.N
        K = self.K

        # discard irrelevant samples
        self.Z_samples = self.Z_samples[(-S):]
        Z_samples_array = np.asarray(self.Z_samples)

        mixture_allocation = np.zeros((S, N, K))
        mixture_allocation[np.repeat(range(S), N), np.tile(range(N), S), Z_samples_array.flatten(order='C')] = 1
        average_alloc = np.mean(mixture_allocation, axis=0)

        for k in range(self.K):
            plt.figure()
            self.context.plot_realisations(average_alloc[:, k], 111)
            plt.show()

            beta_k_samples = samples_array[:, (k * J):((k + 1) * J)]
            plot_traceplots(beta_k_samples, self.context.covariates_names)
            plt.show()

            k_component_indices = np.where(average_alloc[:, k] > (1 / K))[0]

            # Fitted surface
            fitted_surface_map = np.multiply(average_alloc[:, k],
                                             np.dot(self.context.covariates, np.mean(beta_k_samples, axis=0)))
            plt.figure()
            self.context.plot_realisations(fitted_surface_map, 111, plot_title="Log-intensity fitted")
            plt.show()

            # Correlation Matrix
            crime_surface_k = np.dot(self.context.covariates[k_component_indices, :],
                                     np.mean(beta_k_samples, axis=0))
            surface_vars = np.concatenate((np.log(1 + self.context.counts[k_component_indices]).reshape(-1, 1),
                                           crime_surface_k.reshape(-1, 1),
                                           self.context.covariates[k_component_indices, 1:]), axis=1)
            surface_vars_df = pd.DataFrame(surface_vars)
            surface_vars_df.columns = ['log-y', 'log-fitted'] + self.context.covariates_names[1:]
            corr = surface_vars_df.corr()

            f, ax = plt.subplots(figsize=(15, 10))
            sns.heatmap(corr, annot=True, linewidths=.5, ax=ax)
            plt.show()

    def load_samples_snapshot(self, iteration_no):
        beta_filepath = Path(os.getcwd()) / "models" / "snapshots" / f"beta-samples--{self.uid}--{iteration_no}.npy"
        z_filepath = Path(os.getcwd()) / "models" / "snapshots" / f"Z-samples--{self.uid}--{iteration_no}.npy"
        alpha_filepath = Path(os.getcwd()) / "models" / "snapshots" / f"alpha-samples--{self.uid}--{iteration_no}.npy"
        beta_samples = np.load(beta_filepath)
        z_samples = np.load(z_filepath)
        try:
            alpha_samples = np.load(alpha_filepath)
            return beta_samples, z_samples, alpha_samples
        except FileNotFoundError:
            alpha_samples = np.zeros((1, self.K))
            return beta_samples, z_samples, alpha_samples

    def get_initial_estimand(self):
        beta = np.random.normal(0, 1, self.context.J * self.K)
        return beta

    def get_mass_matrix_diag(self):
        beta_m_diag = 5e2 * np.ones(self.context.J * self.K)
        return beta_m_diag

    def __save_output(self, iteration):

        folder_name = Path(os.getcwd()) / "models" / "snapshots"

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        Z_full_path = folder_name / f"Z-samples--{self.uid}--{iteration}"
        Z_samples_array = np.asarray(self.Z_samples)
        if Z_samples_array.shape[0] > 0:
            np.save(Z_full_path, Z_samples_array[::self.thinning, :])

        alpha_full_path = folder_name / f"alpha-samples--{self.uid}--{iteration}"
        alpha_array = np.asarray(self.alpha_samples)
        if alpha_array.shape[0] > 0:
            np.save(alpha_full_path, alpha_array[::self.thinning, :])

        beta_full_path = folder_name / f"beta-samples--{self.uid}--{iteration}"
        beta_array = np.asarray(self.beta_samples)
        if beta_array.shape[0] > 0:
            np.save(beta_full_path, beta_array[::self.thinning, :])

    def run_sampling(self, beta_sampler, number_of_iterations):
        N = self.context.N
        J = self.context.J
        K = self.K
        B = self.block_label_counts.shape[0]

        alpha = np.repeat(1/K, K)
        covariates = self.context.covariates
        counts = self.context.counts

        iteration = 0
        while iteration < number_of_iterations:

            ##########################################################################################
            # BOOKKEEPING
            ##########################################################################################
            # The HMC sampler is adaptive and therefore will discard samples during adaptive phase.
            if len(beta_sampler.samples) < len(self.Z_samples):
                num_current_samples = len(beta_sampler.samples)
                self.Z_samples = self.Z_samples[(-num_current_samples):]
                self.alpha_samples = self.alpha_samples[(-num_current_samples):]
                self.beta_samples = beta_sampler.samples

            if (iteration + 1) % self.n_info_interval == 0:
                self.__save_output(iteration)

            ##########################################################################################
            # SAMPLE ALPHA
            ##########################################################################################
            # We fix alpha to 1/K so there is no need for sampling. Should we choose to treat alpha
            # as random, the samples would have to be saved so we keep the code below for that eventuality.
            self.alpha_samples.append(alpha)
            self.logger.debug(f"Alpha: {alpha[0]}")

            ##########################################################################################
            # SAMPLE BETA
            ##########################################################################################
            beta_sampler.sample_one()

            ##########################################################################################
            # SAMPLE Z
            ##########################################################################################
            current_hmc_estimand = beta_sampler.estimand
            current_beta = current_hmc_estimand[:(J * K)].reshape((J, K), order='F')

            Z_new = zsampler.sample_region(self.Z.astype(np.int64),
                                           counts.astype(np.int64),
                                           covariates.astype(np.float64),
                                           current_beta.astype(np.float64),
                                           alpha.astype(np.float64),
                                           self.block_assignment_numeric.astype(np.int64),
                                           self.block_label_counts)

            self.Z_samples.append(np.where(Z_new > 0)[1])
            self.Z = Z_new

            iteration += 1

        self.logger.info("Sampling completed - saving model.")
        self.beta_samples = beta_sampler.samples
        self.__save_output(iteration)


@click.command()
@click.option('--year', '-y', type=str, default='12015-122015')
@click.option('--type', '-t', default='burglary')
@click.option('--resolution', '-r', type=int, default=400)
@click.option('--model_name', '-m', type=str, default='burglary_raw_0')
@click.option('--interpolation', '-i', type=str, default='weighted')
@click.option('--num_mixtures', '-K', type=int, default=3)
@click.option('--uid', type=str, default=None)
@click.option('--verbose', is_flag=True)
@click.option('--block_type', type=str, default="msoa")
@click.option('--collection_unit', type=str, default="lsoa")
def main(year, type, resolution, model_name, interpolation, num_mixtures, uid, verbose,
         block_type, collection_unit):

    if uid is None:
        uid = build_block_mixture_flat_uid(prefix="BLOCK-MIXTURE-FLAT", chain_no=1, block_scheme=block_type,
                                           c_type=type, t_period=year, model_spec=model_name,
                                           cov_interpolation=interpolation, resolution=resolution, K=num_mixtures)

    log_fmt = '[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s'
    datefmt = '%H:%M:%S'
    if verbose:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=log_fmt)
    else:
        logging.basicConfig(filename=Path('models') / f"log-{uid}.log",
                            filemode='a',
                            format=log_fmt,
                            datefmt=datefmt,
                            level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.info("Building the context.")
    grid_context = GridContextGeo(interpolation=interpolation,
                                  year=year,
                                  resolution=resolution,
                                  crime_type=type,
                                  model_name=model_name,
                                  cov_collection_unit=collection_unit,
                                  covariates_type='raw')

    logger.info("Writing sampling context into a file.")
    context_filename = Path(os.getcwd()) / "models" / f"context--{uid}.pickle"
    with open(context_filename, 'wb') as context_file:
        context_info = {
            'context': grid_context,
            'K': num_mixtures
        }
        pickle.dump(context_info, context_file)

    logger.info("Initialising the model with estimand and mass matrix diagonal")
    model = FlatRegionMixture(uid=uid,
                              grid_context=grid_context,
                              K=num_mixtures,
                              n_info_interval=50_000,
                              thinning=10,
                              block_type=block_type)
    init_estimand = model.get_initial_estimand()
    mass_matrix_diag = model.get_mass_matrix_diag()

    logger.info("Launching HMC sampler.")
    hmc_all_iterations = 120_000
    sampler = HMCSampler(func_lpdf=model.loglik,
                         func_nabla_lpdf=model.nabla_loglik,
                         func_plot=model.plot_traces if verbose else None,
                         init_estimand=init_estimand,
                         init_M_diag=mass_matrix_diag,
                         init_L=10,
                         init_epsilon=5.0e-2,
                         n_burnin=30_000,
                         n_calib=60_000,
                         S=hmc_all_iterations,
                         n_info_interval=model.n_info_interval,
                         thinning=model.thinning,
                         unique_estimation_id=uid,
                         adaptive=True)
    if verbose:
        plt.figure()
        grid_context.plot_realisations(np.log(model.block_assignment_numeric + 1), 111)
        plt.show()

    model.run_sampling(sampler, hmc_all_iterations)
    logger.info("Procedure finished.")


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    main()
