import logging
import os
import pickle
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zsampler
from dotenv import load_dotenv, find_dotenv
from scipy.special import logsumexp, softmax

from src.inference.context_geo import GridContextGeo, gp_inflate_duplicate, gp_deflate_sum
from src.inference.hmc import HMCSampler
from src.inference.priors import BetaPriorWithIntercept, GaussianPrior, GPNonGridPriorSqExpFixed
from src.experiment.visualize import plot_traceplots


class BlockMixtureGpSoftmaxAllocation:

    def __init__(self, *, uid=None,
                 grid_context=None,
                 K=1,
                 block_type="msoa",
                 hmc_all_iterations=100_000,
                 hmc_burn_in=25_000,
                 hmc_calibration=50_000,
                 hmc_info_interval=20_000,
                 hmc_thinning=5,
                 verbose=False,
                 lengthscale=1):
        self.uid = uid
        self.context = grid_context
        self.K = K
        self.NN = self.context.mask.shape[0]
        self.hmc_thinning = hmc_thinning
        self.hmc_info_interval = hmc_info_interval
        self.N = grid_context.counts.shape[0]
        self.J = self.context.J

        # do a random assignment to mixtures
        initial_Z = np.zeros((self.N, self.K), dtype=int)
        initial_Z[np.arange(self.N), np.random.choice(self.K, self.N)] = 1

        self.Z_samples = []

        # Create an (N x 1) vector which gives the corresponding block for each cell.
        if block_type == "lad":
            block_assignment = np.asarray(grid_context.lads)
        elif block_type == "msoa":
            block_assignment = np.asarray(grid_context.msoas)
        elif block_type == "ward":
            block_assignment = np.asarray(grid_context.wards)
        else:
            block_assignment = np.repeat(1, self.N) # a single block

        # read in block centroid coordinates
        block_centroid_file_path = Path(os.getcwd()) / "data" / "processed" / f"{block_type}-centroids-map.csv"
        block_centroids = pd.read_csv(block_centroid_file_path)
        self.coord_x = block_centroids["x"].values
        self.coord_x = self.coord_x - np.min(self.coord_x)
        self.coord_y = block_centroids["y"].values
        self.coord_y = self.coord_y - np.min(self.coord_y)

        self.block_labels = block_centroids.iloc[:, 1].values

        # Create the cell <-> block mapping (mind the ordering of the blocks)
        unique_block_labels = np.unique(self.block_labels)
        self.block_assignment_numeric = np.zeros(block_assignment.shape[0], dtype=np.int)
        for idx_cell, block_label in enumerate(block_assignment):
            self.block_assignment_numeric[idx_cell] = np.where(unique_block_labels == block_label)[0]
        self.block_assignment = block_assignment
        B = np.max(self.block_assignment_numeric) + 1
        self.B = B

        self.lengthscale = lengthscale

        # Priors
        self.beta_prior = BetaPriorWithIntercept(a=1, b=0.01)
        self.f_prior = GPNonGridPriorSqExpFixed(coord_x=self.coord_x, coord_y=self.coord_y,
                                                variance=100, lengthscale=self.lengthscale)
        self.log_theta_prior = GaussianPrior(mean=np.asarray([0]), variance=np.asarray([1e2]))

        init_beta_estimand = np.random.normal(0, 1, self.context.J * self.K)
        init_beta_mass_matrix = 1e3 * np.ones(self.context.J * self.K)
        self.beta_sampler = HMCSampler(func_lpdf=self.beta_loglik,
                                  func_nabla_lpdf=self.nabla_beta_loglik,
                                  func_plot=self.plot_beta if verbose else None,
                                  init_estimand=init_beta_estimand,
                                  init_M_diag=init_beta_mass_matrix,
                                  init_L=20,
                                  init_epsilon=5.0e-2,
                                  n_burnin=hmc_burn_in,
                                  n_calib=hmc_calibration,
                                  S=hmc_all_iterations,
                                  n_info_interval=hmc_info_interval,
                                  thinning=hmc_thinning,
                                  unique_estimation_id=uid,
                                  adaptive=True)

        init_f_estimand = np.random.normal(0, 1, B * self.K)
        init_f_mass_matrix = 1e4 * np.ones(B * self.K)
        self.f_sampler = HMCSampler(func_lpdf=self.f_loglik,
                               func_nabla_lpdf=self.nabla_f_loglik,
                               func_plot=self.plot_f if verbose else None,
                               init_estimand=init_f_estimand,
                               init_M_diag=init_f_mass_matrix,
                               init_L=100,
                               init_epsilon=5.0e-2,
                               n_burnin=hmc_burn_in,
                               n_calib=hmc_calibration,
                               S=hmc_all_iterations,
                               n_info_interval=hmc_info_interval,
                               thinning=hmc_thinning,
                               unique_estimation_id=uid,
                               adaptive=False)

        self.current_beta = self.beta_sampler.estimand
        self.current_f = self.f_sampler.estimand
        self.current_Z = initial_Z

        self.logger = logging.getLogger(__name__)

    def beta_loglik(self, beta_estimand):

        beta_matrix = np.reshape(beta_estimand, (self.J, self.K), order='F')  # build a J x K matrix
        Z = self.current_Z

        counts = self.context.counts
        covariates = self.context.covariates

        fixed_effects = np.sum(np.multiply(Z, np.dot(covariates, beta_matrix)), axis=1)

        poisson_part = np.sum(np.multiply(counts, fixed_effects) - np.exp(fixed_effects))
        beta_part = self.beta_prior.log_pdf(beta_estimand, self.J)

        output = poisson_part + beta_part
        return output

    def nabla_beta_loglik(self, beta_estimand):
        beta_matrix = np.reshape(beta_estimand, (self.J, self.K), order='F')  # build a J x K matrix

        counts = self.context.counts
        covariates = self.context.covariates
        Z = self.current_Z
        fixed_effects = np.sum(np.multiply(Z, np.dot(covariates, beta_matrix)), axis=1)
        nabla_beta_matrix = np.zeros(beta_matrix.shape)
        nabla_beta_matrix += np.dot(covariates.T, Z * counts[:, np.newaxis])

        temp = np.exp(fixed_effects)
        nabla_beta_matrix += (- np.dot(covariates.T, Z * temp[:, np.newaxis]))
        nabla_beta = nabla_beta_matrix.flatten('F')
        nabla_beta += self.beta_prior.nabla_beta_log_pdf(beta_estimand, self.J)

        output = nabla_beta
        return output


    def plot_beta(self, beta_samples):
        beta_samples_array = np.asarray(beta_samples)
        for k in range(self.K):
            beta_k_samples = beta_samples_array[:, (k * self.J):((k + 1) * self.J)]
            plot_traceplots(beta_k_samples, self.context.covariates_names)
            plt.show()

    def sample_Z(self):
        beta_matrix = np.reshape(self.current_beta, (self.J, self.K), order='F')  # build a J x K matrix
        f_matrix = np.reshape(self.current_f, (self.B, self.K), order='F')
        Z = self.current_Z

        f_full_matrix = gp_inflate_duplicate(f_matrix,
                                             self.block_assignment_numeric,
                                             self.N, self.K)
        counts = self.context.counts
        covariates = self.context.covariates

        fixed_effects_all = np.dot(covariates, beta_matrix)
        counts_matrix = np.repeat(counts.reshape((-1, 1)), self.K, axis=1)

        poi_lik = counts_matrix * fixed_effects_all - np.exp(fixed_effects_all)
        gp_log_softmax = f_full_matrix - logsumexp(f_full_matrix, axis=1)[:, np.newaxis]

        prob = softmax(poi_lik + gp_log_softmax, axis=1)

        new_Z = zsampler.sample_bulk_categorical(Z.astype(np.int64), prob.astype(np.float64))
        return new_Z

    def f_loglik(self, F_estimand):

        f_matrix = np.reshape(F_estimand, (self.B, self.K), order='F')
        Z = self.current_Z

        f_full_matrix = gp_inflate_duplicate(f_matrix,
                                             self.block_assignment_numeric,
                                             self.N, self.K)
        output = 0
        temp = f_full_matrix - logsumexp(f_full_matrix, axis=1)[:, np.newaxis]
        output += np.sum(np.multiply(Z, temp))

        for k in range(self.K):
            # GP contribution
            output += self.f_prior.get_logpdf(f=f_matrix[:, k])

        return output

    def nabla_f_loglik(self, F_estimand):
        f_matrix = np.reshape(F_estimand, (self.B, self.K), order='F')

        f_full_matrix = gp_inflate_duplicate(f_matrix,
                                             self.block_assignment_numeric,
                                             self.N, self.K)
        Z = self.current_Z

        f_gradient = np.zeros(f_matrix.shape)

        # nabla f poisson mixture
        temp_matrix = 1 - np.exp(f_full_matrix - logsumexp(f_full_matrix, axis=1)[:, np.newaxis])
        inflated_output_matrix = np.multiply(Z, temp_matrix)
        f_gradient += gp_deflate_sum(inflated_output_matrix, self.block_assignment_numeric, self.N, self.B, self.K)

        for k in range(self.K):
            f_gradient[:, k] += self.f_prior.get_nabla_f(f=f_matrix[:, k])

        return f_gradient.flatten(order='F')

    def plot_f(self, F_samples):

        f_array = np.asarray(F_samples).reshape((-1, self.B, self.K), order='F')
        S = f_array.shape[0]

        # discard irrelevant samples
        self.Z_samples = self.Z_samples[(-S):]
        Z_samples_array = np.asarray(self.Z_samples)

        mixture_allocation = np.zeros((S, self.N, self.K))
        mixture_allocation[np.repeat(range(S), self.N), np.tile(range(self.N), S), Z_samples_array.flatten(order='C')] = 1
        average_alloc = np.mean(mixture_allocation, axis=0)

        for k in range(self.K):
            plt.figure()
            self.context.plot_realisations(average_alloc[:, k], 111)
            plt.show()

        # plot a random traceplot
        idx1 = np.random.choice(self.B)
        plot_traceplots(f_array[:, idx1, :], [f"IDX: {idx1}: K={k}" for k in range(self.K)])
        plt.show()

        latent_weight_samples = softmax(np.mean(f_array, axis=0), axis=1)
        latent_weight_samples_full = gp_inflate_duplicate(latent_weight_samples,
                                                          self.block_assignment_numeric,
                                                          self.N, self.K)
        plt.figure()
        for k in range(self.K):
            self.context.plot_realisations(latent_weight_samples_full[:, k], 111)
            plt.show()


    def load_samples_snapshot(self, iteration_no):
        beta_filepath = Path(os.getcwd()) / "models" / "snapshots" / f"beta-samples--{self.uid}--{iteration_no}.npy"
        F_filepath = Path(os.getcwd()) / "models" / "snapshots" / f"F-samples--{self.uid}--{iteration_no}.npy"
        Z_filepath = Path(os.getcwd()) / "models" / "snapshots" / f"Z-samples--{self.uid}--{iteration_no}.npy"
        beta_samples = np.load(beta_filepath)
        F_samples = np.load(F_filepath)
        Z_samples = np.load(Z_filepath)
        return beta_samples, Z_samples, F_samples

    def __save_output(self, iteration):

        folder_name = Path(os.getcwd()) / "models" / "snapshots"

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        F_full_path = folder_name / f"F-samples--{self.uid}--{iteration}"
        F_samples_array = np.asarray(self.f_sampler.samples)
        if F_samples_array.shape[0] > 0:
            np.save(F_full_path, F_samples_array[::self.hmc_thinning, :])

        beta_full_path = folder_name / f"beta-samples--{self.uid}--{iteration}"
        beta_array = np.asarray(self.beta_sampler.samples)
        if beta_array.shape[0] > 0:
            np.save(beta_full_path, beta_array[::self.hmc_thinning, :])

        Z_full_path = folder_name / f"Z-samples--{self.uid}--{iteration}"
        Z_array = np.asarray(self.Z_samples)
        if Z_array.shape[0] > 0:
            np.save(Z_full_path, Z_array[::self.hmc_thinning, :])

    def run_sampling(self, number_of_iterations):
        iteration = 0
        while iteration < number_of_iterations:

            ##########################################################################################
            # BOOKKEEPING
            ##########################################################################################
            # The HMC samplers are independently adaptive and therefore will discard samples during the adaptive phase.
            num_current_samples = min(len(self.beta_sampler.samples),
                                      len(self.f_sampler.samples))

            self.beta_sampler.samples = self.beta_sampler.samples[(-num_current_samples):]
            self.f_sampler.samples = self.f_sampler.samples[(-num_current_samples):]
            self.Z_samples = self.Z_samples[(-num_current_samples):]

            if (iteration + 1) % self.hmc_info_interval == 0:
                self.__save_output(iteration)

            ##########################################################################################
            # SAMPLE BETA
            ##########################################################################################
            self.beta_sampler.sample_one()
            self.current_beta = self.beta_sampler.estimand

            ##########################################################################################
            # SAMPLE Z
            ##########################################################################################
            new_Z = self.sample_Z()
            self.Z_samples.append(np.where(new_Z > 0)[1])
            self.current_Z = new_Z

            ##########################################################################################
            # SAMPLE F
            ##########################################################################################
            self.f_sampler.sample_one()
            self.current_f = self.f_sampler.estimand

            iteration += 1

        self.logger.info("Sampling completed - saving model.")
        self.__save_output(iteration)


@click.command()
@click.option('--year', '-y', type=str, default='12013-122015')
@click.option('--type', '-t', default='burglary')
@click.option('--resolution', '-r', type=int, default=400)
@click.option('--model_name', '-m', type=str, default='burglary_raw_4')
@click.option('--interpolation', '-i', type=str, default='weighted')
@click.option('--num_mixtures', '-K', type=int, default=3)
@click.option('--uid', type=str, default=None)
@click.option('--verbose', is_flag=True)
@click.option('--block_type', type=str, default="lad")
@click.option('--collection_unit', type=str, default="lsoa")
@click.option('--lengthscale', type=float, default=1500.0)
def main(year, type, resolution, model_name, interpolation, num_mixtures, uid, verbose,
         block_type, collection_unit, lengthscale):
    if uid is None:
        uid = f"blockmixgp--{block_type}--{type}--{model_name}--{interpolation}--{num_mixtures}--{resolution}-{year}"

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

    hmc_all_iterations = 250_000
    hmc_info_interval = 50_000
    hmc_thinning = 10
    hmc_burn_in = 90_000
    hmc_calibration = 150_000

    model = BlockMixtureGpSoftmaxAllocation(uid=uid,
                                            grid_context=grid_context,
                                            K=num_mixtures,
                                            hmc_info_interval=hmc_info_interval,
                                            hmc_all_iterations=hmc_all_iterations,
                                            hmc_thinning=hmc_thinning,
                                            hmc_burn_in=hmc_burn_in,
                                            hmc_calibration=hmc_calibration,
                                            block_type=block_type,
                                            verbose=verbose,
                                            lengthscale=lengthscale)

    model.run_sampling(number_of_iterations=hmc_all_iterations)
    logger.info("Procedure finished.")


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    main()
