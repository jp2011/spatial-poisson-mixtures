import logging
import os
import pickle
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv, find_dotenv

from src.inference.context_geo import GridContextGeo
from src.inference.hmc import HMCSampler
from src.inference.priors import BetaPriorWithIntercept, GaussianPrior, GPGridPriorMatern
from src.experiment.model_management import build_lgcp_uid
from src.experiment.visualize import plot_traceplots


class LgcpMatern:

    def __init__(self, *, uid=None,
                 grid_context=None,
                 thinning=5,
                 n_info_interval=1000):
        self.uid = uid
        self.context = grid_context
        self.NN = self.context.mask.shape[0]
        self.thinning = thinning
        self.n_info_interval = n_info_interval

        # GP prior
        self.gp_prior = GPGridPriorMatern(coord_x=grid_context.x_ticks, coord_y=grid_context.y_ticks,
                                          smoothness=1.5)

        # Beta prior
        self.beta_prior = BetaPriorWithIntercept(a=1, b=0.01)

        # log-theta prior
        self.log_theta_prior = GaussianPrior(mean=np.asarray([0, 0]),
                                             variance=np.asarray([1, 1]))

        self.logger = logging.getLogger(__name__)

    def loglik(self, estimand):
        J = self.context.J
        covariates = self.context.covariates
        counts = self.context.counts

        f = estimand[:self.NN]
        f_cut = f[self.context.mask]
        beta = estimand[(self.NN):(self.NN + J)]
        log_theta = estimand[(self.NN + J):]

        gp_variance = np.exp(log_theta[0])
        gp_lengthscale = np.exp(log_theta[1])

        fixed_effects = np.dot(covariates, beta)
        poisson_part = np.sum(np.multiply(counts, fixed_effects + f_cut) - np.exp(fixed_effects + f_cut))

        gp_part = self.gp_prior.get_logpdf(variance=gp_variance, lengthscale=gp_lengthscale, f=f)

        beta_part = self.beta_prior.log_pdf(beta, J)

        log_theta_part = self.log_theta_prior.log_pdf(log_theta)

        output = poisson_part + gp_part + beta_part + log_theta_part
        return output

    def nabla_loglik(self, estimand):

        J = self.context.J
        covariates = self.context.covariates
        counts = self.context.counts

        f = estimand[:self.NN]
        f_cut = f[self.context.mask]
        beta = estimand[(self.NN):(self.NN + J)]
        log_theta = estimand[(self.NN + J):]

        gp_variance = np.exp(log_theta[0])
        gp_lengthscale = np.exp(log_theta[1])

        fixed_effects = np.dot(covariates, beta)

        # nabla f
        nabla_f = np.zeros(f.shape)
        nabla_f[self.context.mask] += (counts - np.exp(f_cut + fixed_effects))
        nabla_f += self.gp_prior.get_nabla_f(variance=gp_variance, lengthscale=gp_lengthscale, f=f)

        # nabla beta
        nabla_beta = np.dot(covariates.T, counts - np.exp(fixed_effects + f_cut))
        nabla_beta += self.beta_prior.nabla_beta_log_pdf(beta, J)

        # nabla log-theta
        nabla_theta = self.gp_prior.get_nabla_theta(variance=gp_variance, lengthscale=gp_lengthscale, f=f)
        nabla_log_theta = np.multiply(nabla_theta, np.exp(log_theta))
        nabla_log_theta += self.log_theta_prior.nabla_x_log_pdf(log_theta)

        output = np.concatenate((nabla_f, nabla_beta, nabla_log_theta))
        return output


    def plot_traces(self, hmc_samples):

        samples_array = np.asarray(hmc_samples)

        S = samples_array.shape[0]
        NN = self.NN
        J = self.context.J
        N = self.context.N

        beta_samples = samples_array[:, (NN):(NN + J)]
        plot_traceplots(beta_samples, self.context.covariates_names)
        plt.show()

        f_samples = samples_array[:, :(NN)]
        indices_to_plot = np.random.choice(N, 4)
        f_on_map_samples = f_samples[:, self.context.mask]

        chosen_f_samples = f_on_map_samples[:, indices_to_plot]
        theta_samples = np.exp(samples_array[:, (NN + J):])

        samples_gp = np.concatenate((chosen_f_samples, theta_samples), axis=1)
        names = [str(i) for i in indices_to_plot] + ['variance', 'lengthscale']
        plot_traceplots(samples_gp, names)
        plt.show()

    def load_samples_snapshot(self, iteration_no):
        hmc_filepath = Path(os.getcwd()) / "models" / "snapshots" / f"hmc-samples--{self.uid}--{iteration_no}.npy"
        hmc_samples = np.load(hmc_filepath)
        return hmc_samples

    def get_initial_estimand(self):
        f = np.random.normal(0, .1, self.NN)
        beta = np.random.normal(0, 0.1, self.context.J)
        log_theta = np.random.normal(0, 0.1, 2)

        output = np.concatenate((f, beta, log_theta))
        return output

    def get_mass_matrix_diag(self):
        """
        Note that the scaling factors need to be tuned, depending on the resolution which determines the
        dimension  of the quantity we are sampling.
        """
        f_m_diag = 1e4 * np.ones(self.NN)
        beta_m_diag = 1e4 * np.ones(self.context.J)
        log_theta_m_diag = 1e4 * np.ones(2)

        output = np.concatenate((f_m_diag, beta_m_diag, log_theta_m_diag))
        return output

    def __save_output(self, full_path_to_file):
        # no need to save anything else except HMC stuff which is done automatically by the hmc.py routine.
        pass

    def run_sampling(self, hmc_sampler, number_of_iterations):

        iteration = 0
        while iteration < number_of_iterations:
            hmc_sampler.sample_one()
            iteration += 1

        self.logger.info("Sampling completed - saving model.")

        # save HMC samples too
        hmc_sampler.snapshot()


@click.command()
@click.option('--year', '-y', type=str, default='12015-122015')
@click.option('--type', '-t', default='burglary')
@click.option('--resolution', '-r', type=int, default=300)
@click.option('--model_name', '-m', type=str, default='burglary_raw_3')
@click.option('--interpolation', '-i', type=str, default='weighted')
@click.option('--uid', type=str, default=None)
@click.option('--verbose', is_flag=True)
def main(year, type, resolution, model_name, interpolation, uid, verbose):

    if uid is None:
        uid = build_lgcp_uid(prefix='LGCP-MATERN', chain_no=1, c_type=type, t_period=year, model_spec=model_name,
                             resolution=resolution, cov_interpolation=interpolation)

    # Logging matters
    log_fmt = '[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s'
    datefmt = '%H:%M:%S'
    if verbose:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=log_fmt)
    else:
        logging.basicConfig(filename=Path('models') / f"log-{uid}.log", filemode='a', format=log_fmt, datefmt=datefmt,
                            level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.info("Building the context.")
    grid_context = GridContextGeo(interpolation=interpolation,
                                  year=year,
                                  resolution=resolution,
                                  crime_type=type,
                                  model_name=model_name,
                                  covariates_type='raw')

    logger.info("Writing sampling context into a file.")
    context_filename = Path(os.getcwd()) / "models" / f"context--{uid}.pickle"
    with open(context_filename, 'wb') as context_file:
        context_info = {
            'context': grid_context
        }
        pickle.dump(context_info, context_file)

    logger.info("Initialising the model with estimand and mass matrix diagonal")
    model = LgcpMatern(uid=uid,
                       grid_context=grid_context,
                       n_info_interval=10_000,
                       thinning=5)
    init_estimand = model.get_initial_estimand()
    mass_matrix_diag = model.get_mass_matrix_diag()  # check above to tune the scale of the mass matrix.

    logger.info("Launching HMC sampler.")
    hmc_all_iterations = 150_000
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
                         adaptive=True, save_samples=True)

    if verbose:
        plt.figure()
        grid_context.plot_realisations(grid_context.counts, 111)
        plt.show()

    model.run_sampling(sampler, hmc_all_iterations)
    logger.info("Procedure finished.")

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    main()
