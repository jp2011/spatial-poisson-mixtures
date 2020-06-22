
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import Matern, RBF

from src.numeric.kronecker_algebra import kron_invert_matrices, kron_mv_prod, kron_log_det


class BetaPriorWithIntercept:
    """
    Normal-InvGamma prior for the regression coefficients. The intercept is given a uniform (improper) prior.
    """

    def __init__(self, a=1, b=0.001):
        self.a = a
        self.b = b

    def log_pdf(self, beta, J):
        non_intercept_beta = np.delete(beta, np.arange(0, beta.shape[0], J))
        output = (-self.a - 0.5) * np.sum(np.log(0.5 * np.square(non_intercept_beta) + self.b))
        return output

    def nabla_beta_log_pdf(self, beta, J):
        """
        Return the gradient of the log of the pdf of the prior with respect to the beta values.
        """
        part1 = (-self.a - 0.5) * beta
        part2 = 0.5 * np.square(beta) + self.b

        gradient = np.divide(part1, part2)
        gradient[::J] = 0 # intercept has uniform prior
        return gradient

class BetaPriorNoIntercept:
    """
    Normal-InvGamma prior for the regression coefficients which do not include the intercept.
    """

    def __init__(self, a=1, b=0.001):
        self.a = a
        self.b = b

    def log_pdf(self, beta):
        """
        Return the log of the pdf of the prior for the regression coefficients.
        """
        output = (-self.a - 0.5) * np.sum(np.log(0.5 * np.square(beta) + self.b))
        return output

    def nabla_beta_log_pdf(self, beta):
        """
        Return the gradient of the log of the pdf of the prior with respect to the beta values.
        """
        part1 = (-self.a - 0.5) * beta
        part2 = 0.5 * np.square(beta) + self.b
        gradient = np.divide(part1, part2)
        return gradient


class GaussianPrior:

    """A generic prior for N iid Gaussian random variables"""
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def log_pdf(self, x):
        output = -0.5 * np.dot(np.square(x - self.mean), 1 / self.variance)
        return output

    def nabla_x_log_pdf(self, x):
        """
        The gradient of the log-pdf with respect to the values of x
        """
        output = -1 * np.divide(x - self.mean, self.variance)
        return output


class GPPrior:
    """
    Generic class for the zero-mean Gaussian process prior with the covariance function that has paremeters theta.
    """
    def __init__(self):
        pass

    def get_cov_mmatrix(self, *, variance=1, lengthscale=1):
        pass

    def get_logpdf(self, *, variance=1, lengthscale=1, f=None):
        pass

    def get_nabla_f(self, *, variance=None, lengthscale=None, f=None):
        pass

    def get_nabla_theta(self, *, variance=None, lengthscale=None, f=None):
        pass


class GPGridPriorMatern:
    """
    Gaussian process prior for a 2D grid with zero-mean and Matern covariance function.

    Given that the domain is a grid, Kronecker algebra can be used to speed up the computations. Before reading the
    code, we recommend reading the relevant parts of PhD thesis of Yunus Saatci:

    Saatçi, Yunus. 2012. ‘Scalable Inference for Structured Gaussian Process Models’. PhD Thesis, Citeseer.
    """

    def __init__(self, coord_x=None, coord_y=None, smoothness=1.5):
        self.smoothness = smoothness
        self.x_coordinates = np.reshape(coord_x, (-1, 1))
        self.y_coordinates = np.reshape(coord_y, (-1, 1))

    def get_cov_matrix(self, *, variance=1, lengthscale=1, expanded=False):
        k1 = Matern(length_scale=lengthscale, nu=self.smoothness)
        k2 = Matern(length_scale=lengthscale, nu=self.smoothness)

        # we need to split the signal variance into two parts
        K1 = np.sqrt(variance) * k1(self.x_coordinates)
        K2 = np.sqrt(variance) * k2(self.y_coordinates)


        return np.kron(K1, K2) if expanded else [K1, K2]

    def get_logpdf(self, *, variance=1, lengthscale=1, f=None):
        """
        The log-pdf of the GP
        """
        Ks = self.get_cov_matrix(variance=variance, lengthscale=lengthscale)
        log_det_K = kron_log_det(Ks)
        Ks_inv = kron_invert_matrices(Ks)

        return -0.5 * log_det_K - 0.5 * np.dot(f, kron_mv_prod(Ks_inv, f))

    def get_nabla_f(self, *, variance=None, lengthscale=None, f=None):
        """
        Gradient of the log-pdf of the GP with respect to the GP values.
        """
        Ks = self.get_cov_matrix(variance=variance, lengthscale=lengthscale)
        Ks_inv = kron_invert_matrices(Ks)
        output = -1 * kron_mv_prod(Ks_inv, f)
        return output

    def get_nabla_theta(self, *, variance=1, lengthscale=1, f=None):
        """
        Gradient of the log-pdf of the GP with respect to the hyper-parameters.
        """
        k1 = Matern(length_scale=lengthscale, nu=self.smoothness)
        k2 = Matern(length_scale=lengthscale, nu=self.smoothness)

        C1, C_grad_1 = k1(self.x_coordinates, eval_gradient=True)
        C2, C_grad_2 = k2(self.y_coordinates, eval_gradient=True)

        # we need to split the signal variance into two parts
        K1 = np.sqrt(variance) * C1
        K2 = np.sqrt(variance) * C2
        Ks = [K1, K2]
        K_invs = kron_invert_matrices(Ks)

        K1_nabla_var = (0.5 / np.sqrt(variance)) * C1
        K2_nabla_var = (0.5 / np.sqrt(variance)) * C2

        K1_nabla_l = np.sqrt(variance) * C_grad_1[:, :, 0]
        K2_nabla_l = np.sqrt(variance) * C_grad_2[:, :, 0]

        trace_comp_lengtscale = -0.5 * (
                    np.trace(np.dot(K_invs[0], K1_nabla_l)) * np.trace(np.dot(K_invs[1], Ks[1])) + np.trace(
                np.dot(K_invs[1], K2_nabla_l)) * np.trace(np.dot(K_invs[0], Ks[0])))
        trace_comp_var = -0.5 * (
                    np.trace(np.dot(K_invs[0], K1_nabla_var)) * np.trace(np.dot(K_invs[1], Ks[1])) + np.trace(
                np.dot(K_invs[1], K2_nabla_var)) * np.trace(np.dot(K_invs[0], Ks[0])))

        # non-trace component l
        temp = kron_mv_prod(K_invs, f)
        temp = kron_mv_prod([K1_nabla_l, K2], temp) + kron_mv_prod([K1, K2_nabla_l], temp)
        temp = kron_mv_prod(K_invs, temp)
        non_trace_lengthscale = 0.5 * np.dot(f, temp)

        # non-trace component var
        temp = kron_mv_prod(K_invs, f)
        temp = kron_mv_prod([K1_nabla_var, K2], temp) + kron_mv_prod([K1, K2_nabla_var], temp)
        temp = kron_mv_prod(K_invs, temp)
        non_trace_var = 0.5 * np.dot(f, temp)

        return np.asarray([trace_comp_var + non_trace_var, trace_comp_lengtscale + non_trace_lengthscale])

class GPNonGridPriorSqExp:
    """
    Gaussian process prior for an arbitrary 2D domain with zero-mean and squared exponential covariance function.
    """

    def __init__(self, coord_x=None, coord_y=None):
        self.x_coordinates = coord_x
        self.y_coordinates = coord_y
        self.cov_func = RBF

    def get_cov_matrix(self, *, variance=1, lengthscale=1):
        self.XY = np.stack((self.x_coordinates, self.y_coordinates), axis=1)
        distances = cdist(self.XY, self.XY) / lengthscale

        K = np.sqrt(variance) * np.exp(-0.5 * np.square(distances))
        return K

    def get_logpdf(self, *, variance=1, lengthscale=1, f=None):
        K = self.get_cov_matrix(variance=variance, lengthscale=lengthscale)

        sign, val = np.linalg.slogdet(K)
        log_det_K = sign * val

        K_inv = np.linalg.pinv(K)

        return -0.5 * log_det_K - 0.5 * np.dot(f, np.dot(K_inv, f))

    def get_nabla_f(self, *, variance=None, lengthscale=None, f=None):
        """
        Gradient of the log-pdf of the GP with respect to the GP values.
        """
        K = self.get_cov_matrix(variance=variance, lengthscale=lengthscale)
        K_inv = np.linalg.pinv(K)
        output = -1 * np.dot(K_inv, f)
        return output

    def get_nabla_theta(self, *, variance=1, lengthscale=1, f=None):
        """
        Gradient of the log-pdf of the GP with respect to the hyper-parameters.
        """
        K = self.get_cov_matrix(variance=variance, lengthscale=lengthscale)
        K_inv = np.linalg.pinv(K)
        K_nabla_theta = (1 / lengthscale ** 3) * K

        K_inv__time__K_nabla_theta = np.dot(K_inv, K_nabla_theta)

        part1 = 0.5 * np.dot(f, np.dot(K_inv__time__K_nabla_theta, np.dot(K_inv, f)))
        part2 = 0.5 * np.trace(K_inv__time__K_nabla_theta)
        return np.asarray([part1 - part2])


class GPNonGridPriorSqExpFixed:
    """
    Gaussian process prior for an arbitrary 2D domain with zero-mean and squared exponential covariance function, but
    with fixed hyper-parameters.
    """

    def __init__(self, coord_x=None, coord_y=None, variance=1, lengthscale=1):
        self.cov_func = RBF

        self.XY = np.stack((coord_x, coord_y), axis=1)

        distances = cdist(self.XY, self.XY) / lengthscale
        self.K = np.sqrt(variance) * np.exp(-0.5 * np.square(distances))
        self.K_inv = np.linalg.pinv(self.K)

    def get_logpdf(self, *, f=None):
        return - 0.5 * np.dot(f, np.dot(self.K_inv, f))

    def get_nabla_f(self, *, f=None):
        """
        Gradient of the log-pdf of the GP with respect to the GP values.
        """
        output = -1 * np.dot(self.K_inv, f)
        return output


class GPGridPriorSqExp:
    """
    Gaussian process prior for a 2D grid with zero-mean and squared exponential covariance function.

    Given that the domain is a grid, Kronecker algebra can be used to speed up the computations. Before reading the
    code, we recommend reading the relevant parts of PhD thesis of Yunus Saatci:

    Saatçi, Yunus. 2012. ‘Scalable Inference for Structured Gaussian Process Models’. PhD Thesis, Citeseer.
    """

    def __init__(self, coord_x=None, coord_y=None):
        self.x_coordinates = np.reshape(coord_x, (-1, 1))
        self.y_coordinates = np.reshape(coord_y, (-1, 1))
        self.cov_func = RBF

    def get_cov_matrix(self, *, variance=1, lengthscale=1, expanded=False):
        distances_X = cdist(self.x_coordinates, self.x_coordinates) / lengthscale
        K1 = np.sqrt(variance) * np.exp(-0.5 * np.square(distances_X))

        distances_Y = cdist(self.y_coordinates, self.y_coordinates) / lengthscale
        K2 = np.sqrt(variance) * np.exp(-0.5 * np.square(distances_Y))

        return np.kron(K1, K2) if expanded else [K1, K2]

    def get_logpdf(self, *, variance=1, lengthscale=1, f=None):

        Ks = self.get_cov_matrix(variance=variance, lengthscale=lengthscale)

        log_det_K = kron_log_det(Ks)
        Ks_inv = kron_invert_matrices(Ks)

        return -0.5 * log_det_K - 0.5 * np.dot(f, kron_mv_prod(Ks_inv, f))

    def get_nabla_f(self, *, variance=None, lengthscale=None, f=None):
        """
        Gradient of the log-pdf of the GP with respect to the GP values.
        """
        Ks = self.get_cov_matrix(variance=variance, lengthscale=lengthscale)
        Ks_inv = kron_invert_matrices(Ks)
        output = -1 * kron_mv_prod(Ks_inv, f)
        return output

    def get_nabla_theta(self, *, variance=1, lengthscale=1, f=None):
        """
        Gradient of the log-pdf of the GP with respect to the hyper-parameters.
        """

        distances_X = cdist(self.x_coordinates, self.x_coordinates) / lengthscale
        C1 = np.exp(-0.5 * np.square(distances_X))
        C_grad_1 = np.multiply(C1, np.square(distances_X) / lengthscale)

        distances_Y = cdist(self.y_coordinates, self.y_coordinates) / lengthscale
        C2 = np.exp(-0.5 * np.square(distances_Y))
        C_grad_2 = np.multiply(C2, np.square(distances_Y) / lengthscale)

        # we need to split the signal variance into two parts
        K1 = np.sqrt(variance) * C1
        K2 = np.sqrt(variance) * C2
        Ks = [K1, K2]
        K_invs = kron_invert_matrices(Ks)

        K1_nabla_var = (0.5 / np.sqrt(variance)) * C1
        K2_nabla_var = (0.5 / np.sqrt(variance)) * C2

        K1_nabla_l = np.sqrt(variance) * C_grad_1
        K2_nabla_l = np.sqrt(variance) * C_grad_2

        trace_comp_lengtscale = -0.5 * (
                    np.trace(np.dot(K_invs[0], K1_nabla_l)) * np.trace(np.dot(K_invs[1], Ks[1])) + np.trace(
                np.dot(K_invs[1], K2_nabla_l)) * np.trace(np.dot(K_invs[0], Ks[0])))
        trace_comp_var = -0.5 * (
                    np.trace(np.dot(K_invs[0], K1_nabla_var)) * np.trace(np.dot(K_invs[1], Ks[1])) + np.trace(
                np.dot(K_invs[1], K2_nabla_var)) * np.trace(np.dot(K_invs[0], Ks[0])))

        # non-trace component l
        temp = kron_mv_prod(K_invs, f)
        temp = kron_mv_prod([K1_nabla_l, K2], temp) + kron_mv_prod([K1, K2_nabla_l], temp)
        temp = kron_mv_prod(K_invs, temp)
        non_trace_lengthscale = 0.5 * np.dot(f, temp)

        # non-trace component var
        temp = kron_mv_prod(K_invs, f)
        temp = kron_mv_prod([K1_nabla_var, K2], temp) + kron_mv_prod([K1, K2_nabla_var], temp)
        temp = kron_mv_prod(K_invs, temp)
        non_trace_var = 0.5 * np.dot(f, temp)

        return np.asarray([trace_comp_var + non_trace_var, trace_comp_lengtscale + non_trace_lengthscale])