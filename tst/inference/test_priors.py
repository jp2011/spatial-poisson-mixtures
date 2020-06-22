import numpy as np
from sklearn.gaussian_process.kernels import RBF

from src.inference.priors import GPGridPriorSqExp


class TestGPGridPrior:

    def test_get_cov_matrix(self):

        lengthscale = 1.2

        x_coords = np.arange(1.0, 10.0, 1.0, dtype=np.float128)
        y_coords = np.arange(1.0, 15.0, 1.0, dtype=np.float128)

        xx, yy = np.meshgrid(x_coords, y_coords)
        X = np.stack((xx.flatten('F'), yy.flatten('F')), axis=1)

        k = RBF(length_scale=lengthscale)
        expected_K = k(X)
        # distances = cdist(X, X) / lengthscale
        # expected_K = np.exp(-0.5 * np.square(distances))

        gp_grid_prior = GPGridPriorSqExp(coord_x=x_coords, coord_y=y_coords)
        actual_K = gp_grid_prior.get_cov_matrix(variance=1, lengthscale=lengthscale, expanded=True)

        np.testing.assert_allclose(actual_K, expected_K, rtol=1e-10, atol=0)


    def test_get_logpdf(self):

        lengthscale = 1

        x_coords = np.arange(1.0, 10.0, 1.0, dtype=np.float128)
        y_coords = np.arange(1.0, 15.0, 1.0, dtype=np.float128)

        xx, yy = np.meshgrid(x_coords, y_coords)
        k = RBF(length_scale=lengthscale)

        X = np.stack((xx.flatten('F'), yy.flatten('F')), axis=1)
        K = k(X)

        L = np.linalg.cholesky(K)
        f_white = np.random.normal(0, 1, size=K.shape[0])
        # f_coloured = np.dot(L, f_white)

        f = f_white

        sign, logdet_K_abs = np.linalg.slogdet(K)
        logdet_K = sign * logdet_K_abs

        expected_log_pdf = -0.5 * logdet_K - 0.5 * np.dot(f, np.linalg.solve(K, f))
        print(f"expected det part {-0.5 * logdet_K}, \t non-det part {-0.5 * np.dot(f, np.linalg.solve(K, f))}")

        gp_grid_prior = GPGridPriorSqExp(coord_x=x_coords, coord_y=y_coords)
        actual_log_pdf = gp_grid_prior.get_logpdf(variance=1, lengthscale=lengthscale,
                                                  f=f)
        np.testing.assert_equal(actual_log_pdf, expected_log_pdf)