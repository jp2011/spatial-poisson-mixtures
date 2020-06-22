from functools import reduce
import numpy as np
from numba import jit

"""
This file contains functions whic operate on matrices which can be factorised using Kronecker product. In most functions,
the factor matrices K1, K2, ... are passed in as a list.
"""


def kron_product(Ks):
    """
    Compute Kronecker product of the given list of matrices.
    :param Ks: a list of matrices
    """
    return reduce(np.kron, Ks)


def kron_inv(Ks):
    Ks_inv = kron_invert_matrices(Ks)
    return kron_product(Ks_inv)


def kron_eigenvalues(Ks):
    """
    Return a 1-D array of sorted eigenvalues of Ks
    :param Ks: a list of matrices
    """
    # assume K_d = Q_d Lambda_d Q_d^T
    # then Q = kron Q_d and Lambda = kron Lambda_d

    lambdas = []

    for i, K_d in enumerate(Ks):
        # decompose the individual matrices
        w_d, _ = np.linalg.eig(K_d)
        lambdas.append(w_d)

    def helper_function(eig_vals_A, eig_vals_B):
        return np.repeat(eig_vals_A, eig_vals_B.shape[0]) * np.tile(eig_vals_B, eig_vals_A.shape[0])

    eigenvalues = reduce(helper_function, lambdas)
    return np.sort(eigenvalues.real)


def kron_determinant(Ks):

    dets = np.asarray([np.linalg.det(K_i) for K_i in Ks])
    dims = np.asarray([K_i.shape[0] for K_i in Ks])

    determinant = 1
    for i in range(len(Ks)):
        # select exponents
        index_selector = np.ones(len(Ks), dtype=bool)
        index_selector[i] = 0
        determinant = determinant * np.power((dets[i]), np.prod(dims[index_selector]))
    return determinant


def kron_log_det(Ks):
    def log_det_compute(K_i):
        sign, val = np.linalg.slogdet(K_i)
        return sign * val

    log_dets = np.asarray([log_det_compute(K_i) for K_i in Ks])
    dims = np.asarray([K_i.shape[0] for K_i in Ks])

    log_det = 0
    for i in range(len(Ks)):
        # select exponents
        index_selector = np.ones(len(Ks), dtype=bool)
        index_selector[i] = 0

        log_det = log_det + np.prod(dims[index_selector]) * log_dets[i]

    return log_det


@jit
def kron_mv_prod(Ks, b):
    """
    Compute $Kb$ given that K = kron_product(Ks)
    :param Ks: a list of matrices
    :param b: a vector of length K.shape[1]
    """
    x = np.copy(b)
    N = len(b)
    for i, K_d in enumerate(Ks):
        G_d = K_d.shape[0]
        X = x.reshape(G_d, int(N / G_d), order='C')
        Z = np.matmul(K_d, X)
        x = Z.T.flatten('C')
    return x


def kron_diagonal(Ks):
    """
    Compute the diagonal of the matrix specified by its Kronecker components
    :param Ks: a list of matrices whose Kronecker product is equal to K
    """
    diagonals = map(np.diagonal, Ks)

    def helper_function(diag_elems_A, diag_elems_B):
        return np.repeat(diag_elems_A, diag_elems_B.shape[0]) * np.tile(diag_elems_B, diag_elems_A.shape[0])

    return reduce(helper_function, diagonals)


def kron_mm_prod(Ks, M):
    # iterate through the columns of M
    mv_prod = lambda b: kron_mv_prod(Ks, b)
    return np.apply_along_axis(mv_prod, 0, M)


def kron_invert_matrices(Ks):
    """
    Inverts each matrix in the given list and returns a list of the same size.
    :param Ks: a list of matrices
    """
    # return [np.linalg.inv(K_i) for K_i in Ks]
    return [np.linalg.pinv(K_i) for K_i in Ks]



def decompose_matrices(Ks):
    """
    Apply Cholesky decomposition to each matrix in the given list
    :param Ks: a list of matrices
    """
    Ls = []

    for i, K_d in enumerate(Ks):
        Ls.append(np.linalg.cholesky(K_d))
    return Ls


def diag_mv_prod(D, b):
    """
    Compute Db where D is a diagonal matrix (passed in as a vector of the diagonal elements)
    """
    return np.multiply(D, b)


def get_kron_dim(Ks):
    G_ds = [K_d.shape[0] for K_d in Ks]
    return np.prod(G_ds)


def kron_cg_solve(Ks, b, M=None, M_inv=None, i_max=15000, epsilon=1e-6):
    if M == None or M_inv == None:
        _M = lambda b: np.multiply(kron_diagonal(Ks), b)
        _M_inv = lambda b: np.divide(b, kron_diagonal(Ks))
    else:
        _M = M
        _M_inv = M_inv

    n = get_kron_dim(Ks)
    x = np.random.normal(0, 1, n)

    r = b - kron_mv_prod(Ks, x)
    d = _M_inv(r)
    delta_new = np.dot(r, d)
    delta_0 = delta_new

    i = 0
    while i < i_max and delta_new > (epsilon ** 2) * delta_0:
        q = kron_mv_prod(Ks, d)
        alpha = delta_new / np.dot(d, q)
        x = x + alpha * d

        if i % 50 == 0:
            r = b - kron_mv_prod(Ks, x)
        else:
            r = r - alpha * q

        s = _M_inv(r)
        delta_old = delta_new
        delta_new = np.dot(r, s)

        beta = delta_new / delta_old
        d = s + beta * d
        i = i + 1
    if i == i_max:
        print("CG not converged")
    return x






