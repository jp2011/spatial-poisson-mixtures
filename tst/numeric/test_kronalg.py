import numpy as np

import src.numeric.kronecker_algebra as kron

def generate_random_pos_def_matrix(shape):
    A = np.random.random(shape)
    return np.dot(A, A.transpose())  # make pos def.

def test_kron_eigenvalues():

    # Actual
    N_a, N_b, N_c = 4, 6, 3
    A = generate_random_pos_def_matrix((N_a, N_a))
    B = generate_random_pos_def_matrix((N_b, N_b))
    C = generate_random_pos_def_matrix((N_c, N_c))
    sorted_eigvals = kron.kron_eigenvalues([A, B, C])

    # Expected
    K = np.kron(A, np.kron(B, C))
    expected_eigvals, _ = np.linalg.eig(K)

    # Assert
    np.testing.assert_array_almost_equal(sorted_eigvals, np.sort(expected_eigvals))

def test_kron_determinant():
    # Actual
    A = np.asarray([[2, -1, 0],
                    [-1, 2, -1],
                    [0, -1, 2]])
    B = np.asarray([[6, 4],
                    [4, 5]])

    determinant = kron.kron_determinant([A, B])
    K = np.kron(A, B)
    expected = np.linalg.det(K)
    np.testing.assert_almost_equal(determinant, expected)

def test_kron_log_det():
    # Actual
    A = np.asarray([[2, -1, 0],
                    [-1, 2, -1],
                    [0, -1, 2]])
    B = np.asarray([[6, 4],
                    [4, 5]])

    determinant = kron.kron_log_det([A, B])
    K = np.kron(A, B)
    expected_sign, expected_val = np.linalg.slogdet(K)
    np.testing.assert_almost_equal(determinant, expected_sign * expected_val)

def test_kron_product():
    N_a, N_b, N_c = 4, 6, 3
    A = generate_random_pos_def_matrix((N_a, N_a))
    B = generate_random_pos_def_matrix((N_b, N_b))
    C = generate_random_pos_def_matrix((N_c, N_c))

    Ks = [A, B, C]
    actual = kron.kron_product(Ks)

    expected = np.kron(A, np.kron(B, C))

    np.testing.assert_array_almost_equal(actual, expected)

def test_kron_diagonal():
    N_a, N_b, N_c = 4, 6, 3

    A = generate_random_pos_def_matrix((N_a, N_a))
    B = generate_random_pos_def_matrix((N_b, N_b))
    C = generate_random_pos_def_matrix((N_c, N_c))

    Ks = [A, B, C]
    K = np.kron(A, np.kron(B, C))
    actual = kron.kron_diagonal(Ks)

    expected = np.diagonal(K)

    np.testing.assert_array_almost_equal(actual, expected)

def test_invert_matrices():
    N_a, N_b, N_c = 4, 6, 3

    A = generate_random_pos_def_matrix((N_a, N_a))
    B = generate_random_pos_def_matrix((N_b, N_b))
    C = generate_random_pos_def_matrix((N_c, N_c))

    Ks = [A, B, C]
    K = np.kron(A, np.kron(B, C))

    inverted_matrices = kron.kron_invert_matrices(Ks)
    np.testing.assert_array_almost_equal(kron.kron_product(inverted_matrices), np.linalg.inv(K), decimal=2)

def test_kron_mv_prod():
    N_a, N_b, N_c = 4, 6, 3

    A = generate_random_pos_def_matrix((N_a, N_a))
    B = generate_random_pos_def_matrix((N_b, N_b))
    C = generate_random_pos_def_matrix((N_c, N_c))

    Ks = [A, B, C]
    K = np.kron(A, np.kron(B, C))

    x = np.random.normal(loc=0, scale=1, size=N_a * N_b * N_c)
    np.testing.assert_array_almost_equal(kron.kron_mv_prod(Ks, x), np.dot(K, x), decimal=6)


def test_kron_solve():
    N_a, N_b, N_c = 4, 6, 3

    A = generate_random_pos_def_matrix((N_a, N_a))
    B = generate_random_pos_def_matrix((N_b, N_b))
    C = generate_random_pos_def_matrix((N_c, N_c))

    Ks = [A, B, C]
    K = np.kron(A, np.kron(B, C))

    expected_x = np.random.normal(0, 1, size=N_a * N_b * N_c)

    b = np.dot(K, expected_x)

    actual_x = kron.kron_cg_solve(Ks, b)
    np.testing.assert_array_almost_equal(actual_x, expected_x, decimal=6)