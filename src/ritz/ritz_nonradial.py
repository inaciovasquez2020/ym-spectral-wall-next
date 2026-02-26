import numpy as np

def ritz_min_nonradial(A, m, Qfull=None):
    """
    Ritz minimum using a nested subspace.
    Qfull: precomputed orthonormal basis (n x M), M >= m
    """
    if Qfull is None:
        raise ValueError("Qfull must be provided for nested Ritz monotonicity")

    Q = Qfull[:, :m]
    B = Q.T @ A @ Q
    return np.linalg.eigvalsh(B)[0]
