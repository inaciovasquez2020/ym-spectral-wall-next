import numpy as np
import scipy.sparse.linalg as spla

def ritz_min_nonradial(A, m):
    """
    Compute the smallest Ritz value using an m-dimensional Krylov subspace.
    """
    vals = spla.eigsh(A, k=1, which="SA", maxiter=10_000, tol=1e-10, ncv=max(2*m, 20), return_eigenvectors=False)
    return float(vals[0])
