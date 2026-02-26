import numpy as np

def build_nonradial_operator(d, k, ell, n):
    """
    Experimental non-radial operator discretization.
    Returns symmetric matrix approximating the reduced YM operator.
    """
    r = np.linspace(1e-6, np.pi-1e-6, n)
    dr = r[1] - r[0]

    V = (ell*(ell+d-2))/np.sin(r)**2 + k
    diag = 2/dr**2 + V
    off = -1/dr**2 * np.ones(n-1)

    A = np.diag(diag) + np.diag(off,1) + np.diag(off,-1)
    return A
