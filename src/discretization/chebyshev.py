import numpy as np

def chebyshev_grid(n):
    k = np.arange(n)
    return np.cos(np.pi * k / (n-1))

def chebyshev_laplacian(n):
    x = chebyshev_grid(n)
    c = np.ones(n); c[0]=2; c[-1]=2
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i,j] = (c[i]/c[j]) * ((-1)**(i+j)) / (x[i]-x[j])
    D[np.diag_indices(n)] = -np.sum(D, axis=1)
    return D @ D
