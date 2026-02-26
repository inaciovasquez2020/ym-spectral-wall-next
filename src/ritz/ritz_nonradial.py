import numpy as np

def ritz_min_nonradial(A, m):
    Q, _ = np.linalg.qr(np.random.randn(A.shape[0], m))
    B = Q.T @ A @ Q
    return np.linalg.eigvalsh(B)[0]
