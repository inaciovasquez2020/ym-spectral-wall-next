import numpy as np
from src.spectral_wall.operator_nonradial import build_nonradial_operator

def test_nonradial_quadratic_form_positive():
    d = 4
    n = 400
    for k in [1,2,3,4]:
        for ell in [1,2,3,4]:
            A = build_nonradial_operator(d=d, k=k, ell=ell, n=n)
            x = np.random.randn(n)
            q = x @ (A @ x)
            assert q > 0
