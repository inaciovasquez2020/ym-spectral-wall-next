import numpy as np
from src.spectral_wall.operator_nonradial import build_nonradial_operator
from src.ritz.ritz_nonradial import ritz_min_nonradial

def test_ritz_nonradial_monotone():
    d = 4
    n = 300
    ms = [6, 10, 14]
    for k in [1,2,3]:
        for ell in [1,2,3]:
            A = build_nonradial_operator(d=d, k=k, ell=ell, n=n)
            vals = [ritz_min_nonradial(A, m) for m in ms]
            for i in range(len(vals)-1):
                assert vals[i+1] <= vals[i] + 1e-8
            assert vals[-1] > 0
