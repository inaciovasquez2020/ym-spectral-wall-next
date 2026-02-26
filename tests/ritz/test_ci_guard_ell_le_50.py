import numpy as np
from src.spectral_wall.operator_nonradial import build_nonradial_operator
from src.spectral_wall.ritz import ritz_min_nonradial

def test_ci_guard_monotone_up_to_ell_50():
    d = 4
    n = 300
    ms = [6,10,14,20]
    for k in [1,2,3]:
        for ell in [1,2,5,10,20,50]:
            A = build_nonradial_operator(d=d, k=k, ell=ell, n=n)
            vals = [ritz_min_nonradial(A, m) for m in ms]
            for i in range(len(vals)-1):
                assert vals[i+1] <= vals[i] + 1e-8
            assert vals[-1] > 0
