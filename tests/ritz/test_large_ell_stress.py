import numpy as np
from src.spectral_wall.operator_nonradial import build_nonradial_operator
from src.ritz.ritz_nonradial import ritz_min_nonradial

def test_large_ell_stress():
    d = 4
    n = 400
    ms = [8, 12, 16]
    rng = np.random.default_rng(1)

    for k in [1,2]:
        for ell in [5,10,20,40]:
            A = build_nonradial_operator(d=d, k=k, ell=ell, n=n)
            Qfull, _ = np.linalg.qr(rng.standard_normal((n, max(ms))))
            vals = [ritz_min_nonradial(A, m, Qfull) for m in ms]
            for i in range(len(vals)-1):
                assert vals[i+1] <= vals[i] + 1e-8
            assert vals[-1] > 0
