import numpy as np
import csv
from scipy.sparse.linalg import ArpackNoConvergence
from src.spectral_wall.operator_nonradial import build_nonradial_operator
from src.spectral_wall.ritz import ritz_min_nonradial

d = 4
n = 300
ms = [6,10,14,20]
ks = [1,2,3]
ells = list(range(1,401))

rows = []

for k in ks:
    for ell in ells:
        try:
            A = build_nonradial_operator(d=d, k=k, ell=ell, n=n)
            vals = [ritz_min_nonradial(A, m) for m in ms]
            monotone = all(vals[i+1] <= vals[i] + 1e-8 for i in range(len(vals)-1))
            rows.append((k, ell, "ok", monotone, min(vals)))
        except ArpackNoConvergence:
            rows.append((k, ell, "no_convergence", None, None))
        except Exception as e:
            rows.append((k, ell, f"error:{type(e).__name__}", None, None))

with open("data/nonradial_search.csv","w") as f:
    w = csv.writer(f)
    w.writerow(["k","ell","status","monotone","min_val"])
    w.writerows(rows)

print("wrote data/nonradial_search.csv")
