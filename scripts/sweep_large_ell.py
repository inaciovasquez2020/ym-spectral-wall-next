from src.spectral_wall.operator_nonradial import build_nonradial_operator
from src.spectral_wall.ritz import ritz_min_nonradial

d = 4
n = 300
ms = [6,10,14,20]
ells = [50,100,150,200,300,500,800,1000]

for k in [1,2,3]:
    print(f"\n=== k={k} ===")
    for ell in ells:
        A = build_nonradial_operator(d=d, k=k, ell=ell, n=n)
        vals = [ritz_min_nonradial(A, m) for m in ms]
        monotone = all(vals[i+1] <= vals[i] + 1e-8 for i in range(len(vals)-1))
        print(f"ell={ell:4d} vals={[round(v,6) for v in vals]} monotone={monotone}")
