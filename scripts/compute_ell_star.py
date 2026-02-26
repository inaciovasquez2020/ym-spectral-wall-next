from src.spectral_wall.operator_nonradial import build_nonradial_operator
from src.spectral_wall.ritz import ritz_min_nonradial

def is_monotone(vals, tol=1e-8):
    return all(vals[i+1] <= vals[i] + tol for i in range(len(vals)-1))

def ell_star_for_k(k, ell_grid, d=4, n=300, ms=(6,10,14,20)):
    last_ok = None
    for ell in ell_grid:
        A = build_nonradial_operator(d=d, k=k, ell=ell, n=n)
        vals = [ritz_min_nonradial(A, m) for m in ms]
        if is_monotone(vals):
            last_ok = ell
        else:
            break
    return last_ok

def main():
    d = 4
    n = 300
    ms = (6,10,14,20)
    ell_grid = list(range(1, 301, 1))

    rows = []
    for k in [1,2,3]:
        e = ell_star_for_k(k, ell_grid, d=d, n=n, ms=ms)
        rows.append((k, e if e is not None else -1))

    out = "data/ell_star.csv"
    with open(out, "w") as f:
        f.write("k,ell_star\n")
        for k,e in rows:
            f.write(f"{k},{e}\n")
    print("wrote", out)

if __name__ == "__main__":
    main()
