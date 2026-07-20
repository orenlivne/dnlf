#!/usr/bin/env python3
# Regenerate doc/dnlf_scaling.pdf from scripts/scaling_cholmod.csv (the fresh same-run head-to-head of the
# near-linear NLF engine against BOTH direct factorizations: the unsymmetric UMFPACK/COLAMD LU and the fair,
# stronger CHOLMOD supernodal Cholesky under a METIS nested-dissection ordering for the SPD Laplacian J).
# Log-log wall-clock vs edge count, each with its least-squares power-law fit; the two direct solvers exhaust
# RAM (OOM) at the sizes where their curves stop, marked with a wall; NLF completes throughout.
#   Usage:  python3 scripts/make_scaling_fig.py
import csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
ms, nlf, umf, chol = [], [], [], []
with open(os.path.join(here, "scaling_cholmod.csv")) as f:
    for row in csv.DictReader(f):
        ms.append(float(row["m"]))
        nlf.append(float(row["nlf_ac_s"]) if row["nlf_ac_s"] else None)
        umf.append(float(row["umfpack_lu_s"]) if row["umfpack_lu_s"] else None)
        chol.append(float(row["cholmod_metis_s"]) if row["cholmod_metis_s"] else None)
ms = np.array(ms)
def pts(vals): return np.array([(m, t) for m, t in zip(ms, vals) if t is not None])
N, U, C = pts(nlf), pts(umf), pts(chol)

def fit(x, y):                                  # centered (numerically stable) OLS on log-log
    lx, ly = np.log10(x), np.log10(y)
    b = np.sum((lx - lx.mean()) * (ly - ly.mean())) / np.sum((lx - lx.mean()) ** 2)
    return b, ly.mean() - b * lx.mean()
pN, aN = fit(N[:, 0], N[:, 1]); pU, aU = fit(U[:, 0], U[:, 1]); pC, aC = fit(C[:, 0], C[:, 1])
xs = np.logspace(np.log10(ms.min()), np.log10(ms.max()), 100)
mstar = 10 ** ((aC - aN) / (pN - pC))           # NLF vs (fair) CHOLMOD crossover

plt.figure(figsize=(5.4, 3.7))                  # B/W-printable: distinguish series by marker+linestyle only
plt.loglog(N[:, 0], N[:, 1], "o", color="black", ms=6, label=f"NLF (near-linear)  ($m^{{{pN:.2f}}}$)")
plt.loglog(C[:, 0], C[:, 1], "^", color="black", ms=6, label=f"CHOLMOD$+$METIS  ($m^{{{pC:.2f}}}$)")
plt.loglog(U[:, 0], U[:, 1], "s", color="black", ms=6, label=f"UMFPACK$+$COLAMD  ($m^{{{pU:.2f}}}$)")
plt.loglog(xs, 10 ** aN * xs ** pN, "-", color="black", lw=1.3, alpha=.85)
plt.loglog(xs, 10 ** aC * xs ** pC, "--", color="black", lw=1.1, alpha=.7)
plt.loglog(xs, 10 ** aU * xs ** pU, ":", color="black", lw=1.1, alpha=.7)
# mark the OOM walls: the OOM occurs at the next size (= 2x the last completed m) for each direct solver
for P, lab in ((U, "UMFPACK\nOOM"), (C, "CHOLMOD\nOOM")):
    mwall = 2 * P[-1, 0]
    plt.axvline(mwall, color="black", ls=":", lw=1, alpha=.5)
    plt.text(mwall * 1.04, P[:, 1].min() * 1.4, lab, fontsize=7.5, color="black")
plt.xlabel("edges $m$"); plt.ylabel("equilibrium solve time (s)")
plt.legend(fontsize=8, loc="upper left"); plt.grid(True, which="both", ls=":", alpha=.4)
plt.tight_layout()
out = os.path.join(here, "..", "doc", "dnlf_scaling.pdf")
plt.savefig(out)
print(f"wrote {os.path.abspath(out)}  NLF m^{pN:.3f}, CHOLMOD m^{pC:.3f}, UMFPACK m^{pU:.3f}, NLF/CHOLMOD m*={mstar:.2e}")
