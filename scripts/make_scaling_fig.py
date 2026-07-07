#!/usr/bin/env python3
# Regenerate doc/dnlf_scaling.pdf from scripts/scaling_points.csv (the committed synthetic-scaling output).
# Log-log wall-clock vs edge count: near-linear approximate Cholesky vs superlinear direct factorization,
# each with its least-squares power-law fit and the empirical crossover. Direct points at n=32k/64k, which
# the main run capped, are passed on the command line from scripts/direct_tail.jl.
#   Usage:  python3 scripts/make_scaling_fig.py [direct_320k_s direct_640k_s]
import csv, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
ms, ac, di = [], [], []
with open(os.path.join(here, "scaling_points.csv")) as f:
    for row in csv.DictReader(f):
        ms.append(float(row["m"])); ac.append(float(row["approxchol_s"]))
        di.append(float(row["direct_s"]) if row["direct_s"] else None)
ms = np.array(ms)
# optional direct tail (32k, 64k) from direct_tail.jl
tail = [float(x) for x in sys.argv[1:3]] if len(sys.argv) >= 3 else []
di_full = list(di)
for i, v in enumerate([j for j, d in enumerate(di) if d is None]):
    if i < len(tail): di_full[v] = tail[i]

acm = np.array([(m, t) for m, t in zip(ms, ac)])
dim = np.array([(m, t) for m, t in zip(ms, di_full) if t is not None])

def fit(x, y):
    # centered (numerically stable) OLS on log-log; the uncentered normal-equation form
    # (n*Sxx-(Sx)^2, as np.polyfit uses) suffers catastrophic cancellation on these x-ranges.
    lx, ly = np.log10(x), np.log10(y)
    b = np.sum((lx - lx.mean()) * (ly - ly.mean())) / np.sum((lx - lx.mean()) ** 2)
    return b, ly.mean() - b * lx.mean()
pac, aac = fit(acm[:, 0], acm[:, 1]); pdi, adi = fit(dim[:, 0], dim[:, 1])
xs = np.logspace(np.log10(ms.min()), np.log10(ms.max()), 100)
mstar = 10 ** ((adi - aac) / (pac - pdi))

plt.figure(figsize=(5.2, 3.6))
plt.loglog(acm[:, 0], acm[:, 1], "o", color="#1f77b4", ms=6, label=f"approx. Cholesky  (fit $m^{{{pac:.2f}}}$)")
plt.loglog(dim[:, 0], dim[:, 1], "s", color="#d62728", ms=6, label=f"direct factorization  (fit $m^{{{pdi:.2f}}}$)")
plt.loglog(xs, 10 ** aac * xs ** pac, "-", color="#1f77b4", lw=1.2, alpha=.8)
plt.loglog(xs, 10 ** adi * xs ** pdi, "-", color="#d62728", lw=1.2, alpha=.8)
plt.axvline(mstar, color="gray", ls=":", lw=1)
plt.text(mstar * 1.05, acm[:, 1].min() * 1.3, f"crossover\n$m\\approx{mstar/1e3:.0f}{{\\times}}10^3$",
         fontsize=8, color="gray")
plt.xlabel("edges $m$"); plt.ylabel("equilibrium solve time (s)")
plt.legend(fontsize=8, loc="upper left"); plt.grid(True, which="both", ls=":", alpha=.4)
plt.tight_layout()
out = os.path.join(here, "..", "doc", "dnlf_scaling.pdf")
plt.savefig(out); print(f"wrote {os.path.abspath(out)}  (approxChol m^{pac:.2f}, direct m^{pdi:.2f}, m*={mstar:.2e})")
