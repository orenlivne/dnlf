#!/usr/bin/env python3
# Regenerate doc/dnlf_scaling.pdf from scripts/scaling_engines.csv (the clean 3-engine synthetic-scaling run).
# Log-log wall-clock vs edge count: the two near-linear engines --- LAMG+ (default) and approximate Cholesky
# (interchangeable) --- vs superlinear direct factorization, each with its least-squares power-law fit and the
# near-linear-vs-direct crossover.
#   Usage:  python3 scripts/make_scaling_fig.py
import csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
ms, lg, ac, di = [], [], [], []
with open(os.path.join(here, "scaling_engines.csv")) as f:
    for row in csv.DictReader(f):
        ms.append(float(row["m"])); lg.append(float(row["lamg_s"])); ac.append(float(row["approxchol_s"]))
        di.append(float(row["direct_s"]) if row["direct_s"] else None)
ms = np.array(ms)
lgm = np.array([(m, t) for m, t in zip(ms, lg)])
acm = np.array([(m, t) for m, t in zip(ms, ac)])
dim = np.array([(m, t) for m, t in zip(ms, di) if t is not None])

def fit(x, y):                                  # centered (numerically stable) OLS on log-log
    lx, ly = np.log10(x), np.log10(y)
    b = np.sum((lx - lx.mean()) * (ly - ly.mean())) / np.sum((lx - lx.mean()) ** 2)
    return b, ly.mean() - b * lx.mean()
plg, alg = fit(lgm[:, 0], lgm[:, 1]); pac, aac = fit(acm[:, 0], acm[:, 1]); pdi, adi = fit(dim[:, 0], dim[:, 1])
xs = np.logspace(np.log10(ms.min()), np.log10(ms.max()), 100)
mstar = 10 ** ((adi - alg) / (plg - pdi))       # LAMG+ (default near-linear) vs direct crossover

plt.figure(figsize=(5.2, 3.6))
plt.loglog(lgm[:, 0], lgm[:, 1], "o", color="#1f77b4", ms=6, label=f"LAMG+ (default)  (fit $m^{{{plg:.2f}}}$)")
plt.loglog(acm[:, 0], acm[:, 1], "^", color="#2ca02c", ms=6, label=f"approx. Cholesky  (fit $m^{{{pac:.2f}}}$)")
plt.loglog(dim[:, 0], dim[:, 1], "s", color="#d62728", ms=6, label=f"direct factorization  (fit $m^{{{pdi:.2f}}}$)")
plt.loglog(xs, 10 ** alg * xs ** plg, "-", color="#1f77b4", lw=1.2, alpha=.8)
plt.loglog(xs, 10 ** aac * xs ** pac, "-", color="#2ca02c", lw=1.0, alpha=.6)
plt.loglog(xs, 10 ** adi * xs ** pdi, "-", color="#d62728", lw=1.2, alpha=.8)
plt.axvline(mstar, color="gray", ls=":", lw=1)
plt.text(mstar * 1.05, lgm[:, 1].min() * 1.3, f"crossover\n$m\\approx{mstar/1e3:.0f}{{\\times}}10^3$",
         fontsize=8, color="gray")
plt.xlabel("edges $m$"); plt.ylabel("equilibrium solve time (s)")
plt.legend(fontsize=8, loc="upper left"); plt.grid(True, which="both", ls=":", alpha=.4)
plt.tight_layout()
out = os.path.join(here, "..", "doc", "dnlf_scaling.pdf")
plt.savefig(out); print(f"wrote {os.path.abspath(out)}  (LAMG+ m^{plg:.2f}, approxChol m^{pac:.2f}, direct m^{pdi:.2f}, m*={mstar:.2e})")
