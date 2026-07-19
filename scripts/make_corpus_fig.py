#!/usr/bin/env python3
# Regenerate doc/dnlf_corpus.pdf from scripts/scaling_corpus.csv: log-log LAMG+ equilibrium solve time vs
# edge count across a topology-diverse SuiteSparse/SNAP corpus (AS, P2P, collaboration, email, social,
# citation), with the corpus-wide least-squares power-law fit. Demonstrates the near-linear exponent holds
# across real graph families, not just one synthetic size-scaled family.
#   Usage:  python3 scripts/make_corpus_fig.py
import csv, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
ms, ts, names = [], [], []
with open(os.path.join(here, "scaling_corpus_full_FINAL_96.csv")) as f:
    for row in csv.DictReader(f):
        ms.append(float(row["m"])); ts.append(float(row["lamg_s"])); names.append(row["graph"])
ms, ts = np.array(ms), np.array(ts)

def fit(x, y):                                       # centered log-log OLS
    lx, ly = np.log10(x), np.log10(y)
    b = np.sum((lx - lx.mean()) * (ly - ly.mean())) / np.sum((lx - lx.mean()) ** 2)
    return b, ly.mean() - b * lx.mean()
p, a = fit(ms, ts)
big = ms >= 1e5
pb, ab = fit(ms[big], ts[big])                       # large-graph subset (overhead-free) fit

# color by topology family (group prefix, with SNAP sub-categorized)
def fam(nm):
    grp, s = nm.split("__", 1) if "__" in nm else ("", nm)
    if grp == "ML_Graph":   return ("ML $k$-NN", "#8c564b")
    if grp == "LAW":        return ("web crawl", "#e377c2")
    if grp == "Newman":     return ("collaboration / AS", "#2ca02c")
    if grp == "Barabasi" or grp == "Gleich": return ("web crawl", "#e377c2")
    if grp == "DIMACS10":
        if s.startswith(("coAuthors", "citation")): return ("citation / co-author", "#17becf")
        return ("synthetic scale-free", "#bcbd22")
    # SNAP
    if s.startswith(("as-", "Oregon", "caida")): return ("AS / routing", "#1f77b4")
    if s.startswith("p2p"):            return ("P2P overlay", "#ff7f0e")
    if s.startswith("ca-"):            return ("collaboration / AS", "#2ca02c")
    if s.startswith(("email", "wiki", "sx", "College")): return ("email / Q&A / wiki", "#9467bd")
    if s.startswith(("web", "cnr")):   return ("web crawl", "#e377c2")
    if s.startswith(("soc", "loc", "cit", "com", "amazon")): return ("social / citation", "#d62728")
    return ("other", "#7f7f7f")

xs = np.logspace(np.log10(ms.min()), np.log10(ms.max()), 100)
plt.figure(figsize=(5.4, 3.7))
seen = set()
for m, t, nm in zip(ms, ts, names):
    lab, col = fam(nm)
    plt.loglog(m, t, "o", color=col, ms=6, label=lab if lab not in seen else None); seen.add(lab)
plt.loglog(xs, 10 ** a * xs ** p, "-", color="black", lw=1.3, alpha=.85,
           label=f"full corpus: $t\\propto m^{{{p:.2f}}}$")
xb = np.logspace(5, np.log10(ms.max()), 60)
plt.loglog(xb, 10 ** ab * xb ** pb, "--", color="black", lw=1.2, alpha=.6,
           label=f"$m\\geq10^5$: $t\\propto m^{{{pb:.2f}}}$")
plt.xlabel("edges $m$"); plt.ylabel("equilibrium solve time (s)")
plt.legend(fontsize=7, loc="upper left", ncol=1); plt.grid(True, which="both", ls=":", alpha=.4)
plt.tight_layout()
out = os.path.join(here, "..", "doc", "dnlf_corpus.pdf")
plt.savefig(out); print(f"wrote {os.path.abspath(out)}  ({len(ms)} graphs, m in [{int(ms.min())},{int(ms.max())}], full t~m^{p:.2f}, large t~m^{pb:.2f})")
