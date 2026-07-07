# DNLF — Directed Nonlinear Laplacian Flow

A **near-linear-time solver for bilevel congestion network design**: optimize link tolls (or capacities)
over a lower-level Wardrop user equilibrium. Companion code for

> O. E. Livne, *A Near-Linear-Time Solver for Bilevel Congestion Network Design via Directed Nonlinear
> Laplacian Flow* — [`doc/dnlf_sisc.tex`](doc/dnlf_sisc.tex).

Built on the NLF nonlinear-flow solver and its near-linear inner engines:

```
DNLF  →  NLF  →  { approximate Cholesky (Laplacians.jl) | LAMG+ }
```

## What it does

The separable directed user equilibrium is a source-form **nonlinear Laplacian flow**
`B ρ(Bᵀφ − τ) = d` with a one-sided (rectified) edge law. Its Newton Jacobian `J = B diag(ρ′) Bᵀ` is a
**symmetric weighted graph Laplacian on the active subgraph** (orientation-blind rank-one terms), so every
linearized solve is a graph-Laplacian solve. The rectified dead zone makes the active set change mid-solve;
a **smoothing homotopy** in a dead-zone width δ (softplus) keeps the system well conditioned, continued from
smooth to exact with one AMG setup per level and a final polish. The design gradient over *all* `k` tolls is
**one adjoint Laplacian solve** with the same operator.

**Where it wins (honest scope):** on large **irregular, poorly-separable** networks — the scale-free
communication and overlay topologies of *selfish routing* (Internet AS graphs, P2P overlays) — the
near-linear solve (`≈ m^1.2`) overtakes a direct factorization (`≈ m^2.4`) at `m ≈ 10^5`. On **near-planar
road networks** a direct nested-dissection factorization is near-optimal and the method does **not** win;
roads are for correctness validation, not the scaling claim. See `doc/dnlf_sisc.tex` §1 and
[`doc/directed_ue_notes.md`](doc/directed_ue_notes.md).

## Install

Julia ≥ 1.10. NLF and LAMG+ are local `[sources]` path dependencies — clone them as siblings:

```
code/aier/
├── dnlf/       ← this repo
├── nlf/        ← git clone https://github.com/orenlivne/nlf
└── lamgplus/   ← git clone https://github.com/orenlivne/lamgplus
```

```julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Reproduce the paper

```bash
# unit tests — full coverage of every algorithm component (Props 1 & 3, engines, homotopy, design)
julia --project=. test/runtests.jl              #  or:  julia --project=. -e 'using Pkg; Pkg.test()'

# headline numbers + timing (Sioux Falls correctness, adjoint, 5.15% design, size-independence)
julia --project=. examples/reproduce.jl

# the irregular-graph crossover vs. a fair direct baseline (needs the SuiteSparse/SNAP corpus)
AIER_DATA=~/code/aier/data julia --project=. scripts/scaling.jl \
    SNAP__as-735 SNAP__p2p-Gnutella08 SNAP__Oregon-1 SNAP__ca-HepPh SNAP__as-caida
```

`examples/reproduce.jl` prints, on the shipped Sioux Falls network:

| check | this run | paper |
|---|---|---|
| equilibrium vs. Frank–Wolfe | `reldiff ~1e-11` | 1e-11 |
| adjoint vs. finite differences | `~2.6e-3` | 2.6e-3 |
| toll design TSTT reduction | `5.15%` | 5.15% |
| Newton steps / AMG setups | constant in size | constant |

`scripts/scaling.jl` compares the near-linear engine against a **fair** direct baseline (LU factorized once
per continuation level and reused by back-substitution — the same freeze-per-level discipline as the AMG
hierarchy, not a per-step refactor).

## Quickstart

```julia
using DNLF
N = load_tntp_net("data/SiouxFalls/SiouxFalls_net.tntp")   # 24 nodes, 76 directed arcs
φ, f, steps = solve_ue(N, 1, 20, 30000.0)                  # directed UE (near-linear homotopy solver)
g = adjoint_grad(N, 20, φ, f, zeros(N.m))                  # design gradient over all tolls, one solve
```

## Repository layout

```
src/DNLF.jl              solver: rectified + smoothed laws, homotopy solve_flow, adjoint, engines
test/runtests.jl         unit tests — every component (run: Pkg.test)
examples/reproduce.jl    reproduce the paper's headline numbers with timing
scripts/scaling.jl       irregular-graph crossover harness (approxChol vs fair direct) over the SuiteSparse/SNAP corpus
doc/dnlf_sisc.tex        the paper
doc/directed_ue_notes.md derivation notes (Laplacian structure, active set, asymmetric-cost VI)
data/                    Sioux Falls, Anaheim (TNTP)
```

## Limitations / roadmap

Single-commodity in this release; multicommodity OD, a full corpus study, differentiable-bilevel baseline
head-to-heads, and an active-set-deflation that makes one hierarchy valid across all design steps are the
stated next steps (paper §7).

## License

MIT — see [`LICENSE`](LICENSE).
