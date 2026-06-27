# DNLF — Directed Nonlinear Laplacian Flow

Extends **[NLF](https://github.com/orenlivne/nlf)** from the *undirected* Beckmann relaxation to the
**directed** user equilibrium (Wardrop) that traffic / routing practice actually solves. Built on the
NLF stack:

```
DNLF  →  NLF  →  LAMG+
```

Directed arcs carry **nonnegative** flow `f_a ≥ 0` under a **rectified** edge law — an arc carries flow
only once the potential drop exceeds its free-flow cost `t⁰_a`:

```
f_a = ρ_a(g_a),   g_a = φ_init − φ_term,
ρ_a(g) = t_a⁻¹(g)  if g ≥ t⁰_a   else 0 .
```

**The key result** (and the reason the whole NLF stack transfers): the Newton Jacobian

```
J = B diag(ρ'_a) Bᵀ
```

is **still a symmetric weighted graph Laplacian** over the active arcs — `(e_u−e_v)(e_u−e_v)ᵀ` is
orientation-blind. So NLF's chord-Newton, energy globalization, load continuation, and the LAMG+ inner
solve all carry over; the directed-specific pieces are the rectified **active set** (a conductance floor
keeps `J` connected) and the activation **regularization** (standard BPR has `t'(0)=0`, so `ρ'→∞` at
activation without it). Non-symmetry appears only one level further out — *asymmetric* (non-separable)
costs, a variational inequality — where LAMG+ becomes a GMRES preconditioner. See
[`doc/directed_ue_notes.md`](doc/directed_ue_notes.md).

## Status — research prototype

Single-commodity directed UE on the SiouxFalls network, validated against an independent Frank–Wolfe
oracle:

```
== DNLF rectified Newton ==          active arcs 29/76,  conservation 1.3e-7
   Wardrop:  max|reduced cost| USED = 8e-13,  min reduced cost UNUSED = 0 (≥0 ✓)
== vs Frank–Wolfe ==                 ||f_DNLF − f_FW|| / ||f_FW|| = 2.3e-5,  cost rel.diff 4.8e-7
VERDICT: directed UE solved on the NLF stack; matches FW; Wardrop satisfied.
```

So the extension is **feasible**. Crucially, the **value is in design optimization** — the real use
case, not the forward solve.

## Why DNLF: orders-of-magnitude-cheaper design gradients

Real network problems are *bilevel*: pick a design `x` (link **tolls**, the **OD demand** to calibrate,
**capacities** to add) to optimize an objective subject to the equilibrium `f(x)`. The load-bearing
object is the gradient `df/dx`. A SOTA equilibrium solver returns **no derivatives**, so you either
hand-build a problem-specific sensitivity method (Tobin–Friesz / Spiess) or fall back to a
derivative-free loop — **one full equilibrium re-solve per design variable**. DNLF gets the gradient
over *all* design variables from **one adjoint Laplacian solve** (the same symmetric Jacobian) — natively
and in near-linear time, because the equilibrium is `Bρ(Bᵀφ)=αd`.

[`examples/toll_design_sioux.jl`](examples/toll_design_sioux.jl) — optimal congestion pricing
(minimize total travel time over tolls). The adjoint gradient matches finite differences to FD
truncation, and optimal tolls cut total travel time **5.79%**. [`examples/toll_design_scaling.jl`](examples/toll_design_scaling.jl)
— the cost of one full design gradient:

| network | tollable links `k` | DNLF adjoint | derivative-free loop | speedup |
|---|---|---|---|---|
| SiouxFalls | 76 | 1 solve (0.00002 s) | 2·k solves (0.006 s) | **360×** |
| Anaheim | 914 | 1 solve (0.00014 s) | 2·k solves (237 s) | **~1.6 × 10⁶×** |

The adjoint is **one Laplacian solve regardless of `k`**; the loop is `O(k)` equilibrium solves — so the
speedup scales with the number of design variables (thousands of links at city scale, **millions** of OD
entries for matrix calibration). *Honest note:* both sides here use DNLF's own UE solver; a faster
bush-based solver in the loop would shrink the absolute multiplier, but the `O(k)` factor is
unavoidable, so it stays orders of magnitude. This — not the forward solve — is where DNLF earns its
keep: on planar road networks the bush-based SOTA (TAPAS / Algorithm B) owns the forward problem, but it
cannot supply the cheap exact gradients that bilevel design needs.

**Roadmap:** (1) full OD multicommodity; (2) same-instance forward validation vs tap-b to relgap 1e-8;
(3) larger design studies (Chicago / Austin, OD calibration); (4) the asymmetric-cost variational
inequality (non-symmetric inner solve, LAMG+ as preconditioner).

## Requirements

- [Julia](https://julialang.org) ≥ 1.10.
- NLF and LAMG+ are pulled automatically via `Project.toml`'s `[sources]`.

## Install

```julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quickstart

```bash
julia --project=. examples/directed_ue_sioux.jl
```

```julia
using DNLF
N = load_tntp_net("data/SiouxFalls/SiouxFalls_net.tntp")   # 24 nodes, 76 directed arcs
φ, f, steps = solve_ue(N, 1, 20, 30000.0)                  # route demand 1→20 to user equilibrium
f_fw = frank_wolfe(N, 1, 20, 30000.0)                      # independent oracle to validate against
```

## License

MIT — see [`LICENSE`](LICENSE).
