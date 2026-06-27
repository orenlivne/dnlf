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

So the extension is **feasible**. Honest scope (see notes §8): this is **framework completion + a
capability** (NLF returns the full flow-vs-demand curve, sensitivities, and binding-routing structure,
not just one equilibrium) — **not** a speed moat. On road networks the SOTA (bush-based TAPAS /
Algorithm B) is excellent and roads are planar (direct's home turf), so the "only solver" result of the
undirected paper does **not** replicate here.

**Roadmap:** (1) full OD multicommodity; (2) same-instance validation vs tap-b to relgap 1e-8;
(3) scale study; (4) the asymmetric-cost variational inequality (non-symmetric inner solve).

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
