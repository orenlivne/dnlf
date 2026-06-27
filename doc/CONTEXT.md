# DNLF — continuation context (resume here)

Status snapshot so this thread can be picked up cold. Companion to
[`directed_ue_notes.md`](directed_ue_notes.md) (the math + competitive analysis).

## What DNLF is
Directed Nonlinear Laplacian Flow: extends **NLF** (undirected Beckmann relaxation) to the **directed
user equilibrium** (Wardrop) that traffic/routing practice actually solves. Repo:
`github.com/orenlivne/dnlf`, dependency chain **DNLF → NLF → LAMG+** (via `[sources]` in `Project.toml`).

## Core result (proven, two parts)
1. **The Jacobian stays a symmetric Laplacian** for separable costs (directed arcs / rectified law don't
   break it), so NLF's whole stack — chord-Newton, energy globalization, load continuation, LAMG+ inner
   solve — transfers. Non-symmetry only appears with *asymmetric* (non-separable) costs (a VI), where
   LAMG+ becomes a GMRES preconditioner.
2. **The value is design optimization, not the forward solve.** The equilibrium gradient comes from ONE
   adjoint Laplacian solve (same Jacobian), for all design variables at once → **O(k)** advantage over
   the derivative-free loop a SOTA solver forces (k equilibrium re-solves).

## What's built (`src/DNLF.jl`)
- `DirectedNetwork`, `load_tntp_net(path)` — parse TNTP `_net.tntp`.
- `solve_ue(N, r, s, D; tolls, init, ...)` — single-commodity directed UE: rectified-law chord-Newton +
  **energy (Armijo) line search** + **load continuation**; `init` warm-starts (skips continuation).
- `frank_wolfe(N, r, s, D)` — independent UE oracle (validation).
- `tstt(N, f)`, `adjoint_grad(N, s, φ, f, tolls)` (the one adjoint solve), `toll_gradient(...)` (UE + adjoint).

## Experiments (all run with `julia --project=<env> examples/<x>.jl`)
- `directed_ue_sioux.jl` — forward UE on SiouxFalls vs Frank–Wolfe: **rel flow 2.3e-5, Wardrop satisfied,
  232 Newton steps, 29/76 active arcs**. ✓
- `toll_design_sioux.jl` — optimal congestion pricing: adjoint grad matches FD (6.4e-3); **optimal tolls
  cut TSTT 5.79%**; per-gradient = 1 solve vs 152 (=2k) UE solves.
- `toll_design_scaling.jl` — speedup scaling: **SiouxFalls (k=76) 360× → Anaheim (k=914) ~1.6×10⁶×**
  (one adjoint solve vs 2k UE re-solves). Demonstrates O(k).

## How to run (env)
This session used `julia --project=/tmp/nlfload` (an ephemeral env with NLF+LAMG+DNLF dev-linked).
For a fresh machine, either instantiate the repo (pulls NLF+LAMG from GitHub):
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
or, for fast local iteration, dev-link the sibling repos:
```bash
julia --project=/tmp/dnlfenv -e 'using Pkg; Pkg.develop(path="."); Pkg.develop(path="../nlf"); Pkg.develop(path="../mg/lamgplus")'
```

## Implementation notes / gotchas (load-bearing)
- **Regularized BPR** `t_a(f)=t⁰(1+b(f/c)^p)+REG·t⁰/c·f`, `REG=1e-3`: standard BPR has `t'(0)=0` ⇒ `ρ'→∞`
  at activation; the linear term bounds it. Directed-specific.
- **Conductance floor** in the Newton Jacobian (`max(ρ', 1e-6·max ρ')`): the dead zone (inactive arcs)
  disconnects the active subgraph ⇒ singular `J`; the floor keeps it a connected Laplacian. Does NOT move
  the fixed point (residual uses the true rectified flow).
- **Energy line search is essential** — residual-norm line search stalls on the stiff activation.
- Directed UE needs **more Newton steps** than undirected (stiffer activation): ~232 vs 2–4.

## Honest competitive position
- **Forward UE: no moat.** Bush-based TAPAS / Algorithm B (tap-b: 1e-12 in 0.024 s on SiouxFalls) own
  static UE on planar roads (direct's home turf). The undirected "only solver" result does NOT replicate.
- **Design optimization: the real value.** O(k)-cheaper exact gradients vs derivative-free; native +
  near-linear vs hand-built sensitivity (Tobin–Friesz / Spiess). Wins where no off-the-shelf gradient
  exists or the sensitivity linear solve is the bottleneck (large / poorly-separable / non-standard
  objective). Buyers: tolling authorities, traffic consultancies, OD-calibration in PTV/Emme/Aimsun.

## Roadmap (next rungs)
1. **Full OD multicommodity** — the key next step: loads the whole network so `k` is the real toll/OD
   dimension; extends NLF's §mc block-Laplacian + the rectified active set. Makes the design result
   representative (and a publishable/saleable claim).
2. **Same-instance forward validation vs tap-b** to relgap 1e-8 (closes NLF paper's "clearest remaining").
3. **Larger design studies** — Chicago-Sketch / Austin tolling; OD-matrix calibration (k = millions).
4. **Asymmetric-cost VI** — non-symmetric inner solve, LAMG+ as preconditioner; the genuinely new
   linear algebra.

## Relation to the NLF paper
Undirected NLF is being finished/released first; the directed + design-optimization story is its
stepping-stone future-work (a one-paragraph note added to NLF's Extensions section).
