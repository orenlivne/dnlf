# Directed user equilibrium: extending NLF (research notes)

Working notes for a *second* paper: extending NLF from the undirected Beckmann relaxation to the
**directed** user-equilibrium (UE) that traffic / routing practice actually solves.

## 1. The undirected problem NLF solves now

Signed edge flow `f_e ∈ ℝ`, an **odd** monotone edge law, one node potential `φ`:

```
B ρ(Bᵀφ) = α d ,     f_e = ρ_e((Bᵀφ)_e) ,     ρ_e(−x) = −ρ_e(x).
```

It is the stationarity of a convex, **even** energy `E(φ) = Σ_e Ψ_e((Bᵀφ)_e) − α dᵀφ`,
`Ψ_e' = ρ_e`. The Jacobian

```
J = B diag(ρ'_e) Bᵀ ,   ρ'_e > 0
```

is a **symmetric weighted graph Laplacian** (SPD on 1⊥). That symmetry is the whole reason the inner
solve is a Laplacian solve (LAMG+ / multigrid).

## 2. The directed user equilibrium (the practical problem)

Directed arcs `a = (u→v)`, **nonnegative** arc flow `f_a ≥ 0`, separable cost `t_a(f_a)` (BPR:
`t_a(f) = t⁰_a (1 + β (f/c_a)^p)`), defined for `f_a ≥ 0` only — you cannot drive backwards on an arc.
Beckmann (1956):

```
min_{f ≥ 0,  B f = d}   Σ_a ∫₀^{f_a} t_a(x) dx .
```

KKT with node potentials `φ` (multipliers of `B f = d`) and arc multipliers `μ_a ≥ 0` for `f_a ≥ 0`:

```
t_a(f_a) − (φ_u − φ_v) − μ_a = 0,   μ_a ≥ 0,   μ_a f_a = 0.
```

So the arc flow is a **rectified** (one-sided, diode-like) law of the potential drop `x = φ_u − φ_v`:

```
f_a = ρ_a(x) = t_a⁻¹(x)  if x ≥ t⁰_a ,   else 0 .
```

Conservation is still `B ρ(Bᵀφ) = d` — same shape as NLF — but ρ is now **one-sided**, not odd.

### The "effective edge" between two nodes

A two-way road is **two** arcs `u→v` and `v→u` with independent rectified laws. The net flow as a
function of `x = φ_u − φ_v` is monotone with a **dead zone** of width `2·t⁰` around 0:

```
net(x) =  ρ_{uv}(x)        x ≥ t⁰_uv         (forward arc active)
          0                |x| < t⁰          (neither — gradient below free-flow cost)
          −ρ_{vu}(−x)      x ≤ −t⁰_vu        (reverse arc active)
```

That dead zone (from the free-flow threshold `t⁰ > 0`) is exactly what the undirected odd law throws
away (NLF's ρ has `ρ'(0) > 0`, no dead zone, allows counterflow). It is why the undirected objective
differs from the directed UE by ~100% on Sioux Falls: the relaxation permits "shortcut" counterflows
the directed problem forbids.

## 3. Three independent axes of "directed-ness"

| axis | undirected NLF | directed UE | effect |
|---|---|---|---|
| edge law | odd `ρ` | **rectified** `ρ` (`f ≥ 0`, dead zone) | active set on arcs |
| commodities | single source–sink | full **OD matrix** (up to n² pairs) | scale |
| cost coupling | separable | usually separable; *asymmetric* with junctions | symmetry of J |

These are **separable** issues — we can take them one at a time.

## 4. Key result: for separable costs the Jacobian is STILL a symmetric Laplacian

The Newton Jacobian of `B ρ(Bᵀφ) − d` is

```
J = B diag(ρ'_a) Bᵀ = Σ_a ρ'_a (e_u − e_v)(e_u − e_v)ᵀ ,   ρ'_a = 1/t'_a(f_a) ≥ 0.
```

`(e_u − e_v)(e_u − e_v)ᵀ` does **not** depend on the arc's orientation, so **J is a symmetric weighted
graph Laplacian over the currently active arcs** — exactly NLF's inner operator, restricted to the
active subgraph. **The directed problem does not break the Laplacian solver.** The directedness shows
up only as (i) which arcs are active (ρ'_a = 0 in the dead zone / on unused arcs) and (ii) the
one-sided line search. LAMG+ applies unchanged on the active subgraph.

The multicommodity OD case keeps symmetry too: with `f_a = Σ_k f_a^k` and cost a function of the total,
the block Hessian per arc is `t'_a · (1 1ᵀ)` — symmetric — exactly NLF's §multicommodity block
Laplacian, just with the rectified active set.

**Non-symmetry enters only with asymmetric (non-separable) costs** — junction/cross-link interactions
where `∂t_a/∂f_b ≠ ∂t_b/∂f_a`. Then there is no convex potential (Beckmann's line integral is
path-dependent), the problem is a **variational inequality (VI)**, and

```
J = B (∇t)⁻¹ Bᵀ ,   ∇t non-symmetric  ⇒  J non-symmetric  ⇒  not a Laplacian.
```

This is the only place the user's "non-symmetric Jacobian" worry actually bites — and it is a *further*
generalization, not the baseline directed UE.

## 5. What survives / what breaks

**Separable directed OD UE (the textbook / TAPAS problem):**
- **Newton:** yes — convex; damped chord-Newton with energy line search, exactly as NLF. The `f ≥ 0`
  active set → **semismooth / projected Newton** (the rectified ρ is piecewise smooth; ρ' jumps to 0
  at the dead-zone edge). NLF's box-projected relaxer + box-coarsening already do this for max-flow.
- **Inner solve:** **Laplacian, symmetric → LAMG+ unchanged** on the active subgraph.
- **Continuation:** BPR has **no fold** (ρ' bounded away from 0 on active arcs) → continuation is just
  load warm-starting, as in NLF's congestion case. The new "events" are **active-set changes** (arcs
  crossing the `t⁰` threshold) — a parametric active-set walk, much milder than a fold. For a
  *saturating* directed law (Kleinrock min-delay) the capacity fold returns and NLF's
  continuation+deflation applies directly.
- **New work:** (a) the rectified law + dead-zone active set; (b) **OD scale** — the real challenge.

**Asymmetric (non-separable) VI — the frontier:**
- **Newton:** Josephy–Newton / semismooth Newton on the VI; locally fine, but **no energy to descend**
  → globalize on a **merit function** (natural-residual or gap function), not NLF's energy line search.
- **Inner solve:** **non-symmetric** `J = B(∇t)⁻¹Bᵀ` → LAMG+ (SPD) no longer a black box. Options:
  GMRES/BiCGStab **preconditioned by the symmetric part's Laplacian** (use LAMG+ as the preconditioner
  — the symmetric part `½(∇t+∇tᵀ)` is still a Laplacian!), or non-symmetric/Petrov–Galerkin AMG.
- **Continuation:** homotopy/path-following for VIs is standard and should transfer.

## 6. Computational challenges, ranked

1. **OD multicommodity scale.** Up to n² commodities. The paper's §mc (K commodities, one shared
   hierarchy) is the first rung; the real problem needs commodity aggregation / bush-like
   destination-rooted bundling so cost is `O(m·polylog)` not `O(m·#OD)`. This is the dominant issue.
2. **Directed active set / dead zone.** Rectified ρ has ρ'=0 on inactive arcs → J is the active
   subgraph, which changes across the load walk. Coarsening must track an evolving active set (NLF's
   box-coarsening is the starting point, but the cut here is a *routing* boundary, not a min cut).
3. **Asymmetric case only:** non-symmetric inner solve (GMRES + LAMG+ preconditioner) and VI
   globalization (merit function).
4. **Validation target.** A directed solver we already trust: **tap-b / Algorithm B** (bush-based),
   validated to relgap 1e-12 on Sioux Falls. Compare same-instance.

## 7. First experiments (the start)

1. **Single-commodity directed UE.** Reuse NLF's chord-Newton + LAMG+ + box active set, swap the odd
   law for the rectified `ρ_a`. Solve r→s on Sioux Falls / Anaheim; check it converges and that the
   active subgraph is a sensible routing. (Tests whether the rectified active set behaves.)
2. **Validate vs directed ground truth.** Run the *full OD* separable Beckmann on Sioux Falls with
   tap-b and with a multicommodity-NLF prototype; compare link flows / objective to relgap 1e-8. This
   is the experiment the SISC paper flagged as "the clearest remaining."
3. **Scale study.** OD-aggregation strategy; cost vs #OD on Anaheim / Chicago-Sketch / Austin.
4. **(Later) asymmetric VI.** A junction-interaction instance; GMRES + LAMG+-preconditioner; measure
   whether the symmetric-part preconditioner holds.

**Bottom line.** The baseline directed UE (separable BPR) is *much* closer to NLF than the
"undirected ≠ directed" caveat suggests: **the Laplacian inner solver survives**, Newton and
continuation survive, and the genuinely new work is the rectified active set and OD scale — not the
linear algebra. Non-symmetry is real but lives one level further out (asymmetric costs), and even
there the symmetric part gives a natural LAMG+ preconditioner.

## 8. Will it be a moat? Honest competitive assessment

This is the decisive question, and the honest answer is **probably not — the undirected moat does not
replicate.** Recall *why* NLF wins in the undirected case: the moat is the conjunction of (a)
**poorly-separable** graphs (where sparse-direct fills in and first-order ill-conditions), (b) convex
**congestion** (no combinatorial competitor exists), and (c) the linear-algebra cost dominating. Strip
any leg and the moat is gone — on planar meshes/roads NLF only ties direct (it is direct's home turf).

**SOTA for directed UE is excellent and mature.** Bush/origin-based methods — OBA (Bar-Gera 2002),
**Algorithm B** (Dial 2006), **TAPAS** (Bar-Gera 2010) — reach relgap 1e-12 in *seconds* on city
networks by exploiting the acyclic per-origin bush. (We already validated tap-b: 1e-12 in 0.024 s on
Sioux Falls.) This is a combinatorial competitor, and a very good one.

**The dominant directed application — road traffic — is the wrong graph class for NLF.** Road networks
are essentially **planar and well-separated**: nested dissection / sparse-direct is near-optimal, bush
methods are tailored to them, and the conductance Laplacian is benign. This is exactly the regime where
NLF *ties* in the undirected study. So **on roads, NLF has no separability moat and faces a superb
specialized incumbent** → competitive at best, not "only solver."

**The poorly-separable directed graphs are in comms / data-center routing — but the incumbents there
are different and fast.** Internet-AS / DC-fabric / P2P topologies *are* NLF's separability turf, so the
*linear-algebra* moat could exist. But production routing there is max-flow / utility-max with
GPU/LP/ML incumbents (B4/SWAN-class, Teal, GATE, NCFlow/POP), not the sparse-direct NLF beats — and the
problem isn't the convex congestion equilibrium where NLF has no combinatorial rival. (This lane was
researched and dropped.)

**So the moat trifecta never lines up for directed:** roads have the congestion-equilibrium leg but not
poor separability or a slow incumbent; comms has poor separability but not the
no-combinatorial-rival / slow-incumbent legs.

**What IS a genuine NLF differentiator (a capability, not a speed moat).** Bush methods return *one*
equilibrium. NLF returns, for the same near-linear cost, the **full parametric flow-vs-demand curve**,
the **smooth interior flow**, the **binding-constraint (active-routing) structure**, and **sensitivity
to load** — the same "returns the curve and the dual, not just the number" argument that justifies the
max-flow instance. That is valuable precisely as an **inner solve** in problems that call UE repeatedly
and want derivatives: congestion **pricing**, **network design** (bilevel), demand-scenario sweeps,
robust/continuation analysis. There, "solve the whole family + sensitivities in one near-linear sweep"
can beat "re-run a bush solver per scenario."

**Recommendation for the second paper.** Frame it as **framework completion + a capability**, *not* a
moat: (i) the formulation result (directed UE is `B ρ(Bᵀφ)=αd` with a rectified law, and — the nice
surprise — the Jacobian stays a symmetric Laplacian, so the whole NLF stack transfers); (ii) a clean
**same-instance validation vs tap-b** to relgap 1e-8 (closes the SISC paper's "clearest remaining
experiment"); (iii) the **parametric/sensitivity** capability as the honest value-add. Do **not** repeat
the "only solver" narrative — there is no evidence it holds on the directed problem. If a moat exists at
all it is the capability differentiator on bilevel/parametric directed problems, which would need its
own validation against re-running bush solvers.
