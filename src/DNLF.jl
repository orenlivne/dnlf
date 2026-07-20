"""
    DNLF — Directed Nonlinear Laplacian Flow

Extends NLF from the undirected Beckmann relaxation to the **directed** user equilibrium (Wardrop)
that traffic / routing practice actually solves. Built on the NLF stack: `DNLF → NLF → LAMG+`.

Directed arcs carry **nonnegative** flow `f_a ≥ 0` with a **rectified** edge law (an arc carries flow
only once the potential drop exceeds its free-flow cost `t⁰_a`):

    f_a = ρ_a(g_a),   g_a = φ_init − φ_term,
    ρ_a(g) = t_a⁻¹(g) if g ≥ t⁰_a else 0 .

Key fact (see `doc/`): the Newton Jacobian `J = B diag(ρ'_a) Bᵀ` is **still a symmetric weighted graph
Laplacian** over the active arcs, so NLF's whole stack — chord-Newton, energy globalization, load
continuation, and the LAMG+ inner solve — transfers. The directed-specific pieces are the rectified
active set (a conductance floor keeps `J` connected) and the activation regularization.
"""
module DNLF

using NLF                       # the dependency chain: DNLF → NLF → LAMG+ (NLF re-exports LAMG)
import Laplacians               # approxchol_lap — the swappable near-linear inner engine (approx-Cholesky)
import Metis                    # nested-dissection ordering for the CHOLMOD direct baseline (chol_builder)
using LinearAlgebra, SparseArrays, Printf

export DirectedNetwork, load_tntp_net, load_tntp_trips, destination_demands, solve_ue, solve_flow, solve_ue_direct, frank_wolfe, tstt,
       toll_gradient, adjoint_grad, approxchol_builder, lu_builder, chol_builder, rectified_law, ue_energy, smoothed_law,
       solve_sue, mc_flows, mc_resid, mc_jac, mc_blk_precond, mc_pcg, mc_tstt, mc_adjoint

"Directed network with separable BPR arc costs `t_a(f) = t⁰(1 + b (f/c)^p)`."
struct DirectedNetwork
    n::Int; m::Int
    ini::Vector{Int}; ter::Vector{Int}
    cap::Vector{Float64}; t0::Vector{Float64}; b::Vector{Float64}; p::Vector{Float64}
    B::SparseMatrixCSC{Float64,Int}     # incidence: col a = −e_init + e_term
end

"Parse a TNTP `_net.tntp` file into a `DirectedNetwork`."
function load_tntp_net(path)
    A = NTuple{6,Float64}[]; meta = true
    for ln in eachline(path)
        s = strip(ln)
        occursin("END OF METADATA", s) && (meta = false; continue)
        (meta || isempty(s) || startswith(s, "~")) && continue
        t = split(replace(s, ";" => "")); length(t) < 6 && continue
        push!(A, (parse(Float64,t[1]), parse(Float64,t[2]), parse(Float64,t[3]),
                  parse(Float64,t[5]), parse(Float64,t[6]), parse(Float64,t[7])))
    end
    n = Int(maximum(max(a[1], a[2]) for a in A)); m = length(A)
    ini = [Int(a[1]) for a in A]; ter = [Int(a[2]) for a in A]
    B = sparse([ini; ter], [1:m; 1:m], [fill(-1.0, m); fill(1.0, m)], n, m)
    DirectedNetwork(n, m, ini, ter, [a[3] for a in A], [a[4] for a in A],
                    [a[5] for a in A], [a[6] for a in A], B)
end

"Parse a TNTP `_trips.tntp` file into a dense `Z×Z` origin×destination OD matrix."
function load_tntp_trips(path)
    zones = 0; od = zeros(0, 0); orig = 0; meta = true
    for ln in eachline(path)
        s = strip(ln)
        if meta
            occursin("NUMBER OF ZONES", s) && (zones = parse(Int, split(s)[end]))
            occursin("END OF METADATA", s) && (meta = false; od = zeros(zones, zones))
            continue
        end
        (isempty(s) || startswith(s, "~")) && continue     # skip blank + `~` comment lines (e.g. ChicagoSketch's post-metadata Date)
        if startswith(s, "Origin")
            orig = parse(Int, split(s)[2]); continue
        end
        for pair in split(s, ';')
            p = strip(pair); isempty(p) && continue
            kv = split(p, ':'); length(kv) == 2 || continue
            od[orig, parse(Int, strip(kv[1]))] = parse(Float64, strip(kv[2]))
        end
    end
    od
end

"""
    destination_demands(od, n) -> Vector of balanced n-vectors

Destination-bundled multicommodity demands from an `Z×Z` OD matrix `od`, padded to `n` network nodes.
Commodity `j` (one per destination zone with positive incoming demand) has `dⱼ[i] = od[i,j]` at each
origin `i` and `dⱼ[j] = −Σᵢ od[i,j]` at the sink, so every commodity demand sums to zero (conservation).
"""
function destination_demands(od, n)
    Z = size(od, 1); demands = Vector{Float64}[]
    for j in 1:Z
        tot = sum(@view od[:, j]); tot <= 0 && continue
        d = zeros(n); for i in 1:Z; d[i] += od[i, j]; end; d[j] -= tot
        push!(demands, d)
    end
    demands
end

# regularized directed BPR (REG keeps t'(0)>0, bounding ρ' at activation; standard BPR has t'(0)=0)
const REG = 1e-3
tcost(N, a, f) = N.t0[a]*(1 + N.b[a]*(f/N.cap[a])^N.p[a]) + REG*N.t0[a]/N.cap[a]*f
dcost(N, a, f) = N.t0[a]*N.b[a]*N.p[a]*(f/N.cap[a])^(N.p[a]-1)/N.cap[a] + REG*N.t0[a]/N.cap[a]
function rho(N, a, g)                                # rectified inverse + derivative
    g <= N.t0[a] && return (0.0, 0.0)
    f = N.cap[a]*max((g - N.t0[a])/(N.b[a]*N.t0[a]), 0.0)^(1/N.p[a])
    for _ in 1:50
        r = tcost(N, a, f) - g; abs(r) <= 1e-13*(g + 1) && break
        f = max(f - r/dcost(N, a, f), 0.0)
    end
    (f, 1.0/dcost(N, a, f))
end
function arc_psi(N, a, g)                            # ψ_a(g)=∫_0^g ρ_a = g f − T_a(f),  f=ρ_a(g)
    g <= N.t0[a] && return 0.0
    f = rho(N, a, g)[1]
    Tf = N.t0[a]*f + N.t0[a]*N.b[a]/(N.p[a]+1)*f*(f/N.cap[a])^N.p[a] + REG*N.t0[a]/(2N.cap[a])*f^2
    g*f - Tf
end
energy(N, φ, d, τ) = sum(arc_psi(N, a, φ[N.ini[a]] - φ[N.ter[a]] - τ[a]) for a in 1:N.m) - dot(d, φ)

sp_to_sink(N, s) = begin                             # free-flow shortest-path potentials (warm start)
    φ = fill(Inf, N.n); φ[s] = 0.0
    for _ in 1:N.n, a in 1:N.m
        φ[N.ter[a]] < Inf && (φ[N.ini[a]] = min(φ[N.ini[a]], N.t0[a] + φ[N.ter[a]]))
    end
    replace(φ, Inf => maximum(filter(isfinite, φ)) * 2)
end

# ---------------- NLF reuse layer: engine, rectified law, energy ----------------
# approxChol (Kyng–Sachdeva, via Laplacians.jl) as a build_solver closure for NLF's newton_flow!:
# maps a scaled clean graph Laplacian Lc to an apply-closure rhs -> x solving Lc x = rhs.
# Interchangeable with LAMG+ (inner = :multigrid). Same wiring as the NP package.
function approxchol_builder(; tol = 1e-8)
    return Lc -> begin
        A = -(Lc - spdiagm(0 => diag(Lc)))          # adjacency = positive off-diagonals of −Lc
        dropzeros!(A)
        f = Laplacians.approxchol_lap(A; tol = tol)
        rhs -> f(rhs)
    end
end

# Frozen direct factorization as a build_solver: an LU of the node-1-pinned Laplacian, built once per
# continuation level (when `newton_flow!` would rebuild the AMG hierarchy) and reused by back-substitution
# within the level. This is the FAIR direct baseline — the same freeze-once-per-level discipline as the
# multigrid hierarchy — so a comparison isolates factorization vs near-linear cost, not per-step refactoring.
# Returns the zero-mean solution.
function lu_builder()
    return Lc -> begin
        n = size(Lc, 1); keep = 2:n
        F = lu(Lc[keep, keep])
        rhs -> begin
            x = zeros(n); x[keep] = F \ rhs[keep]; x .-= sum(x)/n; x
        end
    end
end

# STRONGER direct baseline: supernodal Cholesky (CHOLMOD) under a METIS **nested-dissection** ordering, the
# fair direct incumbent for the SPD graph-Laplacian J (vs the unsymmetric UMFPACK+COLAMD LU of `lu_builder`).
# Same freeze-once-per-level discipline: order + symbolic + numeric factor once, reuse by back-substitution.
# The node-1-pinned Laplacian Lc[keep,keep] is SPD; Metis gives the ND permutation, CHOLMOD factors with it.
function chol_builder()
    return Lc -> begin
        n = size(Lc, 1); keep = 2:n
        A = Symmetric(Lc[keep, keep])
        p = Vector{Int}(Metis.permutation(sparse(A))[1])   # METIS nested-dissection fill-reducing order
        F = cholesky(A; perm = p)
        rhs -> begin
            x = zeros(n); x[keep] = F \ rhs[keep]; x .-= sum(x)/n; x
        end
    end
end

# Rectified directed edge law as a newton_flow! callback. NLF's callback receives g = Bₙᵀx; with the
# solver incidence Bₙ = −N.B we get gₐ = φ_init − φ_term, so fₐ = ρₐ(gₐ − τₐ), dρₐ = ρ′ₐ ≥ 0.
rectified_law(N::DirectedNetwork, τ) = (f, dρ, g) -> begin
    @inbounds for a in 1:N.m
        (f[a], dρ[a]) = rho(N, a, g[a] - τ[a])
    end
end

# Convex directed-UE flow energy E(x) = Σₐ ψₐ((φ_init−φ_term)−τₐ) − bᵀx, with ∇E = Bₙ ρ(Bₙᵀx−τ) − b.
# Supplied to newton_flow! for the Armijo line search (the principled globalizer for the rectified law).
ue_energy(N::DirectedNetwork, b, τ) = x -> begin
    s = 0.0
    @inbounds for a in 1:N.m
        s += arc_psi(N, a, (x[N.ini[a]] - x[N.ter[a]]) - τ[a])
    end
    s - dot(b, x)
end

# Smoothed rectified law: the excess e = g − t⁰ is passed through δ·softplus(e/δ) (→ max(e,0) as δ→0),
# so ρ′ = sigmoid(e/δ)/t′(f) > 0 EVERYWHERE — no dead zone. This fills the inactive arcs with a small,
# smoothly-vanishing conductance, keeping the Newton Jacobian well-conditioned and slowly varying so the
# AMG hierarchy stays valid when frozen across a δ-level. As δ→0 the law → the exact rectified law.
function rho_smooth(N::DirectedNetwork, a, g, δ)
    δ <= 0 && return rho(N, a, g)
    e = g - N.t0[a]; z = e / δ
    sp    = δ * (max(z, 0.0) + log1p(exp(-abs(z))))    # δ·softplus(z) ≥ 0 (numerically stable)
    dspde = 1.0 / (1.0 + exp(-z))                      # sigmoid(z) ∈ (0,1)
    f = N.cap[a] * max(sp / (N.b[a]*N.t0[a]), 1e-40)^(1/N.p[a])
    for _ in 1:80
        r = (tcost(N, a, f) - N.t0[a]) - sp
        abs(r) <= 1e-14*(sp + 1) && break
        f = max(f - r/dcost(N, a, f), 1e-40)
    end
    (f, dspde / dcost(N, a, f))                        # (flow, ρ′ = df/dg > 0)
end
smoothed_law(N::DirectedNetwork, τ, δ) = (f, dρ, g) -> begin
    @inbounds for a in 1:N.m
        (f[a], dρ[a]) = rho_smooth(N, a, g[a] - τ[a], δ)
    end
end

# Default δ-homotopy schedule: fractions of the mean free-flow cost, geometric from smooth (δ₀=½t̄) down to
# δ_min≈8×10⁻⁹ t̄. The FINE ratio (0.85) is essential: each δ-level is then a small, well-conditioned
# perturbation, so solving every level tight (accurate mode) reaches machine precision without stalling. A
# coarse schedule (few big jumps) or solving only the last level tight leaves each tight solve far from
# converged on the ill-conditioned activation Jacobian, where the frozen chord crawls and stalls near 10⁻³.
const DFRACS = ntuple(k -> 0.5 * 0.85^(k - 1), 111)

"""
    solve_flow(N, d, tolls; inner=:approxchol, init=nothing, Hpack=nothing, ...) -> (φ, f, steps, setups)

Directed equilibrium `B ρ(Bᵀφ − τ) = d` for a balanced demand vector `d` (single commodity; multi-
source/sink allowed). **Cold start** (`init=nothing`): a **smoothing homotopy** in δ — one hierarchy build
per δ-level, frozen within (setups = O(#levels) = O(1) in graph size) — then a hard-law **polish** to the
exact rectified equilibrium (no accuracy compromise). **Warm start** (`init` given): a single hard-law
solve that **reuses the passed-in hierarchy** `Hpack` — this is what removes the per-design-step rebuild.
Every linear solve is delegated to NLF's `newton_flow!` + near-linear engine (`:approxchol` default,
`:multigrid`=LAMG+, `:direct`). Returns `(φ, f, steps, setups)`.
"""
function solve_flow(N::DirectedNetwork, d, tolls; inner = :approxchol, init = nothing,
                    tol = 1e-9, nmax = 300, anderson = 8, Hpack = nothing, dfracs = DFRACS,
                    refresh = 0.25, itol = nothing, inmax = 6,
                    polish_refresh_mode = :residual, polish_active_thresh = 0.01, polish_verbose = false,
                    homotopy_refresh = 1e9, homotopy_refresh_mode = :residual, homotopy_active_thresh = 0.01,
                    polish_refresh = nothing, tight_last = true, tlim = Inf)
    deadline = time() + tlim         # global wall-clock budget across the WHOLE solve (homotopy + polish);
                                      # Inf by default (unchanged behavior) — set finite to bound one pathological
                                      # instance instead of letting it block a batch run indefinitely.
    Bn   = -N.B                                          # (Bₙᵀx)ₐ = φ_init − φ_term  (rho's convention)
    bs   = inner === :approxchol ? approxchol_builder() :
           inner === :lu         ? lu_builder()          :
           inner === :cholmod    ? chol_builder()        : nothing
    isym = (inner === :approxchol || inner === :lu || inner === :cholmod) ? :multigrid : inner   # engine injected via build_solver
    H, SC, ST, setups, GG = Hpack === nothing ?
        (Ref{Any}(nothing), Ref(1.0), Ref(false), Ref(0), Ref(1.0)) : Hpack
    x = init === nothing ? zeros(N.n) : copy(init)
    tot = 0
    # `itol` (intermediate tolerance) enables loose continuation: intermediate δ-levels are solved only to
    # `itol` in `inmax` steps (they just warm-start the next level), tight only at the final level; the polish
    # is then frozen. This keeps the Newton-step count and setups size-independent (paper §5). `itol=nothing`
    # gives the accurate mode (every level tight, adaptive polish → machine precision).
    loose = itol !== nothing
    if init === nothing
        # Cold: smoothing homotopy. The smoothed law has ρ′>0 even at x=0 (no dead zone), so J is a
        # connected Laplacian from the start; rebuild once per δ-level and freeze within ⇒ O(1) builds.
        tmean = sum(N.t0)/N.m
        t00 = time()
        for (i, fr) in enumerate(dfracs)
            if (deadline - time()) <= 0
                polish_verbose && println("  (tlim reached during homotopy — stopping)")
                break
            end
            H[] = nothing
            # `tighten` = solve THIS level to full tol. Accurate mode: every level. Loose mode: only the
            # last level, and only if `tight_last` — otherwise keep even the last level loose and defer ALL
            # tight work to the (adaptive) polish. On large stiff graphs the last homotopy level is FROZEN
            # (homotopy_refresh), so solving it tight there stalls; deferring to the adaptive polish avoids that.
            last = (i == length(dfracs))
            tighten = !loose || (last && tight_last)
            ltol  = tighten ? tol  : itol
            lnmax = tighten ? nmax : inmax
            res = newton_flow!(x, Bn, smoothed_law(N, tolls, fr*tmean), d; inner = isym,
                     build_solver = bs, tol = ltol, nmax = lnmax, anderson = anderson,
                     refresh = homotopy_refresh, refresh_mode = homotopy_refresh_mode,
                     active_thresh = homotopy_active_thresh, tlim = deadline - time(),
                     H = H, SC = SC, ST = ST, setups = setups, GG = GG)
            tot += res.steps
            if polish_verbose
                @printf("level=%-4d/%-4d steps=%-3d resid=%.2e setups=%-4d elapsed=%.1fs\n",
                        i, length(dfracs), res.steps, res.residual, setups[], time()-t00)
                flush(stdout)
            end
        end
    end
    # Polish (cold) / warm solve: exact rectified law with Armijo energy. Accurate mode rebuilds adaptively
    # (→ machine precision); loose mode keeps the hierarchy frozen (→ ≈`itol`, "good enough" for design/scaling).
    remaining = deadline - time()
    # Polish rebuild policy: default keeps prior behavior (frozen in loose mode, `refresh` in accurate mode).
    # `polish_refresh` overrides it — e.g. loose homotopy (cheap) + adaptive-rebuild polish (accurate final
    # solve), decoupling the two so a stiff final exact-law solve isn't stuck on a stale hierarchy.
    prefresh = polish_refresh === nothing ? (loose ? 1e9 : refresh) : polish_refresh
    if remaining > 0
        res = newton_flow!(x, Bn, rectified_law(N, tolls), d; inner = isym, build_solver = bs,
                 energy = ue_energy(N, d, tolls), tol = tol, nmax = nmax, anderson = anderson,
                 refresh = prefresh, H = H, SC = SC, ST = ST, setups = setups, GG = GG,
                 refresh_mode = polish_refresh_mode, active_thresh = polish_active_thresh,
                 verbose = polish_verbose, tlim = remaining)
        tot += res.steps
    elseif polish_verbose
        println("  (tlim reached before polish — skipping)")
    end
    f = zeros(N.m)
    @inbounds for a in 1:N.m; f[a] = rho(N, a, (x[N.ini[a]] - x[N.ter[a]]) - tolls[a])[1]; end
    x, f, tot, setups[]
end

"""
    solve_ue(N, r, s, D; ...) -> (φ, f, steps)

Single-OD convenience wrapper over `solve_flow`: demand `D` from `r` to `s`. See `solve_flow` for the
homotopy / hierarchy-reuse machinery and keyword arguments.
"""
function solve_ue(N::DirectedNetwork, r, s, D; tolls = zeros(N.m), kwargs...)
    d = zeros(N.n); d[r] = D; d[s] = -D
    φ, f, steps, _ = solve_flow(N, d, tolls; kwargs...)
    φ, f, steps
end

"""
    solve_ue_direct(N, r, s, D) -> (φ, f, steps)

Reference solver: the original hand-rolled chord-Newton with a **direct** `J[keep,keep] \\ …` inner
solve (pins the sink). Kept for regression/ground-truth against the near-linear `solve_ue`.
"""
function solve_ue_direct(N::DirectedNetwork, r, s, D; tolls = zeros(N.m), init = nothing,
                  loads = 0.05:0.05:1.0, nmax = 200, tol = 1e-9)
    φ = init === nothing ? sp_to_sink(N, s) : copy(init)
    keep = setdiff(1:N.n, s)                          # pin the sink
    ls = init === nothing ? loads : (1.0,)            # warm start -> skip load continuation
    f = zeros(N.m); dρ = zeros(N.m); tot = 0
    for ℓ in ls
        d = zeros(N.n); d[r] = ℓ*D; d[s] = -ℓ*D
        for _ in 1:nmax
            tot += 1
            @inbounds for a in 1:N.m; (f[a], dρ[a]) = rho(N, a, φ[N.ini[a]] - φ[N.ter[a]] - tolls[a]); end
            R = -(N.B*f) - d                          # = grad E
            norm(R[keep]) < tol*max(ℓ*D, 1) && break
            dρf = max.(dρ, 1e-12 * maximum(dρ; init = 1.0))  # floor keeps J a connected Laplacian
            J = N.B*spdiagm(0 => dρf)*N.B'
            δ = zeros(N.n); δ[keep] = J[keep, keep] \ (-R[keep])
            E0 = energy(N, φ, d, tolls); slope = dot(R[keep], δ[keep]); α = 1.0; moved = false
            for _ in 1:60
                φt = copy(φ); φt[keep] .+= α.*δ[keep]
                if energy(N, φt, d, tolls) <= E0 + 1e-4*α*slope; φ = φt; moved = true; break; end
                α *= 0.5
            end
            moved || break
        end
    end
    @inbounds for a in 1:N.m; f[a] = rho(N, a, φ[N.ini[a]] - φ[N.ter[a]] - tolls[a])[1]; end
    φ, f, tot
end

# ---------------- design optimization: TSTT + its ADJOINT gradient w.r.t. tolls ----------------
"Total system travel time  Σ_a t_a(f_a) f_a  (tolls are transfers, excluded)."
tstt(N::DirectedNetwork, f) = sum(tcost(N, a, f[a])*f[a] for a in 1:N.m)

"∇_τ TSTT given a solved equilibrium (φ, f) — one adjoint Laplacian solve, for ALL tolls."
function adjoint_grad(N::DirectedNetwork, s, φ, f, tolls)
    keep = setdiff(1:N.n, s)
    ρ′ = [rho(N, a, φ[N.ini[a]] - φ[N.ter[a]] - tolls[a])[2] for a in 1:N.m]
    mc = [tcost(N, a, f[a]) + f[a]*dcost(N, a, f[a]) for a in 1:N.m]   # marginal cost m_a
    w  = mc .* ρ′
    J  = N.B*spdiagm(0 => max.(ρ′, 1e-12*maximum(ρ′; init = 1.0)))*N.B'   # tiny floor keeps J connected, ~no bias
    λ  = zeros(N.n); λ[keep] = J[keep, keep] \ (-(N.B*w))[keep]        # the ONE adjoint solve
    -ρ′ .* (mc .+ (N.B'*λ))                                           # ∇_τ TSTT over ALL links
end

function toll_gradient(N::DirectedNetwork, r, s, D; tolls = zeros(N.m))
    φ, f, steps = solve_ue(N, r, s, D; tolls = tolls)
    tstt(N, f), adjoint_grad(N, s, φ, f, tolls), φ, f, steps
end

"Independent Frank–Wolfe UE oracle (shortest-path / all-or-nothing) for validation."
function frank_wolfe(N::DirectedNetwork, r, s, D; iters = 4000)
    out = [Int[] for _ in 1:N.n]; for a in 1:N.m; push!(out[N.ini[a]], a); end
    dij(cost) = begin
        dist = fill(Inf, N.n); pa = zeros(Int, N.n); dist[r] = 0; done = falses(N.n)
        for _ in 1:N.n
            u = 0; best = Inf
            for v in 1:N.n; !done[v] && dist[v] < best && (best = dist[v]; u = v); end
            u == 0 && break; done[u] = true
            for a in out[u]; nd = dist[u] + cost[a]; nd < dist[N.ter[a]] && (dist[N.ter[a]] = nd; pa[N.ter[a]] = a); end
        end
        pa
    end
    aon(pa) = (y = zeros(N.m); v = s; while v != r && pa[v] != 0; y[pa[v]] += D; v = N.ini[pa[v]]; end; y)
    f = aon(dij([tcost(N, a, 0.0) for a in 1:N.m]))
    for _ in 1:iters
        cost = [tcost(N, a, f[a]) for a in 1:N.m]; y = aon(dij(cost))
        dot(cost, f - y)/max(dot(cost, f), 1e-30) < 1e-12 && break
        γ = 0.5; lo = 0.0; hi = 1.0
        for _ in 1:60
            γ = (lo+hi)/2
            sum(tcost(N, a, f[a]+γ*(y[a]-f[a]))*(y[a]-f[a]) for a in 1:N.m) > 0 ? (hi = γ) : (lo = γ)
        end
        f .+= γ.*(y .- f)
    end
    f
end

include("multicommodity.jl")

end # module
