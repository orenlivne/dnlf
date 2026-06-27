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
using LinearAlgebra, SparseArrays, Printf

export DirectedNetwork, load_tntp_net, solve_ue, frank_wolfe, tstt, toll_gradient, adjoint_grad

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

"""
    solve_ue(N, r, s, D) -> (φ, f, steps)

Single-commodity directed UE: route demand `D` from `r` to `s` by NLF-style damped chord-Newton on
the rectified edge law, with an energy (Armijo) line search and load continuation. Returns node
potentials `φ`, arc flows `f ≥ 0`, and total Newton steps.
"""
function solve_ue(N::DirectedNetwork, r, s, D; tolls = zeros(N.m), init = nothing,
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
            dρf = max.(dρ, 1e-6 * maximum(dρ; init = 1.0))   # floor keeps J a connected Laplacian
            J = N.B*spdiagm(0 => dρf)*N.B'
            δ = zeros(N.n); δ[keep] = J[keep, keep] \ (-R[keep])     # (LAMG+ is the inner solve at scale)
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
    J  = N.B*spdiagm(0 => max.(ρ′, 1e-6*maximum(ρ′; init = 1.0)))*N.B'
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

end # module
