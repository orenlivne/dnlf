# ======================================================================================================
# Multicommodity logit stochastic user equilibrium (SUE) on the directed nonlinear Laplacian flow.
#
# K commodities (OD demands dᵏ) share the congested arc costs t_a(f_a), f_a = Σ_k fᵏ_a. With an entropic
# (logit) regularizer of weight γ (a MODEL parameter — the route-choice dispersion, fixed/calibrated), the
# equilibrium is  fᵏ_a = exp((gᵏ_a − t_a(f_a))/γ),  gᵏ_a = φᵏ_init − φᵏ_term,  B fᵏ = −dᵏ.
#
# Solved by γ-continuation damped Newton on the per-commodity potentials φ (n×K). The Newton Jacobian is the
# reduced KKT matrix J = B̃ H⁻¹ B̃ᵀ; a BLOCK-DIAGONAL preconditioner (one near-linear approximate-Cholesky
# solve per commodity) gives single-digit inner iterations near equilibrium — so each Newton step costs
# O(K) near-linear solves and the whole solve is near-linear in the total size Kn.
# ======================================================================================================

_logsumexp(v) = (M = maximum(v); M + log(sum(exp.(v .- M))))

# Total arc flow: monotone scalar root  h(f) = log f − L + (t_a(f)+τ_a)/γ = 0  (bracket + safeguarded Newton).
function mc_arc_total(N::DirectedNetwork, a, L, γ, τa = 0.0)
    h(f) = log(f) - L + (tcost(N, a, f) + τa)/γ
    lo = 1e-14; hi = 1.0
    while h(hi) < 0; hi *= 4; hi > 1e14 && break; end
    f = clamp(exp(min(L, 30.0)), lo, hi)
    for _ in 1:200
        hv = h(f); abs(hv) < 1e-13 && break
        if hv > 0; hi = f; else; lo = f; end
        hp = 1/f + dcost(N, a, f)/γ
        fn = f - hv/hp; (fn <= lo || fn >= hi) && (fn = sqrt(lo*hi)); f = fn
    end
    f
end

"Logit flow map: potentials `φ` (n×K) → (`fk` m×K per-commodity flows, `fa` m total). Stable log-sum-exp."
function mc_flows(N::DirectedNetwork, φ::AbstractMatrix, γ; tolls = nothing)
    K = size(φ, 2); m = N.m; fk = zeros(m, K); fa = zeros(m)
    @inbounds for a in 1:m
        gk = ntuple(k -> φ[N.ini[a], k] - φ[N.ter[a], k], K)
        L = _logsumexp(collect(gk) ./ γ); fa[a] = mc_arc_total(N, a, L, γ, tolls === nothing ? 0.0 : tolls[a])
        for k in 1:K; fk[a, k] = fa[a] * exp(gk[k]/γ - L); end
    end
    fk, fa
end

"Equilibrium residual Rᵏ = −(B fᵏ) − dᵏ (n×K); zero at equilibrium. `d` is a vector of K balanced demands."
mc_resid(N::DirectedNetwork, fk, d) = hcat([-(N.B * fk[:, k]) .- d[k] for k in 1:length(d)]...)

"Exact reduced-Newton Jacobian apply: dφ (n×K) → dR (n×K), from the logit-map linearization."
function mc_jac(N::DirectedNetwork, fk, fa, γ, dφ)
    K = size(fk, 2); tp = [dcost(N, a, fa[a]) for a in 1:N.m]; cinv = 1.0 ./ (γ .+ tp .* fa)
    dg = [dφ[N.ini[a], k] - dφ[N.ter[a], k] for a in 1:N.m, k in 1:K]
    dfa = [cinv[a] * sum(fk[a, k]*dg[a, k] for k in 1:K) for a in 1:N.m]
    dfk = [(fk[a, k]/γ)*dg[a, k] - (fk[a, k]*tp[a]/γ)*dfa[a] for a in 1:N.m, k in 1:K]
    hcat([-(N.B * dfk[:, k]) for k in 1:K]...)
end

# approxChol apply for a weighted graph Laplacian (zero-mean solution). Laplacians.jl's greedy elimination
# can intermittently throw on extreme-contrast per-commodity Laplacians; fall back to the degree ordering.
function _ac(L)
    A = -(L - spdiagm(0 => diag(L))); dropzeros!(A)
    for p in (nothing, Laplacians.ApproxCholParams(:deg))
        try
            f = p === nothing ? Laplacians.approxchol_lap(A; tol = 1e-10) :
                                Laplacians.approxchol_lap(A; tol = 1e-10, params = p)
            return r -> (x = f(r .- sum(r)/length(r)); x .- sum(x)/length(x))
        catch
        end
    end
    # robust fallback (pathological block that trips Laplacians.jl): pinned sparse LDLᵀ with a tiny
    # Tikhonov shift so a near-singular block never yields NaN; zero-mean solution.
    n = size(L, 1); keep = 2:n; F = ldlt(Symmetric(L[keep, keep] + (1e-10*maximum(diag(L)))*I))
    r -> begin x = zeros(n); x[keep] = F \ (r[keep] .- sum(r)/n); x .-= sum(x)/n; x end
end

"Block-diagonal preconditioner: one per-commodity weighted-Laplacian approxChol solve per block."
function mc_blk_precond(N::DirectedNetwork, fk, fa, γ)
    K = size(fk, 2); tp = [dcost(N, a, fa[a]) for a in 1:N.m]; ci = tp ./ (1 .+ tp .* fa ./ γ)
    W = (fk ./ γ) .- (ci ./ γ^2) .* (fk .^ 2)                 # (G_a)_kk > 0, per-commodity conductances
    # relative conductance floor keeps each commodity's Laplacian well-conditioned & connected (a commodity
    # barely uses most arcs → tiny W there); floor bounds the contrast so approxChol stays robust.
    acs = [_ac(N.B * spdiagm(0 => max.(W[:, k], 1e-6 * maximum(W[:, k]))) * N.B') for k in 1:K]
    r -> hcat([acs[k](r[:, k]) for k in 1:K]...)
end

"Preconditioned CG for J dφ = rhs (n×K). Returns (dφ, iters)."
function mc_pcg(N, fk, fa, γ, rhs, Minv; tol = 1e-7, maxit = 300)
    nb = norm(rhs); x = zeros(size(rhs)); r = rhs .- mc_jac(N, fk, fa, γ, x)
    z = Minv(r); p = copy(z); rz = sum(r .* z); its = maxit
    for it in 1:maxit
        Jp = mc_jac(N, fk, fa, γ, p); a = rz/sum(p .* Jp); x .+= a .* p; r .-= a .* Jp
        if norm(r)/nb < tol; its = it; break; end
        z = Minv(r); rz2 = sum(r .* z); p .= z .+ (rz2/rz) .* p; rz = rz2
    end
    x, its
end

"""
    solve_sue(N, d, γ; γ0=2.0, tol=1e-8, verbose=false) -> (φ, fk, fa, inner_iters)

Multicommodity logit SUE at dispersion `γ` (fixed model parameter) for K balanced demands `d`.
γ-continuation damped Newton from `γ0` down to `γ`, block-diagonal-preconditioned inner solve.
Returns per-commodity potentials, per-commodity + total arc flows, and the last inner-iteration count.
"""
function solve_sue(N::DirectedNetwork, d, γ; γ0 = 2.0, tol = 1e-8, tolls = nothing, verbose = false)
    K = length(d); φ = zeros(N.n, K); dn = max(norm(hcat(d...)), 1.0)
    sched = Float64[]; g = γ0; while g > γ*1.001; push!(sched, g); g = max(γ, g/2); end; push!(sched, γ)
    lastits = 0
    for gk in sched
        for _ in 1:60
            fk, fa = mc_flows(N, φ, gk; tolls = tolls); R = mc_resid(N, fk, d); rn = norm(R)/dn
            rn < tol && break
            dφ, its = mc_pcg(N, fk, fa, gk, -R, mc_blk_precond(N, fk, fa, gk)); lastits = its
            α = 1.0
            for _ in 1:30
                fk2, _ = mc_flows(N, φ .+ α .* dφ, gk; tolls = tolls)
                norm(mc_resid(N, fk2, d))/dn < rn && break
                α /= 2
            end
            φ .+= α .* dφ
        end
        if verbose
            fkv, _ = mc_flows(N, φ, gk; tolls = tolls)
            @printf("  γ=%.3g  resid=%.1e  inner=%d\n", gk, norm(mc_resid(N, fkv, d))/dn, lastits)
        end
    end
    fk, fa = mc_flows(N, φ, γ; tolls = tolls); φ, fk, fa, lastits
end

"Total system travel time on the aggregate flow (tolls are transfers, excluded)."
mc_tstt(N::DirectedNetwork, fa) = sum(tcost(N, a, fa[a]) * fa[a] for a in 1:N.m)

"""
    mc_adjoint(N, fk, fa, γ) -> ∇_τ TSTT  (length m)

Design gradient of TSTT w.r.t. per-arc tolls at a solved multicommodity equilibrium `(fk, fa)`.
One block-preconditioned adjoint solve `J λ = ∂TSTT/∂φ` (J symmetric) yields the gradient for ALL tolls:
tolls shift the arc cost by τ_a, the flow sensitivity is `∂fᵏ_a/∂τ_a = −fᵏ_a c̃_a`, `c̃_a = 1/(γ+t'_a f_a)`.
"""
function mc_adjoint(N::DirectedNetwork, fk, fa, γ)
    K = size(fk, 2); tp = [dcost(N, a, fa[a]) for a in 1:N.m]; ctil = 1.0 ./ (γ .+ tp .* fa)
    mcost = [tcost(N, a, fa[a]) + tp[a]*fa[a] for a in 1:N.m]            # ∂TSTT/∂f_a (marginal cost)
    gφ = hcat([-(N.B) * (mcost .* ctil .* fk[:, k]) for k in 1:K]...)   # ∂TSTT/∂φᵏ
    λ, _ = mc_pcg(N, fk, fa, γ, gφ, mc_blk_precond(N, fk, fa, γ); tol = 1e-10)
    grad = -(mcost .* ctil .* fa)                                       # ∂TSTT/∂τ (explicit)
    for k in 1:K; grad .-= (N.B' * λ[:, k]) .* fk[:, k] .* ctil; end    # − λᵀ ∂R/∂τ
    grad
end
