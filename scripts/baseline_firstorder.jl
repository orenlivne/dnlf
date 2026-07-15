# =====================================================================================================
# Head-to-head vs the differentiable-bilevel INNER solver.
#
# Li et al. 2022 (Differentiable Bilevel Programming for Stackelberg Congestion Games; unrolled imitative
# logit dynamics) and Goyal & Lamperski 2023 (entropic mirror descent) solve the lower-level equilibrium
# with a FIRST-ORDER scheme and differentiate through it. Their released code is toy-scale Python/JAX and
# does not run on 10^4-10^5-node irregular graphs, so we reproduce the defining component — a first-order
# inner equilibrium solver — in the same environment and instances, and compare wall-clock.
#
# Fair isolation: BOTH minimize the SAME convex Beckmann energy E(phi)=Sum psi_a((B^T phi)_a - tau_a) - d^T phi,
# to the SAME relative-residual target, from the SAME cold start (zeros). Only the inner solver differs:
#   ours     = Newton + near-linear approximate Cholesky, loose-intermediate smoothing homotopy (solve_flow)
#   baseline = FISTA (accelerated proximal gradient with backtracking line search) on E — a strong, well-tuned
#              first-order method (if anything faster than plain mirror descent), so the comparison is fair.
#
# Usage:  AIER_DATA=/Users/oren/code/data julia --project=. scripts/baseline_firstorder.jl
# =====================================================================================================
using DNLF, LinearAlgebra, SparseArrays, Printf, Random
include(joinpath(@__DIR__, "scaling.jl"))                 # read_mtx_lcc, build_net, datadir

reldiff(a,b) = norm(a-b)/max(norm(b),1e-30)

# gradient of the exact Beckmann energy: ∇E(φ) = B ρ(Bᵀφ − τ) − d  (residual). f_a = ρ_a(φ_ini−φ_ter−τ_a).
function egrad!(net, φ, d, τ, f)
    @inbounds for a in 1:net.m
        f[a] = DNLF.rho(net, a, (φ[net.ini[a]] - φ[net.ter[a]]) - τ[a])[1]
    end
    (-net.B*f) .- d
end

# First-order inner solver: FISTA (accelerated proximal/projected gradient) with backtracking line search on
# the convex Beckmann energy E(φ)=DNLF.energy — the canonical strong first-order method a competent implementer
# would use (guaranteed convergence on convex L-smooth E, adaptive to the local Lipschitz constant), so the
# comparison is fair, not a strawman. Returns (φ, iters, converged, relres).
Eval(net, φ, d, τ) = DNLF.energy(net, φ, d, τ)               # exact Beckmann energy; ∇E = residual (egrad!)
function solve_firstorder(net, d; τ=zeros(net.m), rtol=1e-6, maxit=50_000_000, tbudget=300.0)
    φ = zeros(net.n); y = copy(φ); f = zeros(net.m); nb = max(norm(d), 1.0)
    t = 1.0; L = 1.0; t0 = time(); k = 0; relres = 1.0
    while k < maxit
        k += 1
        g = egrad!(net, y, d, τ, f); relres = norm(g)/nb
        relres < rtol && return (φ, k, true, relres)
        (k & 63) == 0 && time()-t0 > tbudget && return (φ, k, false, relres)
        Ey = Eval(net, y, d, τ); g2 = dot(g, g)              # backtracking: E(y-g/L) ≤ E(y) - ‖g‖²/(2L)
        φnew = similar(y)
        while true
            @. φnew = y - g/L
            Eval(net, φnew, d, τ) <= Ey - g2/(2L) && break
            L *= 2.0
            time()-t0 > tbudget && return (φ, k, false, relres)
        end
        tnew = (1 + sqrt(1 + 4t^2))/2
        @. y = φnew + ((t-1)/tnew)*(φnew - φ)
        φ = φnew; t = tnew
        L *= 0.5                                             # allow the step to grow back (standard FISTA-BT)
    end
    (φ, maxit, false, relres)
end

# ---- correctness gate: first-order must reach the SAME equilibrium as ours -------------------------
net0, d0 = build_net(read_mtx_lcc(joinpath(datadir(),"SNAP__as-735.mtx"))...)
φo,fo,_,_ = DNLF.solve_flow(net0, d0, zeros(net0.m); tol=1e-9)
φb,itb,okb,rrb = solve_firstorder(net0, d0; rtol=1e-5, maxit=5_000_000, tbudget=200.0)
fb = zeros(net0.m); egrad!(net0, φb, d0, zeros(net0.m), fb)
@printf("correctness (as-735): first-order flow vs ours reldiff=%.2e  (converged=%s, relres=%.1e, %d iters)\n\n",
        reldiff(fb, fo), okb, rrb, itb)

# ---- head-to-head timing on families + real corpus (common target rtol; DNF quantified by relres) --
 resid(net,d,f) = min(norm(net.B*f .- d), norm(net.B*f .+ d)) / norm(d)
@printf("%-22s %-8s %-9s | %-9s %-11s | %-9s %-9s %-11s %-8s\n",
        "instance","n","m","ours(s)","ours_resid","fo(s)","fo_iters","fo_relres","fo_conv")
function timepair(name, net, d)
    DNLF.solve_flow(net, d, zeros(net.m); tol=1e-9)                   # compile (accurate mode)
    local fo
    to = @elapsed ((_, fo, _, _) = DNLF.solve_flow(net, d, zeros(net.m); tol=1e-9))   # accurate: best attainable
    ro = resid(net, d, fo)
    local itc, okc, rrc
    tf = @elapsed ((_, itc, okc, rrc) = solve_firstorder(net, d; rtol=1e-6, tbudget=300.0))
    @printf("%-22s %-8d %-9d | %-9.2f %-11.1e | %-9.2f %-9d %-11.1e %-8s\n",
            name, net.n, net.m, to, ro, tf, itc, rrc, okc)
end

# synthetic irregular family (matches reproduce.jl's rand_net)
function rand_net(n; deg=5, seed=1)
    rng=MersenneTwister(seed); E=Set{Tuple{Int,Int}}()
    for u in 1:n,_ in 1:deg; v=rand(rng,1:n); v==u&&continue; push!(E,(min(u,v),max(u,v))); end
    es=collect(E); ini=Int[];ter=Int[]; for (u,v) in es; push!(ini,u);push!(ter,v);push!(ini,v);push!(ter,u); end
    m=length(ini); B=sparse([ini;ter],[1:m;1:m],[fill(-1.0,m);fill(1.0,m)],n,m)
    nt=DNLF.DirectedNetwork(n,m,ini,ter,1000 .*(0.5 .+rand(rng,m)),1 .+rand(rng,m),fill(0.15,m),fill(4.0,m),B)
    dd=zeros(n); for u in randperm(rng,n)[1:n÷8];dd[u]+=1;end; for v in randperm(rng,n)[1:n÷8];dd[v]-=1;end
    dd.-=sum(dd)/n; dd.*=(3000.0*n/(sum(abs,dd)/2)); nt,dd
end
for n in (1000, 2000, 4000)
    nt, dd = rand_net(n); timepair("synthetic-$n", nt, dd)
end
for g in ("SNAP__p2p-Gnutella08", "SNAP__Oregon-1", "SNAP__as-caida")
    p = joinpath(datadir(), g*".mtx"); isfile(p) || (println("  (missing $g)"); continue)
    net, d = build_net(read_mtx_lcc(p)...); timepair(g, net, d)
end
