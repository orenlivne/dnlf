# FULL bilevel design at scale on a mid-size real communication network (p2p overlay -- the paper's canonical
# selfish-overlay-routing example), significantly larger than Anaheim (914 links). Two stages:
#  (1) CALIBRATE demand to MODERATE congestion via a fast cold-solve-only sweep (no stiff Pigouvian re-solve);
#      pick the multiplier whose peak volume/capacity lands in a moderate, Anaheim-like band.
#  (2) run the FULL projected-gradient toll-design LOOP (Alg. 2) to convergence: multiple accepted steps, each
#      one near-linear adjoint (all m gradients) + one warm equilibrium re-solve; report the TSTT reduction,
#      per-step cost, accepted steps, and total wall-clock -- the "design at scale" result the referees asked for.
#   Usage: AIER_DATA=~/code/data julia --project=. scripts/design_scale.jl <graph> <budget_s>
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
import Laplacians
include(joinpath(@__DIR__, "scaling.jl"))

g      = length(ARGS) >= 1 ? ARGS[1] : "SNAP__p2p-Gnutella31"
budget = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 2400.0
t0     = time()
path = joinpath(datadir(), g*".mtx")
n, edges = read_mtx_lcc(path); net, d0 = build_net(n, edges)
s = argmin(d0)
@printf("graph=%s n=%d m=%d\n", g, net.n, net.m); flush(stdout)

_zeromean!(x) = (x .-= sum(x)/length(x); x)
function laplacian_clean(J)
    off = J - spdiagm(0 => diag(J)); off = (off + off')/2; dropzeros!(off)
    off + spdiagm(0 => -vec(sum(off, dims=2)))
end
function coldsolve(d)
    Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    _, f, st, su = DNLF.solve_flow(net, d, zeros(net.m); inner=:multigrid, Hpack=Hp,
                                    itol=3e-2, inmax=6, tight_last=false, polish_refresh=0.25, tlim=180.0)
    f, st, su
end
adj(net, φ, f, tolls) = begin
    ρ′ = [DNLF.rho(net, a, φ[net.ini[a]] - φ[net.ter[a]] - tolls[a])[2] for a in 1:net.m]
    mc = [DNLF.tcost(net, a, f[a]) + f[a]*DNLF.dcost(net, a, f[a]) for a in 1:net.m]
    w  = mc .* ρ′; ρ′f = max.(ρ′, 1e-12*maximum(ρ′; init=1.0))
    Lc = laplacian_clean(net.B * spdiagm(0 => ρ′f) * net.B')
    A  = -(Lc - spdiagm(0 => diag(Lc))); dropzeros!(A)
    sv = Laplacians.approxchol_lap(A; tol=1e-8)
    λ = sv(_zeromean!(-(net.B*w))); _zeromean!(λ)
    -ρ′ .* (mc .+ (net.B'*λ))
end

function main(net, d0, budget, t0, g, s)
    # (1) calibration: sweep a WIDE demand range (graph-dependent congestion) and pick moderate peak V/C
    println("--- calibration (cold solve, peak V/C) ---"); flush(stdout)
    chosen = 0.0; best = 1e18
    for mult in (1e-3, 1e-2, 1e-1, 1.0)
        f, _, _ = coldsolve(d0 .* mult)
        mx = maximum(f[a]/net.cap[a] for a in 1:net.m); me = sum(f[a]/net.cap[a] for a in 1:net.m)/net.m
        @printf("  mult=%.1e  mean V/C=%.3f  max V/C=%.2f\n", mult, me, mx); flush(stdout)
        score = (1.5 <= mx <= 50) ? abs(log(mx/4)) : 1e9      # target moderate peak V/C ~4
        if score < best; best = score; chosen = mult; end
    end
    chosen == 0.0 && (chosen = 1.0)
    @printf("chosen demand multiplier = %.1e\n", chosen); flush(stdout)

    # (2) full design loop
    d = d0 .* chosen
    Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    tc = @elapsed ((φ, f, st0, su0) = DNLF.solve_flow(net, d, zeros(net.m); inner=:multigrid, Hpack=Hp,
                                                       itol=3e-2, inmax=6, tight_last=false, polish_refresh=0.25))
    T0 = DNLF.tstt(net, f); T = T0; τ = zeros(net.m)
    @printf("\ncold equilibrium: %.1fs steps=%d setups=%d  TSTT0=%.4e\n", tc, st0, su0, T0)
    @printf("%-5s %-8s %-9s %-6s %-11s %-8s %-8s\n","step","t_adj","t_solve","trials","TSTT","red%","elapsed"); flush(stdout)
    accepted = 0
    for it in 1:20
        (time()-t0) > budget && (println("  (budget reached)"); break)
        ta = @elapsed (gv = adj(net, φ, f, τ))
        mprem = sum(f[a]*DNLF.dcost(net,a,f[a]) for a in 1:net.m)/net.m
        gmax = maximum(abs, gv); step = gmax>0 ? 0.5*mprem/gmax : 0.0
        acc = false; ts = 0.0; tr = 0
        for _ in 1:6
            tr += 1; τt = max.(τ .- step.*gv, 0.0)
            dt = @elapsed ((φt, ft, _, _) = DNLF.solve_flow(net, d, τt; inner=:multigrid, Hpack=Hp,
                                init=φ, tol=1e-3, polish_refresh=0.25, tlim=200.0))
            ts += dt; Tt = DNLF.tstt(net, ft)
            if Tt < T; τ=τt; φ=φt; f=ft; T=Tt; acc=true; accepted+=1; break; end
            step *= 0.5
        end
        @printf("%-5d %-8.2f %-9.1f %-6d %-11.4e %-8.3f %-8.0f\n", it, ta, ts, tr, T, 100*(1-T/T0), time()-t0); flush(stdout)
        acc || (println("  (converged)"); break)
    end
    @printf("\nDESIGN-AT-SCALE: %s  n=%d m=%d  mult=%.1e\n", g, net.n, net.m, chosen)
    @printf("  TSTT %.4e -> %.4e   reduction=%.2f%%   accepted steps=%d   total=%.0fs\n",
            T0, T, 100*(1-T/T0), accepted, time()-t0)
end

main(net, d0, budget, t0, g, s)
println("DONE")
