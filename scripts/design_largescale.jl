# LARGE-SCALE bilevel toll design (addresses the reviewer risk: design shown only at Anaheim scale while the
# forward solve scales to millions). Runs the projected-gradient toll-design loop (Algorithm 2) on a
# multi-million-arc irregular graph, with BOTH the forward equilibrium and the adjoint gradient computed by
# the NEAR-LINEAR engine -- the adjoint here solves J λ = B w via approximate Cholesky (NOT the direct
# J[keep,keep]\rhs in src, which is fine at Anaheim scale but is the very cubic factorization the paper argues
# against). Reports: TSTT reduction, per-design-step wall-clock, and that each step is one warm equilibrium
# re-solve + one near-linear adjoint solve. Self-caps at a wall-clock budget so it fits the allotted window.
#   Usage: AIER_DATA=~/code/data julia --project=. scripts/design_largescale.jl <graph> <budget_s>
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
import Laplacians
include(joinpath(@__DIR__, "scaling.jl"))

g       = length(ARGS) >= 1 ? ARGS[1] : "DIMACS10__coAuthorsDBLP"
budget  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 4800.0    # ~80 min default
t_start = time()

path = joinpath(datadir(), g*".mtx")
n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
dmult = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.05   # scale demand to MODERATE congestion (build_net's
d = d .* dmult                                               # default is pathologically over-capacity here)
s = argmin(d)                       # pin the largest sink (grounds the singular Laplacian)
@printf("graph=%s n=%d m=%d  budget=%.0fs\n", g, net.n, net.m, budget); flush(stdout)

_zeromean!(x) = (x .-= sum(x)/length(x); x)
function laplacian_clean(J)
    off = J - spdiagm(0 => diag(J)); off = (off + off')/2; dropzeros!(off)
    off + spdiagm(0 => -vec(sum(off, dims=2)))
end

# NEAR-LINEAR adjoint: ∇_τ TSTT via one approxChol solve of the active-subgraph Laplacian J = B diag(ρ') Bᵀ.
function adjoint_grad_nl(net, φ, f, tolls)
    ρ′ = [DNLF.rho(net, a, φ[net.ini[a]] - φ[net.ter[a]] - tolls[a])[2] for a in 1:net.m]
    mc = [DNLF.tcost(net, a, f[a]) + f[a]*DNLF.dcost(net, a, f[a]) for a in 1:net.m]
    w  = mc .* ρ′
    ρ′f = max.(ρ′, 1e-12*maximum(ρ′; init=1.0))
    Lc = laplacian_clean(net.B * spdiagm(0 => ρ′f) * net.B')
    A  = -(Lc - spdiagm(0 => diag(Lc))); dropzeros!(A)          # adjacency for approxChol
    solvefn = Laplacians.approxchol_lap(A; tol=1e-8)
    rhs = _zeromean!(-(net.B * w))
    λ = solvefn(rhs); _zeromean!(λ)
    -ρ′ .* (mc .+ (net.B' * λ)), ρ′
end

# everything imperative wrapped in a function (Julia top-level for-loops have soft-scope issues with mutated globals)
function run_design(net, d, s, budget, t_start, g)
    Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    t_cold = @elapsed ((φ, f, steps0, setups0) = DNLF.solve_flow(net, d, zeros(net.m); inner=:multigrid, Hpack=Hp,
                         itol=3e-2, inmax=6, tight_last=false, polish_refresh=0.25))
    resid0 = norm(net.B*f .+ d)/max(norm(d),1.0)
    T0 = DNLF.tstt(net, f); T = T0
    vc = [f[a]/net.cap[a] for a in 1:net.m]
    @printf("cold equilibrium: %.1fs steps=%d setups=%d resid=%.2e  TSTT0=%.4e  mean(f/cap)=%.2f max(f/cap)=%.1f active=%d\n",
            t_cold, steps0, setups0, resid0, T0, sum(vc)/net.m, maximum(vc), count(>(1e-9), f))
    flush(stdout)

    tmean = sum(net.t0)/net.m
    τ = zeros(net.m); accepted = 0; nsolve = 0
    @printf("%-5s %-9s %-9s %-6s %-11s %-8s %-8s\n","step","t_adj(s)","t_solve(s)","trials","TSTT","red%","elapsed"); flush(stdout)
    for it in 1:20
        (time() - t_start) > budget && (println("  (budget reached — stopping design loop)"); break)
        t_adj = @elapsed ((gv, _) = adjoint_grad_nl(net, φ, f, τ))
        # scale the first trial so the largest toll change is ~10% of the mean free-flow cost, regardless of
        # gradient magnitude (avoids the mis-scaled fixed-lr backtracking storm); then backtrack by halving.
        # scale the first trial to the MARGINAL-COST (Pigouvian) toll magnitude, not free-flow: the optimal
        # congestion toll is τ*_a = f_a t'_a(f_a), which under heavy congestion far exceeds t⁰. Target a toll
        # change of half the mean Pigouvian toll on the largest-gradient link; then backtrack by halving.
        mprem = sum(f[a]*DNLF.dcost(net, a, f[a]) for a in 1:net.m)/net.m
        gmax = maximum(abs, gv); step = gmax > 0 ? 0.5*mprem/gmax : 0.0
        acc = false; t_solve = 0.0; trials = 0
        for _ in 1:5
            (time() - t_start) > budget && break
            trials += 1
            τt = max.(τ .- step .* gv, 0.0)
            # warm re-solve: reuse the running hierarchy (Hp) as preconditioner (adaptive polish refreshes it
            # as needed); design needs only ~1e-3, so no tight tol. This is the near-linear per-step cost.
            dt = @elapsed ((φt, ft, _, _) = DNLF.solve_flow(net, d, τt; inner=:multigrid, Hpack=Hp,
                                init=φ, tol=1e-3, polish_refresh=0.25))
            t_solve += dt; nsolve += 1
            Tt = DNLF.tstt(net, ft)
            if Tt < T
                τ = τt; φ = φt; f = ft; T = Tt; acc = true; accepted += 1; break
            end
            step *= 0.5
        end
        @printf("%-5d %-9.1f %-9.1f %-6d %-11.4e %-8.3f %-8.0f\n", it, t_adj, t_solve, trials, T, 100*(1-T/T0), time()-t_start)
        flush(stdout)
        acc || (println("  (no further decrease — converged)"); break)
    end
    @printf("  (total equilibrium re-solves in design loop: %d)\n", nsolve)

    @printf("\nDESIGN SUMMARY: %s  m=%d\n", g, net.m)
    @printf("  TSTT %.4e -> %.4e   reduction = %.2f%%   accepted steps = %d\n", T0, T, 100*(1-T/T0), accepted)
    @printf("  total wall-clock = %.0fs (cold solve %.0fs + design loop)\n", time()-t_start, t_cold)
end

run_design(net, d, s, budget, t_start, g)
println("DONE")
