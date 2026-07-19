# Confirm the ~40s/level (vs ~6s of primitives) is line-search backtracking: wrap the smoothed law in a
# call-counter and run ONE homotopy level (inmax=6 steps, frozen hierarchy = the loose-mode default) on
# soc-Epinions1. If law is called ~6-12x, it's not backtracking; if ~100-400x, the frozen-direction line
# search is halving many times per step -- THAT is the large-graph cost, not level count or law cost.
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))

g = "SNAP__soc-Epinions1"
path = joinpath(datadir(), g*".mtx")
n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
@printf("graph=%s n=%d m=%d\n", g, net.n, net.m); flush(stdout)

Bn = -net.B
tmean = sum(net.t0)/net.m

# wrap the smoothed law with a call counter
callcount = Ref(0)
function counted_law(δ)
    base = DNLF.smoothed_law(net, zeros(net.m), δ)
    (f, dρ, gg) -> (callcount[] += 1; base(f, dρ, gg))
end

# run one level (level 1: δ = 0.5*tmean), inmax=6 steps, frozen hierarchy, anderson=8 -- exactly the
# loose-mode intermediate-level config
for δfrac in (0.5, )   # representative level
    callcount[] = 0
    x = zeros(net.n)
    H, SC, ST, setups, GG = Ref{Any}(nothing), Ref(1.0), Ref(false), Ref(0), Ref(1.0)
    t = @elapsed res = NLF.newton_flow!(x, Bn, counted_law(δfrac*tmean), d; inner=:multigrid,
                     tol=3e-2, nmax=6, anderson=8, refresh=1e9, H=H, SC=SC, ST=ST, setups=setups, GG=GG)
    @printf("level δ=%.3f: %d Newton steps, %d law calls, %d setups, %.1fs, resid=%.3e\n",
            δfrac*tmean, res.steps, callcount[], setups[], t, res.residual)
    @printf("  => law calls per Newton step = %.1f  (>>1 means line-search backtracking dominates)\n",
            callcount[] / max(res.steps,1))
    flush(stdout)
end
println("DONE")
