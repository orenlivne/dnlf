# Second isolation: with eta ruled out (identical stall at eta=0.05..1e-4), test whether rebuilding the
# AMG hierarchy EVERY step (refresh=0, true Newton, no freezing at all) converges cleanly on level 1 of
# wb-cs-stanford. If yes, this confirms the mechanism is a stale/frozen linearization (not inner-solve
# inaccuracy, not a coding bug) — a real modified-Newton stagnation specific to this graph's conditioning.
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))

g = "Gleich__wb-cs-stanford"
path = joinpath(datadir(), g*".mtx")
n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
tmean = sum(net.t0)/net.m
fr = DNLF.DFRACS[1]
Bn = -net.B
law = DNLF.smoothed_law(net, zeros(net.m), fr*tmean)
@printf("graph=%s n=%d m=%d level1_delta_frac=%.3f\n", g, net.n, net.m, fr); flush(stdout)

function trial(label, refresh; anderson=8)
    x = zeros(net.n)
    H, SC, ST, setups, GG = Ref{Any}(nothing), Ref(1.0), Ref(false), Ref(0), Ref(1.0)
    t = @elapsed res = NLF.newton_flow!(x, Bn, law, d; inner=:multigrid, tol=1e-9, nmax=300,
                     anderson=anderson, refresh=refresh, H=H, SC=SC, ST=ST, setups=setups, GG=GG)
    @printf("%-30s refresh=%-8g anderson=%-3d => t=%6.1fs steps=%-4d resid=%.3e setups=%d converged=%s\n",
            label, refresh, anderson, t, res.steps, res.residual, setups[], res.converged)
    flush(stdout)
end

trial("frozen (baseline)", 1e9)
trial("rebuild EVERY step (true Newton)", 0.0)
trial("rebuild every step, no Anderson", 0.0; anderson=0)
println("DONE")
