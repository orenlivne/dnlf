# Clean isolation: does tightening the inner PCG accuracy (eta) alone fix the level-1 stall on the
# smallest confirmed outlier (wb-cs-stanford), holding the hierarchy FROZEN (refresh=1e9, the committed
# default)? If yes: the bug is inner-solve inaccuracy compounding with chord-Newton, a simple threshold
# fix. If no: the stall is genuine modified-Newton stagnation from a stale linearization, not inner accuracy.
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

function trial(label, eta)
    x = zeros(net.n)
    H, SC, ST, setups, GG = Ref{Any}(nothing), Ref(1.0), Ref(false), Ref(0), Ref(1.0)
    t = @elapsed res = NLF.newton_flow!(x, Bn, law, d; inner=:multigrid, tol=1e-9, nmax=300,
                     anderson=8, refresh=1e9, eta=eta, H=H, SC=SC, ST=ST, setups=setups, GG=GG)
    @printf("%-28s eta=%-8g => t=%6.1fs steps=%-4d resid=%.3e setups=%d converged=%s\n",
            label, eta, t, res.steps, res.residual, setups[], res.converged)
    flush(stdout)
end

trial("baseline", 0.05)
trial("tighter", 0.01)
trial("tight", 0.001)
trial("very tight", 1e-4)
println("DONE")
