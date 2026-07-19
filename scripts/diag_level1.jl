# Isolate JUST level 1 of the smoothing homotopy (the smoothest, easiest problem — largest delta) on
# a stress graph, and compare the current frozen-within-level Jacobian (refresh=1e9, status quo) against
# allowing the AMG hierarchy to rebuild mid-level when the residual genuinely stalls. Fast (single level,
# not the full 111-level chain), with live per-iteration output.
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))

g = length(ARGS) >= 1 ? ARGS[1] : "Gleich__wb-cs-stanford"
path = joinpath(datadir(), g*".mtx")
n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
tmean = sum(net.t0)/net.m
fr = DNLF.DFRACS[1]                                   # level-1 fraction (largest delta, "easiest")
@printf("graph=%s n=%d m=%d level1_delta_frac=%.3f\n", g, net.n, net.m, fr); flush(stdout)

Bn = -net.B
law = DNLF.smoothed_law(net, zeros(net.m), fr*tmean)

function trial(label, refresh, refresh_mode)
    x = zeros(net.n)
    H, SC, ST, setups, GG = Ref{Any}(nothing), Ref(1.0), Ref(false), Ref(0), Ref(1.0)
    println("\n--- $label (refresh=$refresh, mode=$refresh_mode) ---"); flush(stdout)
    t = @elapsed res = NLF.newton_flow!(x, Bn, law, d; inner=:multigrid, tol=1e-9, nmax=300,
                     anderson=8, refresh=refresh, refresh_mode=refresh_mode,
                     H=H, SC=SC, ST=ST, setups=setups, GG=GG, verbose=true)
    @printf("  => t=%.1fs steps=%d resid=%.3e setups=%d converged=%s\n",
            t, res.steps, res.residual, setups[], res.converged)
end

trial("baseline (frozen within level)", 1e9, :residual)
trial("residual-ratio refresh (like polish default)", 0.25, :residual)
trial("active-set refresh", 0.25, :activeset)
println("\nDONE")
