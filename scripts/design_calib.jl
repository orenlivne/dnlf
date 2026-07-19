# Calibrate demand to MODERATE (Anaheim-like) congestion on a mid-scale real communication network, so a
# FULL bilevel design loop gives a clean, non-trivial TSTT reduction (unlike the pathologically over-congested
# build_net default). Sweep demand multipliers; report congestion (f/cap) and the analytic marginal-cost
# (Pigouvian) TSTT reduction at each. Pick the mult with peak V/C ~2-4 and a real reduction for the design run.
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))

g = length(ARGS) >= 1 ? ARGS[1] : "SNAP__as-caida"
path = joinpath(datadir(), g*".mtx")
n, edges = read_mtx_lcc(path); net, d0 = build_net(n, edges)
@printf("graph=%s n=%d m=%d\n", g, net.n, net.m); flush(stdout)
@printf("%-8s %-9s %-11s %-11s %-8s %-10s\n","mult","t(s)","mean f/cap","max f/cap","active%","Pig.red%"); flush(stdout)

for mult in (5e-4, 2e-3, 8e-3, 3e-2)
    d = d0 .* mult
    Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    t = @elapsed ((_, f, _, _) = DNLF.solve_flow(net, d, zeros(net.m); inner=:multigrid, Hpack=Hp,
                                                  itol=3e-2, inmax=6, tight_last=false, polish_refresh=0.25))
    vc = [f[a]/net.cap[a] for a in 1:net.m]
    T0 = DNLF.tstt(net, f)
    τ = [f[a]*DNLF.dcost(net, a, f[a]) for a in 1:net.m]                     # marginal-cost (Pigouvian) toll
    Hp2 = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    _, f2, _, _ = DNLF.solve_flow(net, d, τ; inner=:multigrid, Hpack=Hp2, itol=3e-2, inmax=6,
                                   tight_last=false, polish_refresh=0.25)
    Tp = DNLF.tstt(net, f2)
    @printf("%-8.1e %-9.1f %-11.2f %-11.1f %-8.1f %-10.2f\n",
            mult, t, sum(vc)/net.m, maximum(vc), 100*count(>(1e-9),f)/net.m, 100*(1-Tp/T0)); flush(stdout)
end
println("DONE")
