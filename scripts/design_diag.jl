# Fast diagnostic: is the CAIDA instance congestible, and does the ANALYTIC marginal-cost (Pigouvian) toll
# reduce TSTT? tau*_a = f_a t'_a(f_a) is the congestion externality; the tolled UE at tau* is the system
# optimum. Reports congestion level (f/cap) and TSTT reduction at tau* -- one warm re-solve, no gradient loop.
# Also sweeps a demand multiplier to find a congested operating point if the default is uncongested.
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))

g = "DIMACS10__caidaRouterLevel"
path = joinpath(datadir(), g*".mtx")
n, edges = read_mtx_lcc(path); net, d0 = build_net(n, edges)
@printf("graph=%s n=%d m=%d\n", g, net.n, net.m); flush(stdout)

for mult in (1.0, 5.0, 20.0)
    d = d0 .* mult
    Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    _, f, _, _ = DNLF.solve_flow(net, d, zeros(net.m); inner=:multigrid, Hpack=Hp,
                                  itol=3e-2, inmax=6, tight_last=false, polish_refresh=0.25)
    T0 = DNLF.tstt(net, f)
    vc = [f[a]/net.cap[a] for a in 1:net.m]
    active = count(>(1e-9), f)
    # marginal-cost (Pigouvian) toll and its tolled UE
    τ = [f[a]*DNLF.dcost(net, a, f[a]) for a in 1:net.m]
    _, f2, _, _ = DNLF.solve_flow(net, d, τ; inner=:multigrid, Hpack=Hp, init=nothing,
                                   itol=3e-2, inmax=6, tight_last=false, polish_refresh=0.25)
    Tp = DNLF.tstt(net, f2)
    @printf("mult=%-5.1f active=%-7d  mean(f/cap)=%.2f max(f/cap)=%.1f  TSTT0=%.3e  Pigouvian red=%.2f%%\n",
            mult, active, sum(vc)/net.m, maximum(vc), T0, 100*(1-Tp/T0)); flush(stdout)
end
println("DONE")
