# Does the smallest capped graph (Newman__cond-mat-2003, m=232362) converge with more budget, or is it
# capped regardless of tlim (genuine per-graph difficulty, not a simple size-proportional time need)?
# Sweep tlim on this one graph (loose-intermediate mode, same config as the corpus run).
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))

g = "Newman__cond-mat-2003"
path = joinpath(datadir(), g*".mtx")
n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
@printf("graph=%s n=%d m=%d\n", g, net.n, net.m); flush(stdout)

for tlim in (300.0, 600.0, 1200.0, 2400.0)
    Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    t = @elapsed ((φ, f, steps, setups) = DNLF.solve_flow(net, d, zeros(net.m); inner=:multigrid, Hpack=Hp,
                                                            itol=3e-2, inmax=6, tlim=tlim))
    resid = norm(net.B*f .+ d) / max(norm(d),1.0)
    @printf("tlim=%-6.0f => t=%7.1fs steps=%-5d setups=%-5d resid=%.3e  %s\n",
            tlim, t, steps, setups, resid, t >= tlim - 1 ? "(HIT CAP)" : "(converged before cap)")
    flush(stdout)
end
println("DONE")
