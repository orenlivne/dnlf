# Verify loose-intermediate mode sidesteps the pathology on ALL known outliers, not just wb-cs-stanford:
# Newman__power, DIMACS10__uk (wrong-regime, excluded from corpus anyway, but check), cond-mat-2003,
# NotreDame_www, caidaRouterLevel.
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))

graphs = ["Gleich__wb-cs-stanford", "Newman__cond-mat-2003", "Barabasi__NotreDame_www",
          "DIMACS10__caidaRouterLevel", "SNAP__p2p-Gnutella24"]  # last one: mild outlier for comparison

for g in graphs
    path = joinpath(datadir(), g*".mtx")
    if !isfile(path); println("(missing $g)"); continue; end
    n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
    net.m > 3_000_000 && (println("(skip $g, too large for this quick check)"); continue)
    Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    t = @elapsed ((φ, f, steps, setups) = DNLF.solve_flow(net, d, zeros(net.m); inner=:multigrid, Hpack=Hp,
                                                            itol=3e-2, inmax=6, tlim=300.0))
    resid = norm(net.B*f .+ d) / max(norm(d),1.0)
    @printf("%-28s n=%-8d m=%-9d t=%7.1fs steps=%-5d setups=%-5d resid=%.3e\n",
            g, net.n, net.m, t, steps, setups, resid)
    flush(stdout)
end
println("DONE")
