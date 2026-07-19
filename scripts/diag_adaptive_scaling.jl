# Verify the loose-homotopy + adaptive-polish fix generalizes AND that polish setups stay bounded across
# graph SIZE (the paper's claim is bounded-independent-of-size; the 21->127 jump on cond-mat-2003 only
# matters if it keeps growing with m). Span ~2 decades of size across diverse capped/pathological graphs.
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))

graphs = ["Gleich__wb-cs-stanford", "SNAP__ca-HepPh", "SNAP__soc-Epinions1",
          "SNAP__web-NotreDame", "SNAP__amazon0601"]  # 52k, 235k, 811k, 2.2M, 4.9M arcs

@printf("%-28s %-9s %-8s %-9s %-8s\n","graph","m","t(s)","setups","resid"); flush(stdout)
for g in graphs
    path = joinpath(datadir(), g*".mtx"); isfile(path) || (println("(missing $g)"); continue)
    n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
    Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    t = @elapsed ((φ, f, steps, setups) = DNLF.solve_flow(net, d, zeros(net.m); inner=:multigrid, Hpack=Hp,
                     itol=3e-2, inmax=6, polish_refresh=0.25, tlim=600.0))
    resid = norm(net.B*f .+ d) / max(norm(d),1.0)
    @printf("%-28s %-9d %-8.1f %-9d %.3e\n", g, net.m, t, setups, resid); flush(stdout)
end
println("DONE")
