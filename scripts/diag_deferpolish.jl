# The final level (111) solved tight-and-FROZEN is the entire large-graph cost (110 loose levels = 32s;
# level 111 = 220s+ and stalls). Fix: keep the WHOLE homotopy loose (tight_last=false) and let the ADAPTIVE
# polish do the single tight solve. Test on soc-Epinions1 (was 1.8e-2 @ cap) + verify it doesn't hurt the
# medium graphs the adaptive polish already fixed (cond-mat-2003, ca-HepPh) or the pathological small one.
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))

graphs = ["SNAP__soc-Epinions1", "Newman__cond-mat-2003", "SNAP__ca-HepPh", "Gleich__wb-cs-stanford",
          "SNAP__amazon0601"]  # 811k, 232k, 235k, 52k, 4.9M

@printf("%-28s %-9s %-8s %-9s %-8s\n","graph","m","t(s)","setups","resid"); flush(stdout)
for g in graphs
    path = joinpath(datadir(), g*".mtx"); isfile(path) || (println("(missing $g)"); continue)
    n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
    Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    t = @elapsed ((φ, f, steps, setups) = DNLF.solve_flow(net, d, zeros(net.m); inner=:multigrid, Hpack=Hp,
                     itol=3e-2, inmax=6, tight_last=false, polish_refresh=0.25, tlim=600.0))
    resid = norm(net.B*f .+ d) / max(norm(d),1.0)
    @printf("%-28s %-9d %-8.1f %-9d %.3e\n", g, net.m, t, setups, resid); flush(stdout)
end
println("DONE")
