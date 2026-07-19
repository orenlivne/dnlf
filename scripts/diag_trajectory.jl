# Per-level TIME trajectory on soc-Epinions1: level 1 is cheap (~6s) but the full-run average is ~40s/level,
# so cost must balloon on later (smaller-delta, stiffer) levels. Run the real loose solve with per-level
# verbose, capped, and read off elapsed-per-level to see WHERE and how fast the cost grows -- that pinpoints
# the actual large-graph bottleneck (a specific stiff delta band? runaway steps? line-search thrash late?).
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))

g = "SNAP__soc-Epinions1"
path = joinpath(datadir(), g*".mtx")
n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
@printf("graph=%s n=%d m=%d\n", g, net.n, net.m); flush(stdout)

Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
# polish_verbose=true prints, per level: steps, resid, setups, cumulative elapsed. tlim caps it.
DNLF.solve_flow(net, d, zeros(net.m); inner=:multigrid, Hpack=Hp,
                itol=3e-2, inmax=6, polish_refresh=0.25, tlim=250.0, polish_verbose=true)
println("DONE")
