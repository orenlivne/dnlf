# Solving the right problem: level 1 is only a warm-start waypoint for level 2, not a value anyone needs
# at 1e-9. The committed solver already has a loose-intermediate mode (itol~3e-2, only the final level +
# polish tighten to full tol) -- exactly what the paper already uses for design (proven lossless there).
# Test: does the FULL committed default solve_flow, run with itol (loose intermediate levels), simply
# avoid ever needing to fully resolve the pathological level-1 near-null mode, converging cleanly to 1e-9
# at the END via the final level + polish instead? Compare against accurate mode (itol=nothing, what
# scaling_corpus_full.jl uses, and what stalled/timed out).
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))

g = "Gleich__wb-cs-stanford"
path = joinpath(datadir(), g*".mtx")
n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
@printf("graph=%s n=%d m=%d\n", g, net.n, net.m); flush(stdout)

Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
t = @elapsed ((φ, f, steps, setups) = DNLF.solve_flow(net, d, zeros(net.m); inner=:multigrid, Hpack=Hp,
                                                        itol=3e-2, inmax=6, tlim=180.0))
resid = norm(net.B*f .+ d) / max(norm(d),1.0)
@printf("LOOSE-INTERMEDIATE mode: t=%.1fs steps=%d setups=%d resid=%.3e\n", t, steps, setups, resid)
println("DONE")
