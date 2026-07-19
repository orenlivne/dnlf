# Live-instrumented diagnostic on the polish stage of a single stress graph: prints every chord-Newton
# iteration (residual, rebuild flag, cumulative setups, elapsed) so the stall is directly observable
# instead of waiting on a black-box multi-minute run.
#   Usage: AIER_DATA=~/code/data julia --project=. scripts/diag_polish.jl <graphname> [refresh_mode]
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))

g = length(ARGS) >= 1 ? ARGS[1] : "Gleich__wb-cs-stanford"
mode = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :residual

path = joinpath(datadir(), g*".mtx")
n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
@printf("graph=%s n=%d m=%d refresh_mode=%s\n", g, net.n, net.m, mode); flush(stdout)

Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
t = @elapsed ((φ, f, steps, setups) = DNLF.solve_flow(net, d, zeros(net.m); inner=:multigrid, Hpack=Hp,
                                                        polish_refresh_mode=mode, polish_verbose=true))
resid = norm(net.B * f .+ d) / max(norm(d), eps())
@printf("\nDONE: t=%.1fs steps=%d setups=%d resid=%.3e\n", t, steps, setups, resid)
