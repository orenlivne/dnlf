# A/B test the polish-stage rebuild criterion on the two corpus outliers (Gleich__wb-cs-stanford,
# Newman__cond-mat-2003) that stalled under the residual-ratio heuristic (:residual, baseline) with
# 209 and 167 setups / 1294s and 2295s respectively, vs. machine-precision converged neighbors of
# similar size at ~90-110 setups / tens-to-hundreds of seconds. Tests the active-set-change criterion
# (:activeset): rebuild only when the arc active/inactive partition itself has changed materially,
# not merely when the residual failed to drop fast enough.
#   Usage:  AIER_DATA=~/code/data julia --project=. scripts/ab_refresh_mode.jl
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))

const CASES = ["Gleich__wb-cs-stanford", "Newman__cond-mat-2003"]
const BASELINE = Dict("Gleich__wb-cs-stanford" => (1293.901, 209, 4.319e-07),
                       "Newman__cond-mat-2003" => (2295.045, 167, 5.768e-09))

function run_one(g, mode)
    path = joinpath(datadir(), g*".mtx")
    n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
    Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    t = @elapsed ((φ, f, _, setups) = DNLF.solve_flow(net, d, zeros(net.m); inner=:multigrid, Hpack=Hp,
                                                        polish_refresh_mode=mode))
    resid = norm(net.B * f .+ d) / max(norm(d), eps())
    net.m, t, setups, resid
end

println("warming up (compile)...")
run_one(CASES[1], :residual)   # compile both code paths once, untimed

for g in CASES
    (bt, bs, br) = BASELINE[g]
    @printf("\n%s\n", g)
    @printf("  baseline (:residual, from corpus run): %8.1fs  setups=%-4d  resid=%.2e\n", bt, bs, br)
    m, t, s, r = run_one(g, :activeset)
    @printf("  :activeset                           : %8.1fs  setups=%-4d  resid=%.2e\n", t, s, r)
    @printf("  -> %.1fx time, %.1fx setups, resid %s\n", bt/t, bs/s, r <= 1e-8 ? "OK (<=1e-8)" : "WORSE")
end
println("\nDONE")
