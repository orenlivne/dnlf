# Hypothesis: cond-mat-2003's loose-mode plateau (1.6e-2) is the FROZEN polish, not a fundamental limit.
# Loose mode freezes BOTH the homotopy and the final exact-law polish (refresh = loose ? 1e9). Test: keep
# the homotopy loose/cheap but let the polish rebuild adaptively (polish_refresh=0.25, the accurate-mode
# polish behavior). If the residual drops toward 1e-9, the floor was a stale-polish artifact (cheap fix). If
# it plateaus like wb-cs-stanford did under rebuild-every-step, it's the genuine near-null mode.
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))

g = "Newman__cond-mat-2003"
path = joinpath(datadir(), g*".mtx")
n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
@printf("graph=%s n=%d m=%d\n", g, net.n, net.m); flush(stdout)

function trial(label; polish_refresh, tlim=900.0)
    Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    t = @elapsed ((φ, f, steps, setups) = DNLF.solve_flow(net, d, zeros(net.m); inner=:multigrid, Hpack=Hp,
                     itol=3e-2, inmax=6, polish_refresh=polish_refresh, tlim=tlim))
    resid = norm(net.B*f .+ d) / max(norm(d),1.0)
    @printf("%-38s t=%7.1fs steps=%-5d setups=%-5d resid=%.3e\n", label, t, steps, setups, resid)
    flush(stdout)
end

trial("loose homotopy + FROZEN polish (baseline)"; polish_refresh=1e9)
trial("loose homotopy + ADAPTIVE polish (0.25)";   polish_refresh=0.25)
println("DONE")
