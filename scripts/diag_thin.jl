# Thin the δ-homotopy on a homotopy-time-bound large graph (soc-Epinions1, 811k arcs). The 111-level count
# is purely a consequence of the 0.85 geometric ratio reaching δ≈8e-9; a COARSER ratio = fewer, bigger steps.
# The fine ratio was chosen so each level is a small perturbation the frozen-per-level solve handles -- but
# with ADAPTIVE polish (0.25) now cleaning up the tail, a coarse "get close" homotopy may reach good residual
# far faster. Sweep the ratio; report #levels, time, setups, final residual. Also emit a per-level trajectory
# for the coarsest schedule (verbose) to see whether coarse levels flow smoothly or stall.
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))

g = "SNAP__soc-Epinions1"
path = joinpath(datadir(), g*".mtx")
n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
@printf("graph=%s n=%d m=%d\n", g, net.n, net.m); flush(stdout)

# geometric δ-schedule from 0.5 down to ~8e-9, parameterized by ratio (fewer levels for smaller ratio)
function schedule(ratio)
    fr = Float64[]; v = 0.5
    while v > 8e-9; push!(fr, v); v *= ratio; end
    fr
end

function trial(ratio; tlim=900.0, verbose=false)
    dfracs = schedule(ratio)
    Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    t = @elapsed ((φ, f, steps, setups) = DNLF.solve_flow(net, d, zeros(net.m); inner=:multigrid, Hpack=Hp,
                     itol=3e-2, inmax=6, dfracs=dfracs, polish_refresh=0.25, tlim=tlim, polish_verbose=verbose))
    resid = norm(net.B*f .+ d) / max(norm(d),1.0)
    @printf(">> ratio=%.2f (%3d levels): t=%7.1fs steps=%-5d setups=%-5d resid=%.3e\n",
            ratio, length(dfracs), t, steps, setups, resid)
    flush(stdout)
end

for r in (0.85, 0.70, 0.55, 0.40)
    trial(r)
end
println("DONE")
