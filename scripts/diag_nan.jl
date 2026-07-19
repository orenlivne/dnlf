# Third isolation: does the search direction delta go NaN/Inf right where linesearch fails (rebuild-every-
# step, no Anderson, broke at step ~27)? If so, the "stall" is a numerical breakdown in the near-singular
# linear solve on a hub-dominated graph (extreme conductance floor/scale contrast), not algorithmic hardness.
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
import LAMG: LAMGOptions, setup, solve

g = "Gleich__wb-cs-stanford"
include(joinpath(@__DIR__, "scaling.jl"))
path = joinpath(datadir(), g*".mtx")
n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
tmean = sum(net.t0)/net.m
fr = DNLF.DFRACS[1]
Bn = -net.B
law = DNLF.smoothed_law(net, zeros(net.m), fr*tmean)

# Manually replicate the newton_flow! loop (refresh=0, no anderson) so we can inspect delta/dρf at the
# exact iteration where the real function's line search fails.
_zeromean!(x) = (x .-= sum(x)/length(x); x)
function laplacian_clean(J)
    off = J - spdiagm(0 => diag(J)); off = (off + off')/2; dropzeros!(off)
    off + spdiagm(0 => -vec(sum(off, dims=2)))
end

function run(net, d, Bn, law)
x = zeros(net.n); m = net.m
f = zeros(m); dρ = zeros(m); nr_prev = Inf
bn = max(norm(d), 1.0)
for it in 1:35
    g_ = Bn' * x; law(f, dρ, g_); r = Bn * f .- d; nr = norm(r)
    dρf = max.(dρ, 1e-12 * maximum(dρ))
    SC = maximum(dρf)
    Lc = laplacian_clean((Bn * Diagonal(dρf) * Bn') ./ SC)
    H = setup(Lc; options = LAMGOptions())
    rhs = _zeromean!(-Vector(r) ./ SC)
    δ, info = solve(H, rhs; options = LAMGOptions(tol = 0.05))
    δ = _zeromean!(δ)
    nan_d = any(isnan, δ); inf_d = any(isinf, δ)
    dmax = maximum(abs, δ); dρmax = maximum(dρf); dρmin = minimum(dρf)
    @printf("it=%-3d nr=%.3e dρ_range=[%.2e,%.2e] contrast=%.2e |δ|max=%.2e nan=%s inf=%s\n",
            it, nr, dρmin, dρmax, dρmax/dρmin, dmax, nan_d, inf_d)
    flush(stdout)
    if nan_d || inf_d
        println("  ==> NaN/Inf CONFIRMED at it=$it — stopping diagnostic here")
        break
    end
    # accept step via residual monotonicity (same rule as the real solver, no energy in homotopy)
    τ = 1.0; accepted = false
    for _ in 1:60
        xt = _zeromean!(x .+ τ .* δ); ft = similar(f); dt = similar(dρ); law(ft, dt, g_ .+ 0)  # placeholder
        gt = Bn' * xt; law(ft, dt, gt)
        if norm(Bn*ft .- d) <= nr; x = xt; accepted = true; break; end
        τ *= 0.5
    end
    if !accepted
        println("  ==> linesearch FAILED at it=$it (60 halvings, no NaN/Inf detected in δ) — genuine breakdown")
        break
    end
end
end
run(net, d, Bn, law)
println("DONE")
