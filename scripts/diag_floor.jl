# Fourth isolation: the level-1 stall on wb-cs-stanford plateaus at resid~2.57e-3 with the smoothed
# conductance contrast pinned at 1e12 (exactly 1e-12 x max, i.e. most arcs sit at the artificial floor) --
# a double-precision conditioning wall, not staleness or inner-PCG looseness (both already ruled out).
# Test: does RAISING the floor (reducing contrast) let Newton converge past the plateau, holding the
# hierarchy FROZEN (refresh=1e9, the committed default) so this isolates the floor threshold alone?
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))

g = "Gleich__wb-cs-stanford"
path = joinpath(datadir(), g*".mtx")
n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
tmean = sum(net.t0)/net.m
fr = DNLF.DFRACS[1]
Bn = -net.B
law = DNLF.smoothed_law(net, zeros(net.m), fr*tmean)
@printf("graph=%s n=%d m=%d\n", g, net.n, net.m); flush(stdout)

# newton_flow! hardcodes the 1e-12 floor internally, so to sweep it we replicate the loop here directly
# (same structure as diag_nan.jl, frozen hierarchy this time — refresh=1e9, matching the real default).
_zeromean!(x) = (x .-= sum(x)/length(x); x)
function laplacian_clean(J)
    off = J - spdiagm(0 => diag(J)); off = (off + off')/2; dropzeros!(off)
    off + spdiagm(0 => -vec(sum(off, dims=2)))
end
import LAMG: LAMGOptions, setup, solve

function trial(floor_rel; nmax=300)
    x = zeros(net.n); m = net.m
    f = zeros(m); dρ = zeros(m)
    bn = max(norm(d), 1.0)
    H = nothing; SC = 1.0
    local nr, it
    for i in 1:nmax
        it = i
        gp = Bn' * x; law(f, dρ, gp); r = Bn * f .- d; nr = norm(r)
        nr < 1e-9 * bn && break
        dρf = max.(dρ, floor_rel * maximum(dρ))
        SC = maximum(dρf); Lc = laplacian_clean((Bn * Diagonal(dρf) * Bn') ./ SC)
        H = setup(Lc; options = LAMGOptions())     # rebuild EVERY step (true Newton) — isolates floor's
                                                    # effect on the achievable residual floor, not staleness
        rhs = _zeromean!(-Vector(r) ./ SC)
        δ, info = solve(H, rhs; options = LAMGOptions(tol = 0.05))
        δ = _zeromean!(δ)
        τ = 1.0; accepted = false
        for _ in 1:60
            xt = _zeromean!(x .+ τ .* δ); ft = similar(f); dt = similar(dρ)
            gt = Bn' * xt; law(ft, dt, gt)
            if norm(Bn*ft .- d) <= nr; x = xt; accepted = true; break; end
            τ *= 0.5
        end
        accepted || break
    end
    contrast = maximum(max.(dρ, floor_rel*maximum(dρ))) / minimum(max.(dρ, floor_rel*maximum(dρ)))
    @printf("floor_rel=%-10.0e => it=%-4d resid=%.3e contrast=%.2e\n", floor_rel, it, nr, contrast)
    flush(stdout)
end

trial(1e-12)   # baseline (committed default)
trial(1e-10)
trial(1e-8)
trial(1e-6)
trial(1e-4)
println("DONE")
