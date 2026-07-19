# Fifth isolation: does a proper Armijo SUFFICIENT-DECREASE line search (not just residual-monotonicity,
# the current homotopy rule) escape the ~2.4e-3 plateau found with rebuild-every-step + floor sweeps?
# Merit G(x) = 0.5||r(x)||^2 (principled surrogate energy; closed-form arc_psi isn't available for the
# smoothed law). Armijo: accept tau if G(x+tau*delta) <= G(x) - c*tau*||r||^2 (standard slope estimate,
# since an exact Newton direction satisfies grad(G).delta ~= -||r||^2).
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

_zeromean!(x) = (x .-= sum(x)/length(x); x)
function laplacian_clean(J)
    off = J - spdiagm(0 => diag(J)); off = (off + off')/2; dropzeros!(off)
    off + spdiagm(0 => -vec(sum(off, dims=2)))
end
import LAMG: LAMGOptions, setup, solve

function trial(label; rebuild_every_step=true, nmax=300, floor_rel=1e-12, c=1e-4, eta=0.05)
    x = zeros(net.n); m = net.m
    f = zeros(m); dρ = zeros(m)
    bn = max(norm(d), 1.0)
    H = nothing; SC = 1.0
    local nr = 0.0; local it = 0; accepted_count = 0
    for i in 1:nmax
        it = i
        gp = Bn' * x; law(f, dρ, gp); r = Bn * f .- d; nr = norm(r)
        nr < 1e-9 * bn && break
        dρf = max.(dρ, floor_rel * maximum(dρ))
        if rebuild_every_step || H === nothing
            SC = maximum(dρf); Lc = laplacian_clean((Bn * Diagonal(dρf) * Bn') ./ SC)
            H = setup(Lc; options = LAMGOptions())
        end
        rhs = _zeromean!(-Vector(r) ./ SC)
        δ, info = solve(H, rhs; options = LAMGOptions(tol = eta))
        δ = _zeromean!(δ)
        G0 = 0.5 * nr^2
        τ = 1.0; accepted = false
        for _ in 1:60
            xt = _zeromean!(x .+ τ .* δ); ft = similar(f); dt = similar(dρ)
            gt = Bn' * xt; law(ft, dt, gt); rt = Bn*ft .- d
            Gt = 0.5 * norm(rt)^2
            if Gt <= G0 - c*τ*nr^2; x = xt; accepted = true; accepted_count += 1; break; end
            τ *= 0.5
        end
        accepted || break
    end
    @printf("%-40s it=%-4d resid=%.3e accepted=%d/%d\n", label, it, nr, accepted_count, it)
    flush(stdout)
end

trial("Armijo, rebuild-every-step, floor=1e-12")
trial("Armijo, rebuild-every-step, floor=1e-8"; floor_rel=1e-8)
trial("Armijo, FROZEN hierarchy (default), floor=1e-12"; rebuild_every_step=false)
trial("Armijo, rebuild-every-step, eta=1e-3"; eta=1e-3)
trial("Armijo, rebuild-every-step, eta=1e-6"; eta=1e-6)
trial("Armijo, rebuild-every-step, eta=1e-3, floor=1e-8"; eta=1e-3, floor_rel=1e-8)
println("DONE")
