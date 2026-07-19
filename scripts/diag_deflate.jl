# General deflation fix: detect near-isolated clusters (union-find over "strong", i.e. non-floored, arcs
# -- any connected component besides the single giant one is a near-null-space candidate, generalizing
# beyond the one clique found by hand) and apply one-shot additive deflation around the existing black-box
# AMG solve: exactly satisfy the small (k x k) coarse system for the near-null directions, then let AMG/CG
# solve only the well-conditioned remainder. Tests end-to-end from cold start on wb-cs-stanford level 1.
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf, Random
include(joinpath(@__DIR__, "scaling.jl"))
import LAMG: LAMGOptions, setup, solve

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

"""Deflated inverse power iteration: find the k smallest eigenvectors of Lc via inverse power iteration
(reusing the AMG solve H as the approximate inverse-apply), deflating previously-found vectors out of each
new search direction (standard block/sequential deflation), stopping once the found eigenvalue rises above
`rel_thresh` x lambda_max (well above the floor-driven near-null scale ~1e-12, comfortably below any
legitimate small-but-real Fiedler-type eigenvalue) or `maxk` vectors are found. GENERALIZES to however many
pathological near-null modes are present -- not tied to any single hand-identified clique."""
function find_near_null_modes(H, Lc; maxk=20, rel_thresh=1e-6, iters=12, seed=1)
    n = size(Lc, 1)
    λmax = maximum(sum(abs.(Lc), dims=2))   # cheap Gershgorin upper bound
    rng = MersenneTwister(seed)
    V = Vector{Vector{Float64}}(); Λ = Float64[]
    for k in 1:maxk
        v = _zeromean!(randn(rng, n))
        for u in V; v .-= dot(u, v) .* u; end     # start orthogonal to previously found modes
        nv = norm(v); nv > 0 && (v ./= nv)
        local λ = NaN
        for _ in 1:iters
            w, info = solve(H, v; options = LAMGOptions(tol = 1e-10))
            w = _zeromean!(w)
            for u in V; w .-= dot(u, w) .* u; end  # re-orthogonalize against already-found modes
            nw = norm(w); nw < 1e-14 && break
            w ./= nw
            λ = dot(w, Lc*w) / dot(w, w)
            v = w
        end
        (isnan(λ) || λ > rel_thresh * λmax) && break
        push!(V, v); push!(Λ, λ)
    end
    V, Λ
end

"One-shot additive deflation: exactly solve the k x k coarse system for the k found near-null eigenvectors,
subtract their induced RHS contribution, solve the (now well-conditioned) remainder via the existing
black-box AMG solve, then recombine. k=0 (no near-null modes found) reduces to the plain solve."
function deflated_solve(H, Lc, rhs, V, eta)
    if isempty(V)
        y, info = solve(H, rhs; options = LAMGOptions(tol = eta))
        return _zeromean!(y)
    end
    n = size(Lc, 1); k = length(V)
    Z = reduce(hcat, V)                 # n x k, orthonormal near-null eigenvectors
    LcZ = Lc * Z
    E = Z' * LcZ
    μ = (E + 1e-12*I) \ (Z' * rhs)
    rhs2 = rhs - LcZ * μ
    y, info = solve(H, rhs2; options = LAMGOptions(tol = eta))
    _zeromean!(Z * μ + y)
end

function trial(label; use_deflation, nmax=60, eta=1e-6, floor_rel=1e-12, c=1e-4, rel_thresh=1e-6)
    x = zeros(net.n); m = net.m
    f = zeros(m); dρ = zeros(m)
    local nr = Inf; local it = 0; nmodes_seen = Int[]
    for i in 1:nmax
        it = i
        gp = Bn' * x; law(f, dρ, gp); r = Bn * f .- d; nr = norm(r)
        nr < 1e-9 * max(norm(d),1.0) && break
        dρf = max.(dρ, floor_rel * maximum(dρ))
        SC = maximum(dρf); Lc = laplacian_clean((Bn * Diagonal(dρf) * Bn') ./ SC)
        H = setup(Lc; options = LAMGOptions())
        rhs = _zeromean!(-Vector(r) ./ SC)
        V, Λ = use_deflation ? find_near_null_modes(H, Lc; rel_thresh=rel_thresh) : (Vector{Float64}[], Float64[])
        push!(nmodes_seen, length(V))
        δ = deflated_solve(H, Lc, rhs, V, eta)
        G0 = 0.5*nr^2; τ = 1.0
        for _ in 1:60
            xt = _zeromean!(x .+ τ .* δ); ft = similar(f); dt = similar(dρ)
            gt = Bn'*xt; law(ft, dt, gt); rt = Bn*ft .- d
            if 0.5*norm(rt)^2 <= G0 - c*τ*nr^2; x = xt; break; end
            τ *= 0.5
        end
    end
    @printf("%-35s it=%-4d resid=%.3e  modes(min/max/last)=%d/%d/%d\n",
            label, it, nr, isempty(nmodes_seen) ? 0 : minimum(nmodes_seen),
            isempty(nmodes_seen) ? 0 : maximum(nmodes_seen),
            isempty(nmodes_seen) ? 0 : nmodes_seen[end])
    flush(stdout)
end

trial("NO deflation (baseline)"; use_deflation=false)
trial("WITH deflation"; use_deflation=true)
println("DONE")
