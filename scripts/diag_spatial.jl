# Spatial diagnosis of the ~2.4e-3 plateau on wb-cs-stanford level 1: run to the plateau (rebuild-every-
# step + Armijo + tight eta, our fastest-converging configuration), then (1) identify which NODES carry
# the residual mass and cross-reference with topological degree (degree-1 leaf hypothesis), and (2)
# estimate the smallest Jacobian eigenvalues via inverse power iteration (reusing the AMG solve as an
# approximate inverse-apply) to see how large the near-null-space is.
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

# topological degree per node (from the LCC-extracted, build_net-processed network)
deg = zeros(Int, net.n)
for a in 1:net.m; deg[net.ini[a]] += 1; end   # each undirected edge -> 2 arcs, so this double counts consistently
@printf("degree-1 node count: %d / %d (%.1f%%)\n", count(==(1), deg), net.n, 100*count(==(1),deg)/net.n)

_zeromean!(x) = (x .-= sum(x)/length(x); x)
function laplacian_clean(J)
    off = J - spdiagm(0 => diag(J)); off = (off + off')/2; dropzeros!(off)
    off + spdiagm(0 => -vec(sum(off, dims=2)))
end

function run_to_plateau(net, d, Bn, law; nmax=60, eta=1e-6, floor_rel=1e-12, c=1e-4)
    x = zeros(net.n); m = net.m
    f = zeros(m); dρ = zeros(m)
    local r, dρf
    for i in 1:nmax
        gp = Bn' * x; law(f, dρ, gp); r = Bn * f .- d; nr = norm(r)
        nr < 1e-9 * max(norm(d),1.0) && break
        dρf = max.(dρ, floor_rel * maximum(dρ))
        SC = maximum(dρf); Lc = laplacian_clean((Bn * Diagonal(dρf) * Bn') ./ SC)
        H = setup(Lc; options = LAMGOptions())
        rhs = _zeromean!(-Vector(r) ./ SC)
        δ, info = solve(H, rhs; options = LAMGOptions(tol = eta)); δ = _zeromean!(δ)
        G0 = 0.5*nr^2; τ = 1.0
        for _ in 1:60
            xt = _zeromean!(x .+ τ .* δ); ft = similar(f); dt = similar(dρ)
            gt = Bn'*xt; law(ft, dt, gt); rt = Bn*ft .- d
            if 0.5*norm(rt)^2 <= G0 - c*τ*nr^2; x = xt; break; end
            τ *= 0.5
        end
    end
    x, f, dρ, r
end

x, f, dρ, r = run_to_plateau(net, d, Bn, law)
@printf("plateau residual norm = %.3e\n", norm(r)); flush(stdout)

# (1) spatial residual analysis
ord = sortperm(abs.(r), rev=true)
println("\ntop-20 residual nodes: (node, |r_i|, topological_degree, node_demand d_i)")
for k in 1:20
    i = ord[k]
    @printf("  node=%-6d |r|=%.3e  deg=%-3d  d_i=%.3e\n", i, abs(r[i]), deg[i], d[i])
end
top20deg1 = count(==(1), deg[ord[1:20]])
top100deg1 = count(==(1), deg[ord[1:100]])
@printf("\ndegree-1 among top-20 residual nodes: %d/20\n", top20deg1)
@printf("degree-1 among top-100 residual nodes: %d/100\n", top100deg1)
@printf("residual mass in top-20 nodes: %.1f%% of total ||r||^2\n", 100*sum(abs2, r[ord[1:20]])/sum(abs2,r))
@printf("residual mass in top-100 nodes: %.1f%% of total ||r||^2\n", 100*sum(abs2, r[ord[1:100]])/sum(abs2,r))

# (2) smallest-eigenvalue estimate via inverse power iteration (reusing the AMG solve as approx J^{-1})
dρf = max.(dρ, 1e-12 * maximum(dρ))
SC = maximum(dρf); Lc = laplacian_clean((Bn * Diagonal(dρf) * Bn') ./ SC)
H = setup(Lc; options = LAMGOptions())
function inv_power(H, Lc, n)
    rng = MersenneTwister(1); v = _zeromean!(randn(rng, n)); v ./= norm(v)
    λ = NaN
    for it in 1:25
        w, info = solve(H, v; options = LAMGOptions(tol = 1e-10))
        w = _zeromean!(w); w ./= norm(w)
        λ = dot(w, Lc*w) / dot(w,w)
        v = w
        it % 5 == 0 && @printf("  it=%-3d rayleigh(smallest)=%.4e\n", it, λ)
    end
    λ, v
end
println("\ninverse power iteration (Rayleigh quotient -> smallest eigenvalue of the SCALED, clean Jacobian):")
λ, evec = inv_power(H, Lc, net.n)
eord = sortperm(abs.(evec), rev=true)
println("\nnear-null eigenvector: top-10 nodes by |component| (localization check)")
for k in 1:10
    i = eord[k]
    @printf("  node=%-6d |evec|=%.4f  deg=%-3d\n", i, abs(evec[i]), deg[i])
end
# identify the specific arc(s) between the top-support nodes, and their dρf value vs the floor
top2 = eord[1:2]
println("\narcs touching the two dominant nodes, with dρf value (floor = ", 1e-12*maximum(dρf), "):")
for a in 1:net.m
    if net.ini[a] in top2 || net.ter[a] in top2
        @printf("  arc %-6d: %d -> %d   dρf=%.3e  %s\n", a, net.ini[a], net.ter[a], dρf[a],
                dρf[a] <= 1.01*1e-12*maximum(dρf) ? "<-- AT FLOOR" : "")
    end
end
λmax = maximum(sum(abs.(Lc), dims=2))   # Gershgorin upper bound, cheap
@printf("estimated smallest eigenvalue ~ %.3e ; Gershgorin upper bound on largest ~ %.3e ; ratio ~ %.2e\n",
        λ, λmax, λmax/max(λ,1e-300))
println("DONE")
