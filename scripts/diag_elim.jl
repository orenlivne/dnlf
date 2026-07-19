# Structural pre-elimination fix: iteratively Schur-complement-eliminate nodes with small WEIGHTED degree
# (diag(Lc), which for a clean graph Laplacian equals the total conductance to its neighbors) -- generalizes
# LAMG+'s existing topological degree-1 elimination to the actual pathology found (a moderate-degree cluster
# with tiny TOTAL conductance to the rest of the graph, not literal graph disconnection). Exact, local,
# O(deg^2) per eliminated node -- stays O(m) as long as the number of eliminated nodes is small (capped
# below, matching real-graph expectations: this should be rare/localized, not pervasive).
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
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

"""Iteratively eliminate the lowest-weighted-degree ACTIVE node whenever its weighted degree (diag(Lc))
falls below `rel_thresh` x max(diag). Exact Schur-complement (block-elimination) update on the neighbor
submatrix: L_new[j,k] = L[j,k] - L[j,i]*L[i,k]/L[i,i] for all neighbor pairs j,k of the eliminated node i
-- the standard, sign-unambiguous formula (equivalent to one step of Gaussian elimination). Capped at
`maxk` eliminations and `max_deg` per-node degree as a safety bound so fill-in (and hence cost) stays
bounded -- this pathology is expected to be rare/localized, not pervasive, in real graphs."""
function eliminate_weak_nodes(Lc; rel_thresh=1e-6, maxk=200, max_deg=64)
    n = size(Lc, 1)
    L = Matrix(Lc)  # dense working copy -- fine for the SMALL local neighborhoods we touch; see note below
    keep = trues(n)
    order = Int[]; coeff_nbrs = Vector{Vector{Int}}(); coeff_vals = Vector{Vector{Float64}}(); coeff_dii = Float64[]
    dmax0 = maximum(diag(Lc))
    for step in 1:maxk
        d = [keep[i] ? L[i,i] : Inf for i in 1:n]
        val, i = findmin(d)
        (val > rel_thresh * dmax0 || !isfinite(val)) && break
        nbrs = [j for j in 1:n if keep[j] && j != i && L[i,j] != 0]
        length(nbrs) > max_deg && break     # refuse to eliminate a hub -- not the intended target, and would blow up fill-in cost
        Lii = L[i,i]
        Lij = [L[j,i] for j in nbrs]
        for (a,ja) in enumerate(nbrs), (b,jb) in enumerate(nbrs)
            L[ja,jb] -= Lij[a]*Lij[b]/Lii
        end
        push!(order, i); push!(coeff_nbrs, nbrs); push!(coeff_vals, Lij); push!(coeff_dii, Lii)
        keep[i] = false
        L[i,:] .= 0; L[:,i] .= 0
    end
    kept = findall(keep)
    kept, order, coeff_nbrs, coeff_vals, coeff_dii, sparse(L[kept,kept])
end

"Solve Lc*x=rhs by eliminating weak nodes first (exact), solving the reduced (well-conditioned) system via
the existing black-box AMG solve, then back-substituting the eliminated nodes' values (also exact)."
function eliminated_solve(Lc, rhs, eta; rel_thresh=1e-6)
    n = size(Lc,1)
    kept, order, cn, cv, cd, Lred = eliminate_weak_nodes(Lc; rel_thresh=rel_thresh)
    rhs_red = copy(rhs[kept])
    # fold each eliminated node's RHS contribution into its (still-active-at-elim-time) neighbors that
    # survive to the reduced system; since elimination order processes nodes one at a time and neighbors of
    # an eliminated node may themselves later be eliminated, we apply the RHS update directly on a full
    # working copy first, then restrict to `kept` -- simplest correct approach for this diagnostic.
    rhs_work = copy(rhs)
    for (idx, i) in enumerate(order)
        nbrs, vals, dii = cn[idx], cv[idx], cd[idx]
        for (a,j) in enumerate(nbrs)
            rhs_work[j] -= vals[a] * rhs_work[i] / dii
        end
    end
    rhs_red = rhs_work[kept]
    if length(kept) == n   # nothing eliminated
        y, info = solve(setup(Lc; options=LAMGOptions()), rhs; options = LAMGOptions(tol = eta))
        return _zeromean!(y), 0
    end
    Hred = setup(Lred; options=LAMGOptions())
    y, info = solve(Hred, rhs_red; options = LAMGOptions(tol = eta))
    x = zeros(n); x[kept] = y
    for idx in length(order):-1:1
        i = order[idx]; nbrs, vals, dii = cn[idx], cv[idx], cd[idx]
        x[i] = (rhs_work[i] - sum(vals[a]*x[nbrs[a]] for a in eachindex(nbrs); init=0.0)) / dii
    end
    _zeromean!(x), length(order)
end

function trial(label; use_elim, nmax=60, eta=1e-6, floor_rel=1e-12, c=1e-4, rel_thresh=1e-6)
    x = zeros(net.n); m = net.m
    f = zeros(m); dρ = zeros(m)
    local nr = Inf; local it = 0; nelim_seen = Int[]
    for i in 1:nmax
        it = i
        gp = Bn' * x; law(f, dρ, gp); r = Bn * f .- d; nr = norm(r)
        nr < 1e-9 * max(norm(d),1.0) && break
        dρf = max.(dρ, floor_rel * maximum(dρ))
        SC = maximum(dρf); Lc = laplacian_clean((Bn * Diagonal(dρf) * Bn') ./ SC)
        rhs = _zeromean!(-Vector(r) ./ SC)
        δ, nelim = use_elim ? eliminated_solve(Lc, rhs, eta; rel_thresh=rel_thresh) :
                    (_zeromean!(solve(setup(Lc;options=LAMGOptions()), rhs; options = LAMGOptions(tol = eta))[1]), 0)
        push!(nelim_seen, nelim)
        G0 = 0.5*nr^2; τ = 1.0
        for _ in 1:60
            xt = _zeromean!(x .+ τ .* δ); ft = similar(f); dt = similar(dρ)
            gt = Bn'*xt; law(ft, dt, gt); rt = Bn*ft .- d
            if 0.5*norm(rt)^2 <= G0 - c*τ*nr^2; x = xt; break; end
            τ *= 0.5
        end
    end
    @printf("%-35s it=%-4d resid=%.3e  eliminated(min/max/last)=%d/%d/%d\n",
            label, it, nr, isempty(nelim_seen) ? 0 : minimum(nelim_seen),
            isempty(nelim_seen) ? 0 : maximum(nelim_seen),
            isempty(nelim_seen) ? 0 : nelim_seen[end])
    flush(stdout)
end

"Eliminate EXACTLY a given node set (bypassing threshold selection) -- cleanest test of whether removing
precisely the known pathological cluster fixes the plateau."
function eliminate_exact_nodes(Lc, targets)
    n = size(Lc, 1); L = Matrix(Lc); keep = trues(n)
    order = Int[]; coeff_nbrs = Vector{Vector{Int}}(); coeff_vals = Vector{Vector{Float64}}(); coeff_dii = Float64[]
    for i in targets
        !keep[i] && continue
        nbrs = [j for j in 1:n if keep[j] && j != i && L[i,j] != 0]
        Lii = L[i,i]; Lij = [L[j,i] for j in nbrs]
        for (a,ja) in enumerate(nbrs), (b,jb) in enumerate(nbrs)
            L[ja,jb] -= Lij[a]*Lij[b]/Lii
        end
        push!(order, i); push!(coeff_nbrs, nbrs); push!(coeff_vals, Lij); push!(coeff_dii, Lii)
        keep[i] = false; L[i,:] .= 0; L[:,i] .= 0
    end
    kept = findall(keep)
    kept, order, coeff_nbrs, coeff_vals, coeff_dii, sparse(L[kept,kept])
end
function eliminated_solve_exact(Lc, rhs, eta, targets)
    n = size(Lc,1)
    kept, order, cn, cv, cd, Lred = eliminate_exact_nodes(Lc, targets)
    rhs_work = copy(rhs)
    for (idx, i) in enumerate(order)
        nbrs, vals, dii = cn[idx], cv[idx], cd[idx]
        for (a,j) in enumerate(nbrs); rhs_work[j] -= vals[a]*rhs_work[i]/dii; end
    end
    rhs_red = rhs_work[kept]
    Hred = setup(Lred; options=LAMGOptions())
    y, info = solve(Hred, rhs_red; options = LAMGOptions(tol = eta))
    x = zeros(n); x[kept] = y
    for idx in length(order):-1:1
        i = order[idx]; nbrs, vals, dii = cn[idx], cv[idx], cd[idx]
        x[i] = (rhs_work[i] - sum(vals[a]*x[nbrs[a]] for a in eachindex(nbrs); init=0.0)) / dii
    end
    _zeromean!(x), length(order)
end
function trial_exact(label, targets; nmax=60, eta=1e-6, floor_rel=1e-12, c=1e-4)
    x = zeros(net.n); m = net.m
    f = zeros(m); dρ = zeros(m)
    local nr = Inf; local it = 0
    for i in 1:nmax
        it = i
        gp = Bn' * x; law(f, dρ, gp); r = Bn * f .- d; nr = norm(r)
        nr < 1e-9 * max(norm(d),1.0) && break
        dρf = max.(dρ, floor_rel * maximum(dρ))
        SC = maximum(dρf); Lc = laplacian_clean((Bn * Diagonal(dρf) * Bn') ./ SC)
        rhs = _zeromean!(-Vector(r) ./ SC)
        δ, nelim = eliminated_solve_exact(Lc, rhs, eta, targets)
        G0 = 0.5*nr^2; τ = 1.0
        for _ in 1:60
            xt = _zeromean!(x .+ τ .* δ); ft = similar(f); dt = similar(dρ)
            gt = Bn'*xt; law(ft, dt, gt); rt = Bn*ft .- d
            if 0.5*norm(rt)^2 <= G0 - c*τ*nr^2; x = xt; break; end
            τ *= 0.5
        end
    end
    @printf("%-35s it=%-4d resid=%.3e  (eliminated %d target nodes)\n", label, it, nr, length(targets))
    flush(stdout)
end

trial("NO elimination (baseline)"; use_elim=false)
trial("WITH threshold-based structural pre-elimination"; use_elim=true)
# the exact cluster the eigenvector localized on earlier (diag_spatial3.log): nodes 8878,8882,8883,8884,
# 8905,8906,8916,8920,8921,8857, plus their immediate neighbors from the printed arc list (8871,8872,8876,
# 8867,8868,8870,8872,8846,8849,8862,8831,8853,8919,8922) to fully bound the pathological pocket
targets = unique([8878,8882,8883,8884,8905,8906,8916,8920,8921,8857,
                  8871,8872,8876,8867,8868,8870,8846,8849,8862,8831,8853,8919,8922,8826,8875])
trial_exact("WITH EXACT known-cluster elimination", targets)
println("DONE")
