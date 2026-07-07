# Reproducible scaling study: directed user-equilibrium solve on REAL irregular (poorly-separable)
# communication/overlay networks from the SuiteSparse/SNAP collection. Compares the near-linear engine
# (approximate Cholesky) against a FAIR direct baseline (LU factorization FROZEN per continuation level and
# reused by back-substitution within the level — the same freezing discipline as the AMG hierarchy, so the
# comparison isolates factorization vs near-linear cost, not per-step refactoring). Loose tolerances at
# intermediate continuation levels, tight final solve.
#
# Usage:  AIER_DATA=~/code/aier/data julia --project=. scripts/scaling.jl [graph1.mtx graph2.mtx ...]
# Data:   SuiteSparse/SNAP .mtx graphs under $AIER_DATA (default ~/code/aier/data).
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf, Random

datadir() = get(ENV, "AIER_DATA", joinpath(homedir(), "code", "aier", "data"))

"Read a symmetric MatrixMarket .mtx as an undirected edge list on its largest connected component."
function read_mtx_lcc(path)
    I = Int[]; J = Int[]; n = 0; started = false
    for ln in eachline(path)
        (isempty(ln) || startswith(ln, "%")) && continue
        t = split(ln)
        if !started; n = parse(Int, t[1]); started = true; continue; end
        i = parse(Int, t[1]); j = parse(Int, t[2]); i == j && continue
        push!(I, i); push!(J, j)
    end
    # adjacency for BFS (largest connected component)
    adj = [Int[] for _ in 1:n]
    for k in eachindex(I); push!(adj[I[k]], J[k]); push!(adj[J[k]], I[k]); end
    seen = zeros(Int, n); comp = 0
    for s in 1:n
        seen[s] != 0 && continue; comp += 1; q = [s]; seen[s] = comp
        while !isempty(q); u = pop!(q); for v in adj[u]; seen[v]==0 && (seen[v]=comp; push!(q,v)); end; end
    end
    # pick largest component
    sizes = zeros(Int, comp); for c in seen; sizes[c] += 1; end
    big = argmax(sizes); keepn = findall(==(big), seen)
    remap = zeros(Int, n); for (k,v) in enumerate(keepn); remap[v] = k; end
    Ie = Int[]; Je = Int[]
    for k in eachindex(I)
        if seen[I[k]] == big
            lo, hi = minmax(remap[I[k]], remap[J[k]])
            lo < hi && (push!(Ie, lo); push!(Je, hi))                                 # dedup undirected
        end
    end
    length(keepn), unique(collect(zip(Ie, Je)))
end

"Undirected edge list -> DirectedNetwork (each edge = two opposing arcs, BPR costs) + congested demand."
function build_net(n, edges; seed=1)
    rng = MersenneTwister(seed); ini = Int[]; ter = Int[]
    for (u, v) in edges; push!(ini,u);push!(ter,v); push!(ini,v);push!(ter,u); end
    m = length(ini)
    B = sparse([ini; ter], [1:m; 1:m], [fill(-1.0,m); fill(1.0,m)], n, m)
    net = DNLF.DirectedNetwork(n, m, ini, ter, 1000 .*(0.5 .+ rand(rng,m)), 1 .+ rand(rng,m),
                               fill(0.15,m), fill(4.0,m), B)
    d = zeros(n); for u in randperm(rng,n)[1:max(2,n÷8)]; d[u]+=1; end
    for v in randperm(rng,n)[1:max(2,n÷8)]; d[v]-=1; end
    d .-= sum(d)/n; d .*= (3000.0*n/(sum(abs,d)/2)); net, d
end

"Loose-intermediate smoothing-homotopy solve via the package `solve_flow` (the committed solver), with
engine :approxchol (near-linear) or :lu (fair frozen-per-level direct baseline)."
function solve_loose(net, d; inner=:approxchol)
    Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    _, _, _, setups = DNLF.solve_flow(net, d, zeros(net.m); inner = inner, itol = 3e-2, inmax = 6, Hpack = Hp)
    nothing, setups, 0.0
end

# graphs: default a size-spanning set of irregular communication/overlay + collaboration networks
DEFAULT = ["SNAP__as-735","SNAP__p2p-Gnutella08","SNAP__Oregon-1","SNAP__ca-HepPh","SNAP__ca-AstroPh",
           "SNAP__p2p-Gnutella24","SNAP__as-caida","SNAP__p2p-Gnutella30","SNAP__email-Enron"]
graphs = isempty(ARGS) ? DEFAULT : [replace(a, ".mtx"=>"") for a in ARGS]

@printf("%-26s %-8s %-9s %-11s %-9s %-11s %-9s %-7s\n","graph","n","m","ac_time","ac_bld","lu_time","lu_bld","ratio")
for g in graphs
    path = joinpath(datadir(), g*".mtx"); isfile(path) || (println("  (missing $g)"); continue)
    n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
    net.m > 1_200_000 && (println("  (skip $g, m=$(net.m) too large)"); continue)
    solve_loose(net,d)  # compile once (first graph pays it)
    t_ac = @elapsed ((xa,ba,ra) = solve_loose(net,d; inner=:approxchol))
    t_lu = net.m <= 150_000 ? (@elapsed ((xl,bl,rl) = solve_loose(net,d; inner=:lu))) : NaN
    lub = @isdefined(bl) ? bl : 0
    @printf("%-26s %-8d %-9d %-11.2f %-9d %-11s %-9s %-7s\n", g, net.n, net.m, t_ac, ba,
            isnan(t_lu) ? "-" : @sprintf("%.2f",t_lu), isnan(t_lu) ? "-" : string(lub),
            isnan(t_lu) ? "-" : @sprintf("%.2f",t_ac/t_lu))
end
