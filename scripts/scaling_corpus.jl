# Corpus scaling study (paper §5, closes limitation on single-family scaling): time the near-linear
# directed-UE equilibrium solve (default LAMG+ engine) across a size-spanning, TOPOLOGY-DIVERSE set of real
# irregular graphs from the SuiteSparse/SNAP collection --- autonomous-system, peer-to-peer, collaboration,
# email, signed-social, citation, and Q&A networks --- and fit the wall-clock exponent t ~ m^p over the whole
# corpus (not one random-graph family). Also records the size-independent inner-engine setup count.
#   Usage:  AIER_DATA=~/code/data julia --project=. scripts/scaling_corpus.jl
#   Emits:  per-graph rows (incrementally) + the corpus-wide fitted exponent + scaling_corpus.csv.
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf, Random
include(joinpath(@__DIR__, "scaling.jl"))          # reuse read_mtx_lcc / build_net

# curated size-spanning, topology-diverse corpus (arcs ~ 2 x nnz, ~28k to ~1M)
const CORPUS = [
    "SNAP__as-735","SNAP__CollegeMsg","SNAP__Oregon-1","SNAP__email-Eu-core","SNAP__ca-HepTh",
    "SNAP__p2p-Gnutella06","SNAP__Oregon-2","SNAP__soc-sign-bitcoin-otc","SNAP__p2p-Gnutella04",
    "SNAP__p2p-Gnutella25","SNAP__p2p-Gnutella24","SNAP__p2p-Gnutella30","SNAP__ca-CondMat",
    "SNAP__wiki-Vote","SNAP__as-caida","SNAP__ca-HepPh","SNAP__p2p-Gnutella31","SNAP__email-Enron",
    "SNAP__wiki-RfA","SNAP__ca-AstroPh"]

loglogfit(ms, ts) = (x = log10.(ms); y = log10.(ts); xm = sum(x)/length(x); ym = sum(y)/length(y);
                     b = sum((x .- xm).*(y .- ym)) / sum((x .- xm).^2); b)

function corpus_run()
    # graphs: a chunk file (one name per line) passed as ARGS[1], else the built-in CORPUS
    graphs = isempty(ARGS) ? CORPUS : [strip(l) for l in eachline(ARGS[1]) if !isempty(strip(l))]
    csvpath = get(ENV, "CORPUS_CSV", joinpath(@__DIR__, "scaling_corpus.csv"))
    ms = Float64[]; ts = Float64[]; bs = Int[]; names = String[]
    @printf("%-26s %-9s %-10s %-7s\n", "graph", "m", "LAMG+ (s)", "setups"); flush(stdout)
    csv = open(csvpath, "w"); println(csv, "graph,m,lamg_s,setups"); flush(csv)
    warmed = false
    for g in graphs
        path = joinpath(datadir(), g*".mtx"); isfile(path) || (println("  (missing $g)"); continue)
        n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
        net.m > 1_200_000 && (println("  (skip $g, m=$(net.m))"); continue)
        warmed || (solve_loose(net, d; inner=:multigrid); warmed = true)          # compile once
        Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
        t = @elapsed DNLF.solve_flow(net, d, zeros(net.m); inner=:multigrid, itol=3e-2, inmax=6, Hpack=Hp)
        b = Hp[4][]
        push!(ms, net.m); push!(ts, t); push!(bs, b); push!(names, g)
        @printf("%-26s %-9d %-10.2f %-7d\n", g, net.m, t, b); flush(stdout)
        println(csv, @sprintf("%s,%d,%.3f,%d", g, net.m, t, b)); flush(csv)
    end
    close(csv)
    p = loglogfit(ms, ts)
    @printf("\ncorpus: %d graphs, m in [%d, %d]; LAMG+ wall-clock t ~ m^%.2f; setups in [%d, %d]\n",
            length(ms), Int(minimum(ms)), Int(maximum(ms)), p, minimum(bs), maximum(bs))
end

corpus_run()
