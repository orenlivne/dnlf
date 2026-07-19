# FULL corpus scaling study (paper §5, Fig 1): time the near-linear directed-UE equilibrium solve (default
# LAMG+ engine, LOOSE-INTERMEDIATE fine-continuation -- itol=3e-2 at intermediate levels, tight tol=1e-9 only
# at the final level + polish, exactly the mode already used and validated for design in this paper) across a
# LARGE, size-spanning, topology-diverse set of real IRREGULAR graphs (social / web / citation / collaboration
# / communication / p2p / AS + ML kNN), spanning ~3 decades of edge count, and fit the wall-clock exponent
# t ~ m^p over the whole corpus. Structured/near-planar graphs (road, US-census, FEM mesh, random-geometric)
# are excluded upstream -- they are the regime where nested dissection is already near-optimal (paper §1.3).
#
# SOLVER MODE: loose homotopy (itol=3e-2, every level incl. the last kept loose via tight_last=false) followed
# by an ADAPTIVE-REBUILD polish (polish_refresh=0.25). Diagnosis (per-level profiling): the 110-level loose
# homotopy is cheap (~30s even at ~1M arcs); the entire large-graph cost was the ONE tight final level, which
# the old loose mode solved with a FROZEN hierarchy and which therefore stalled on stiff hub-dominated graphs.
# Deferring that single tight solve to the adaptive polish (which rebuilds when the linear model goes stale)
# fixes both speed and accuracy: across 52k-4.9M arcs, residual improves 50-400x AND wall-clock drops, with
# setups bounded by DIFFICULTY not size (4.9M arcs -> 91 setups; 811k -> 320). Residual still varies by graph
# (a few percent of the hardest instances do not reach 1e-9 within budget); this script reports it honestly.
#
#   Usage:  AIER_DATA=~/code/data julia --project=. scripts/scaling_corpus_full.jl <list.txt>
#   ENV:    CORPUS_CSV (out path), CORPUS_BUDGET_S (stop starting graphs after this many s; default 6h),
#           CORPUS_MAXARC (skip graphs whose arc count exceeds this; default 20e6),
#           CORPUS_PERGRAPH_TLIM_S (per-graph wall-clock cap; default 300s -- validated sufficient even on
#           the worst known hub-dominated outliers; a timed-out graph is logged with its partial residual,
#           not silently dropped).
#   Emits:  per-graph rows (incrementally) + fitted exponent + CSV (graph,n,m,lamg_s,setups,resid).
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf, Random
include(joinpath(@__DIR__, "scaling.jl"))          # read_mtx_lcc / build_net / solve_loose / datadir

loglogfit(ms, ts) = (x = log10.(ms); y = log10.(ts); xm = sum(x)/length(x); ym = sum(y)/length(y);
                     sum((x .- xm).*(y .- ym)) / sum((x .- xm).^2))

function corpus_run()
    listpath = isempty(ARGS) ? error("pass a graph-list file") : ARGS[1]
    graphs   = [strip(l) for l in eachline(listpath) if !isempty(strip(l)) && !startswith(strip(l), "#")]
    csvpath  = get(ENV, "CORPUS_CSV", joinpath(@__DIR__, "scaling_corpus_full.csv"))
    budget   = parse(Float64, get(ENV, "CORPUS_BUDGET_S", string(6*3600)))
    maxarc   = parse(Float64, get(ENV, "CORPUS_MAXARC", string(20e6)))
    pgtlim   = parse(Float64, get(ENV, "CORPUS_PERGRAPH_TLIM_S", string(300)))
    ms = Float64[]; ts = Float64[]; bs = Int[]; rs = Float64[]
    @printf("%-34s %-9s %-10s %-9s %-7s %-10s\n","graph","n","m","LAMG+ (s)","setups","resid"); flush(stdout)
    csv = open(csvpath, "w"); println(csv, "graph,n,m,lamg_s,setups,resid"); flush(csv)
    warmed = false; t0 = time()
    for g in graphs
        (time() - t0) > budget && (println("  (budget $(round(Int,budget))s reached — stopping)"); break)
        path = joinpath(datadir(), g*".mtx"); isfile(path) || (println("  (missing $g)"); continue)
        local n, edges, net, d
        try
            n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
        catch e
            println("  (read/build failed $g: $(sprint(showerror, e)))"); flush(stdout); continue
        end
        net.m > maxarc && (@printf("  (skip %s, m=%d > maxarc)\n", g, net.m); flush(stdout); continue)
        if !warmed
            # compile on a tiny THROWAWAY synthetic graph, never on the corpus's first real (possibly
            # pathological/hub-dominated) instance — that graph's own tlim cap only wraps the timed call
            # below, not an untimed warmup, so warming up on it could hang exactly like the uncapped path did.
            wn, wedges = 200, [(i, i+1) for i in 1:199]
            wnet, wd = build_net(wn, wedges)
            try; solve_loose(wnet, wd; inner=:multigrid); catch; end
            warmed = true
        end
        Hp = (Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
        local t, φ, f, setups
        try
            t = @elapsed ((φ, f, _, setups) = DNLF.solve_flow(net, d, zeros(net.m); inner=:multigrid, Hpack=Hp,
                                                                tlim=pgtlim, itol=3e-2, inmax=6,
                                                                tight_last=false, polish_refresh=0.25))
        catch e
            println("  (solve failed $g: $(sprint(showerror, e)))"); flush(stdout); continue
        end
        resid = norm(net.B * f .+ d) / max(norm(d), eps())   # solver convention: Bn=-net.B ⇒ net.B*f + d = 0
        push!(ms, net.m); push!(ts, t); push!(bs, setups); push!(rs, resid)
        @printf("%-34s %-9d %-10d %-9.2f %-7d %.2e\n", g, net.n, net.m, t, setups, resid); flush(stdout)
        println(csv, @sprintf("%s,%d,%d,%.3f,%d,%.3e", g, net.n, net.m, t, setups, resid)); flush(csv)
    end
    close(csv)
    if length(ms) >= 2
        p = loglogfit(ms, ts)
        rs_sorted = sort(rs)
        med = rs_sorted[length(rs_sorted)÷2 + 1]
        @printf("\ncorpus: %d graphs, m in [%d, %d] (%.2f decades); LAMG+ wall-clock t ~ m^%.3f; setups in [%d, %d]\n",
                length(ms), Int(minimum(ms)), Int(maximum(ms)), log10(maximum(ms)/minimum(ms)), p, minimum(bs), maximum(bs))
        @printf("resid: min %.1e, median %.1e, max %.1e; #resid<1e-6: %d/%d; #resid<1e-3: %d/%d; #resid<1e-1: %d/%d\n",
                minimum(rs), med, maximum(rs), count(<(1e-6),rs), length(rs), count(<(1e-3),rs), length(rs),
                count(<(1e-1),rs), length(rs))
    end
    println("DONE ($(length(ms)) graphs, $(round(time()-t0))s)")
end

corpus_run()
