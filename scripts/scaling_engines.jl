# =====================================================================================================
# Engine scaling study: on the same size-scaled irregular family as scaling_synthetic.jl, time BOTH
# near-linear inner engines — approximate Cholesky (approxChol, the default) and LAMG+ (:multigrid) —
# against the fair frozen-per-level direct baseline, and fit each wall-clock exponent t ~ m^p in-run.
# Purpose: show the framework admits a provably O(m) inner engine (LAMG+, exponent ≈ 1 with a larger
# constant) while the default approxChol is empirically m^~1.2 with a much smaller constant — so the
# near-linear complexity is a property of the framework, and the engine is a constant-factor choice.
#
# Usage:  julia --project=. scripts/scaling_engines.jl
# Emits:  per-size rows, the three fitted exponents, and scaling_engines.csv.
# =====================================================================================================
using DNLF, LinearAlgebra, SparseArrays, Printf, Random

function rand_net(n; deg=5, seed=1)                       # identical generator to scaling_synthetic.jl
    rng=MersenneTwister(seed); E=Set{Tuple{Int,Int}}()
    for u in 1:n,_ in 1:deg; v=rand(rng,1:n); v==u&&continue; push!(E,(min(u,v),max(u,v))); end
    es=collect(E); ini=Int[];ter=Int[]; for (u,v) in es; push!(ini,u);push!(ter,v);push!(ini,v);push!(ter,u); end
    m=length(ini); B=sparse([ini;ter],[1:m;1:m],[fill(-1.0,m);fill(1.0,m)],n,m)
    nt=DNLF.DirectedNetwork(n,m,ini,ter,1000 .*(0.5 .+rand(rng,m)),1 .+rand(rng,m),fill(0.15,m),fill(4.0,m),B)
    dd=zeros(n); for u in randperm(rng,n)[1:n÷8];dd[u]+=1;end; for v in randperm(rng,n)[1:n÷8];dd[v]-=1;end
    dd.-=sum(dd)/n; dd.*=(3000.0*n/(sum(abs,dd)/2)); nt,dd
end

solve(net,d;inner) = DNLF.solve_flow(net,d,zeros(net.m); inner=inner)   # accurate mode (fine schedule) → true 1e-9

function loglogfit(ms, ts)                                # centered least-squares log-log slope
    x = log10.(ms); y = log10.(ts); xm = sum(x)/length(x); ym = sum(y)/length(y)
    b = sum((x .- xm).*(y .- ym)) / sum((x .- xm).^2); b, ym - b*xm
end

NS = (1000, 2000, 4000, 8000, 16000, 32000, 64000)        # m up to ~6.4e5 (same family as Table 1)
LUCAP = 45_000                                            # direct: the fine-continuation solve does ~100 inner
                                                          # solves, each an LU factorization here, so we cap direct
                                                          # at the small, crossover-relevant sizes (near-linear runs full range)

let n0 = rand_net(1000)                                   # compile each engine once (not an untimed pre-solve per size)
    solve(n0[1],n0[2]; inner=:multigrid); solve(n0[1],n0[2]; inner=:approxchol); solve(n0[1],n0[2]; inner=:lu)
end
ms=Float64[]; tlm=Float64[]; tac=Float64[]; mlu=Float64[]; tlu=Float64[]
@printf("%-8s %-9s | %-11s %-11s %-11s\n","n","m","LAMG+","approxChol","direct")
for n in NS
    net,d = rand_net(n)
    tm = @elapsed solve(net,d; inner=:multigrid)
    ta = @elapsed solve(net,d; inner=:approxchol)
    push!(ms,net.m); push!(tlm,tm); push!(tac,ta)
    tl = NaN
    if net.m <= LUCAP
        tl = @elapsed solve(net,d; inner=:lu)
        push!(mlu,net.m); push!(tlu,tl)
    end
    @printf("%-8d %-9d | %-11.2f %-11.2f %-11s\n", net.n, net.m, tm, ta, isnan(tl) ? "OOM" : @sprintf("%.2f",tl))
    flush(stdout)
end

plm,_ = loglogfit(ms, tlm); pac,_ = loglogfit(ms, tac); plu,_ = loglogfit(mlu, tlu)
@printf("\nLAMG+:  t ~ m^%.2f      approxChol:  t ~ m^%.2f      direct:  t ~ m^%.2f\n", plm, pac, plu)
@printf("LAMG+/approxChol wall-clock ratio at largest m: %.2fx\n", tlm[end]/tac[end])
open(joinpath(@__DIR__,"scaling_engines.csv"),"w") do io
    println(io,"m,lamg_s,approxchol_s,direct_s")
    for i in eachindex(ms)
        di = findfirst(==(ms[i]), mlu)
        println(io, @sprintf("%d,%.3f,%.3f,%s", Int(ms[i]), tlm[i], tac[i], di===nothing ? "" : @sprintf("%.3f",tlu[di])))
    end
end
println("wrote scaling_engines.csv")
