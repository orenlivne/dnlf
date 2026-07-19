# Multi-seed synthetic timing to put a confidence interval on the near-linear exponent of Table 1 (reviewer
# ask: single-run timings, no variance). Re-run the NLF (LAMG+) equilibrium solve on random degree-5 graphs at
# the same sizes as tab:scaling, over 3 seeds; fit the log-log exponent per seed and report median + spread.
using DNLF, NLF, LinearAlgebra, SparseArrays, Random, Printf
function rand_net(n; deg=5, seed=1)
    rng=MersenneTwister(seed); E=Set{Tuple{Int,Int}}()
    for u in 1:n,_ in 1:deg; v=rand(rng,1:n); v==u&&continue; push!(E,(min(u,v),max(u,v))); end
    es=collect(E); ini=Int[];ter=Int[]; for (u,v) in es; push!(ini,u);push!(ter,v);push!(ini,v);push!(ter,u); end
    m=length(ini); B=sparse([ini;ter],[1:m;1:m],[fill(-1.0,m);fill(1.0,m)],n,m)
    nt=DNLF.DirectedNetwork(n,m,ini,ter,1000 .*(0.5 .+rand(rng,m)),1 .+rand(rng,m),fill(0.15,m),fill(4.0,m),B)
    dd=zeros(n); for u in randperm(rng,n)[1:n÷8];dd[u]+=1;end; for v in randperm(rng,n)[1:n÷8];dd[v]-=1;end
    dd.-=sum(dd)/n; dd.*=(3000.0*n/(sum(abs,dd)/2)); nt,dd
end
loglogfit(ms,ts)=(x=log10.(ms);y=log10.(ts);xm=sum(x)/length(x);ym=sum(y)/length(y);
                  sum((x.-xm).*(y.-ym))/sum((x.-xm).^2))

sizes = (1000,2000,4000,8000,16000,32000)
seeds = (1,2,3)
let n0=rand_net(500); DNLF.solve_flow(n0[1],n0[2],zeros(n0[1].m); inner=:multigrid); end  # compile
exps = Float64[]
for sd in seeds
    ms=Float64[]; ts=Float64[]
    for n in sizes
        net,d = rand_net(n; seed=sd)
        t=@elapsed DNLF.solve_flow(net,d,zeros(net.m); inner=:multigrid)   # accurate mode, true 1e-9
        push!(ms,net.m); push!(ts,t)
        @printf("seed=%d n=%-6d m=%-8d t=%.2fs\n", sd, n, net.m, t); flush(stdout)
    end
    p=loglogfit(ms,ts); push!(exps,p)
    @printf("  >> seed=%d exponent m^%.3f\n", sd, p); flush(stdout)
end
sort!(exps)
@printf("\nEXPONENT over %d seeds: median m^%.3f, range [%.3f, %.3f]  (values: %s)\n",
        length(exps), exps[div(length(exps)+1,2)], minimum(exps), maximum(exps),
        join((@sprintf("%.3f",e) for e in exps), ", "))
println("DONE")
