# 5th data point for Table 3 (tab:mc) right block: inner CG iters vs n at K=8, spread demand, γ=0.1.
# Uses the exact same rand_net/demand/seed recipe as scaling_mc.jl so it extends that column consistently.
using DNLF, LinearAlgebra, SparseArrays, Random, Printf
function rand_net(n; deg=5, seed=1)
    rng=MersenneTwister(seed); E=Set{Tuple{Int,Int}}()
    for u in 1:n,_ in 1:deg; v=rand(rng,1:n); v==u&&continue; push!(E,(min(u,v),max(u,v))); end
    es=collect(E); ini=Int[];ter=Int[]; for (u,v) in es; push!(ini,u);push!(ter,v);push!(ini,v);push!(ter,u); end
    m=length(ini); B=sparse([ini;ter],[1:m;1:m],[fill(-1.0,m);fill(1.0,m)],n,m)
    DNLF.DirectedNetwork(n,m,ini,ter,1000 .*(0.5 .+rand(rng,m)),1 .+rand(rng,m),fill(0.15,m),fill(4.0,m),B)
end
function demand(N,rng; D0=3.0e3, S=6)
    v=zeros(N.n); for _ in 1:S; v[rand(rng,1:N.n)]+=D0/S; v[rand(rng,1:N.n)]-=D0/S; end
    v.-=sum(v)/N.n; v
end
# warm up (compile) on the existing n=8000 point, then time the new n=16000 point.
for n in (8000, 16000)
    Nn=rand_net(n); r=MersenneTwister(7)
    t=@elapsed (it=DNLF.solve_sue(Nn,[demand(Nn,r) for _ in 1:8],0.1)[4])
    @printf("n=%-7d m=%-9d iters=%-4d  (%.1fs)\n", n, Nn.m, it, t)
end
println("DONE")
