# Reproduces the multicommodity logit-SUE iteration table (paper Table 4): block-preconditioned inner
# conjugate-gradient count vs. commodities K (single-OD and spread demand) and vs. graph size n, plus a
# γ-scan showing the honest degradation toward the deterministic limit γ→0.
#   Usage:  julia --project=. scripts/scaling_mc.jl
using DNLF, LinearAlgebra, SparseArrays, Random, Printf
function rand_net(n; deg=5, seed=1)
    rng=MersenneTwister(seed); E=Set{Tuple{Int,Int}}()
    for u in 1:n,_ in 1:deg; v=rand(rng,1:n); v==u&&continue; push!(E,(min(u,v),max(u,v))); end
    es=collect(E); ini=Int[];ter=Int[]; for (u,v) in es; push!(ini,u);push!(ter,v);push!(ini,v);push!(ter,u); end
    m=length(ini); B=sparse([ini;ter],[1:m;1:m],[fill(-1.0,m);fill(1.0,m)],n,m)
    DNLF.DirectedNetwork(n,m,ini,ter,1000 .*(0.5 .+rand(rng,m)),1 .+rand(rng,m),fill(0.15,m),fill(4.0,m),B)
end
function demand(N,rng,mode; D0=3.0e3, S=6)
    v=zeros(N.n)
    if mode==:single; v[rand(rng,1:N.n)]+=D0; v[rand(rng,1:N.n)]-=D0
    else; for _ in 1:S; v[rand(rng,1:N.n)]+=D0/S; v[rand(rng,1:N.n)]-=D0/S; end; end
    v.-=sum(v)/N.n; v
end
println("Table 4a — inner CG iters vs K (n=1500, γ=0.1):")
@printf("%-5s %-11s %-9s\n","K","single-OD","spread")
N=rand_net(1500)
for K in (2,4,8,12,16)
    r1=MersenneTwister(7); i1=DNLF.solve_sue(N,[demand(N,r1,:single) for _ in 1:K],0.1)[4]
    r2=MersenneTwister(7); i2=DNLF.solve_sue(N,[demand(N,r2,:spread) for _ in 1:K],0.1)[4]
    @printf("%-5d %-11d %-9d\n",K,i1,i2)
end
println("\nTable 4b — inner CG iters vs n (K=8, spread, γ=0.1):")
@printf("%-8s %-9s %-6s\n","n","m","iters")
for n in (1000,2000,4000,8000)
    Nn=rand_net(n); r=MersenneTwister(7); it=DNLF.solve_sue(Nn,[demand(Nn,r,:spread) for _ in 1:8],0.1)[4]
    @printf("%-8d %-9d %-6d\n",n,Nn.m,it)
end
println("\nγ-scan (honest degradation toward deterministic limit; n=1500, K=4, single-OD):")
@printf("%-9s %-6s\n","γ","iters")
for γ in (0.2,0.1,0.05,0.02,0.01)
    r=MersenneTwister(7); it=DNLF.solve_sue(N,[demand(N,r,:single) for _ in 1:4],γ)[4]
    @printf("%-9.2g %-6d\n",γ,it)
end
