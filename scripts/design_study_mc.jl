# Multi-instance multicommodity logit-SUE congestion-design study (paper Table 5). Runs the near-linear
# bilevel design (γ-continuation SUE solve + adjoint projected-gradient tolling) across instances varying
# commodities K, demand level, dispersion γ, network size/seed; reports TSTT reduction, inner-iteration
# count, and the fraction of arcs tolled. Well-spread multi-OD demands (the realistic regime).
#   Usage:  julia --project=. scripts/design_study_mc.jl
using DNLF, LinearAlgebra, SparseArrays, Random, Printf

function rand_net(n; deg=5, seed=1)
    rng=MersenneTwister(seed); E=Set{Tuple{Int,Int}}()
    for u in 1:n,_ in 1:deg; v=rand(rng,1:n); v==u&&continue; push!(E,(min(u,v),max(u,v))); end
    es=collect(E); ini=Int[];ter=Int[]; for (u,v) in es; push!(ini,u);push!(ter,v);push!(ini,v);push!(ter,u); end
    m=length(ini); B=sparse([ini;ter],[1:m;1:m],[fill(-1.0,m);fill(1.0,m)],n,m)
    DNLF.DirectedNetwork(n,m,ini,ter,1000 .*(0.5 .+rand(rng,m)),1 .+rand(rng,m),fill(0.15,m),fill(4.0,m),B)
end
function spread_demand(N,rng,S,D)
    v=zeros(N.n); for _ in 1:S; v[rand(rng,1:N.n)]+=D; v[rand(rng,1:N.n)]-=D; end; v.-=sum(v)/N.n; v
end
function toll_design(N,dk,γ,fk,fa; steps=15)
    τ=zeros(N.m); T=DNLF.mc_tstt(N,fa)
    for _ in 1:steps
        g=DNLF.mc_adjoint(N,fk,fa,γ); lr=5e-4; acc=false
        for _ in 1:20
            τt=max.(τ.-lr.*g,0.0); _,fkt,fat,_=solve_sue(N,dk,γ; tolls=τt)
            if DNLF.mc_tstt(N,fat)<T; τ=τt; fk=fkt; fa=fat; T=DNLF.mc_tstt(N,fat); acc=true; break; end
            lr/=2
        end
        acc || break
    end
    τ,T
end

# instances: (n, seed, K, γ, S OD-pairs/commodity, D per pair)
insts = [(800,1,4,0.10,10,300.0),(800,1,8,0.10,10,300.0),(800,1,16,0.10,10,300.0),
         (800,1,8,0.15,10,300.0),(800,1,8,0.10,10,600.0),(800,2,8,0.10,8,400.0),
         (1200,1,8,0.10,12,300.0),(1600,1,8,0.10,12,300.0)]
@printf("%-6s %-3s %-5s %-4s %-11s %-9s %-6s %-8s\n","n","K","γ","m","TSTT-red%","its","time","tolled%")
DNLF.solve_sue(rand_net(300),[spread_demand(rand_net(300),MersenneTwister(1),8,300.0) for _ in 1:3],0.1) # compile
for (n,sd,K,γ,S,D) in insts
    N=rand_net(n;seed=sd); rng=MersenneTwister(100+sd)
    dk=[spread_demand(N,rng,S,D) for _ in 1:K]
    _,fk,fa,its=solve_sue(N,dk,γ); T0=DNLF.mc_tstt(N,fa)
    t=@elapsed ((τ,T)=toll_design(N,dk,γ,fk,fa))
    @printf("%-6d %-3d %-5.2g %-4d %-11.2f %-9d %-6.0fs %-8.1f\n",
            n,K,γ,N.m,100*(1-T/T0),its,t,100*count(>(1e-9),τ)/N.m)
end
