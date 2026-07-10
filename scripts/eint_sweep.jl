# Sensitivity of the loose intermediate tolerance ε_int (itol): total chord-Newton steps + wall-clock,
# final accuracy held tight (tol=1e-9). Shows the nested-iteration flat region (ε_int insensitivity).
using DNLF, SparseArrays, Random, Printf
function rand_net(n; deg=5, seed=1)
    rng=MersenneTwister(seed); E=Set{Tuple{Int,Int}}()
    for u in 1:n,_ in 1:deg; v=rand(rng,1:n); v==u&&continue; push!(E,(min(u,v),max(u,v))); end
    es=collect(E); ini=Int[];ter=Int[]; for (u,v) in es; push!(ini,u);push!(ter,v);push!(ini,v);push!(ter,u); end
    m=length(ini); B=sparse([ini;ter],[1:m;1:m],[fill(-1.0,m);fill(1.0,m)],n,m)
    nt=DNLF.DirectedNetwork(n,m,ini,ter,1000 .*(0.5 .+rand(rng,m)),1 .+rand(rng,m),fill(0.15,m),fill(4.0,m),B)
    dd=zeros(n); for u in randperm(rng,n)[1:n÷8];dd[u]+=1;end; for v in randperm(rng,n)[1:n÷8];dd[v]-=1;end
    dd.-=sum(dd)/n; dd.*=(3000.0*n/(sum(abs,dd)/2)); nt,dd
end
net,d=rand_net(8000)
DNLF.solve_flow(net,d,zeros(net.m); inner=:approxchol, itol=3e-2, inmax=40)  # compile
@printf("%-8s %-8s %-9s %-9s\n","ε_int","steps","time(s)","setups")
for itol in (3e-1,1e-1,3e-2,1e-2,3e-3,1e-3)
    Hp=(Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    t=@elapsed ((_,_,s)=DNLF.solve_flow(net,d,zeros(net.m); inner=:approxchol, itol=itol, inmax=40, tol=1e-9, Hpack=Hp))
    @printf("%-8.0e %-8d %-9.1f %-9d\n", itol, s, t, Hp[4][]); flush(stdout)
end
