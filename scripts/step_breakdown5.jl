# Before writing PCG: does the solver's EXISTING accurate mode (itol=nothing, adaptive refresh every level)
# already give bounded steps + stable low setups + true 1e-9, faster than the stalling loose mode? Compare
# loose (current scaling config) vs accurate across sizes, engine=approxchol.
using DNLF, SparseArrays, Random, Printf, LinearAlgebra
function rand_net(n; deg=5, seed=1)
    rng=MersenneTwister(seed); E=Set{Tuple{Int,Int}}()
    for u in 1:n,_ in 1:deg; v=rand(rng,1:n); v==u&&continue; push!(E,(min(u,v),max(u,v))); end
    es=collect(E); ini=Int[];ter=Int[]; for (u,v) in es; push!(ini,u);push!(ter,v);push!(ini,v);push!(ter,u); end
    m=length(ini); B=sparse([ini;ter],[1:m;1:m],[fill(-1.0,m);fill(1.0,m)],n,m)
    nt=DNLF.DirectedNetwork(n,m,ini,ter,1000 .*(0.5 .+rand(rng,m)),1 .+rand(rng,m),fill(0.15,m),fill(4.0,m),B)
    dd=zeros(n); for u in randperm(rng,n)[1:n÷8];dd[u]+=1;end; for v in randperm(rng,n)[1:n÷8];dd[v]-=1;end
    dd.-=sum(dd)/n; dd.*=(3000.0*n/(sum(abs,dd)/2)); nt,dd
end
relres(N,d,x,tolls)=(f=[DNLF.rho(N,a,(x[N.ini[a]]-x[N.ter[a]])-tolls[a])[1] for a in 1:N.m]; norm(N.B*f .+ d)/norm(d))
solveit(net,d; kw...) = (Hp=(Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0));
    (x,f,steps,setups)=DNLF.solve_flow(net,d,zeros(net.m); inner=:approxchol, tol=1e-9, Hpack=Hp, kw...);
    (steps=steps,setups=setups,rr=relres(net,d,x,zeros(net.m))))
net,d=rand_net(2000); solveit(net,d; itol=3e-2, inmax=6); solveit(net,d)  # compile both
@printf("%-6s %-8s | %-24s | %-24s | %-7s\n","n","m","LOOSE (steps/setups/s/res)","ACCURATE (steps/setups/s/res)","speedup")
for n in (4000, 8000, 16000)
    net,d=rand_net(n)
    tL=@elapsed (L=solveit(net,d; itol=3e-2, inmax=6)); tA=@elapsed (A=solveit(net,d))  # accurate = itol default nothing
    @printf("%-6d %-8d | %4d /%3d /%6.1f /%.0e | %4d /%3d /%6.1f /%.0e | %.1fx\n",
            n, net.m, L.steps,L.setups,tL,L.rr, A.steps,A.setups,tA,A.rr, tL/tA)
end
