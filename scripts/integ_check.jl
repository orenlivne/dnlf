# Confirm the INTEGRATED solve_flow (default polish=:ssn) gives the speedup + convergence vs legacy :chord,
# through the real code path, for both near-linear engines.
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
rr(net,d,f)=min(norm(net.B*f.-d),norm(net.B*f.+d))/norm(d)
sv(net,d;kw...)=(Hp=(Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0));
    (x,f,s,su)=DNLF.solve_flow(net,d,zeros(net.m); itol=3e-2, inmax=6, tol=1e-9, Hpack=Hp, kw...); (s=s,su=su,rr=rr(net,d,f)))
net,d=rand_net(2000); sv(net,d;inner=:approxchol); sv(net,d;inner=:approxchol,polish=:chord)  # compile
@printf("%-6s %-10s %-8s %-9s %-9s %-9s\n","n","engine","polish","steps/setups","time(s)","rel.res")
for n in (8000, 16000), eng in (:approxchol, :multigrid)
    net,d=rand_net(n)
    tS=@elapsed (S=sv(net,d;inner=eng));            @printf("%-6d %-10s %-8s %d/%-6d %-9.1f %.0e\n", n, eng, "ssn",   S.s,S.su,tS,S.rr)
    tC=@elapsed (C=sv(net,d;inner=eng,polish=:chord)); @printf("%-6d %-10s %-8s %d/%-6d %-9.1f %.0e  (%.1fx)\n", n, eng, "chord", C.s,C.su,tC,C.rr, tC/tS)
end
