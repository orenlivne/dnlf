# Is the frozen-vs-refresh finding size-independent (the property the paper hinges on)? Compare, across sizes,
# the current frozen tight end-game vs a refreshed one (endrefresh=0.5). Report steps/setups/time/resid + speedup.
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
function run(net,d; itol=3e-2,inmax=6,tol=1e-9,nmax=300,anderson=8,endrefresh=1e9)
    tolls=zeros(net.m); Bn=-net.B; bs=DNLF.approxchol_builder(); isym=:multigrid
    H,SC,ST,setups,GG=Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0)
    x=zeros(net.n); tmean=sum(net.t0)/net.m; dfracs=DNLF.DFRACS; tot=0
    for (i,fr) in enumerate(dfracs)
        H[]=nothing; last=(i==length(dfracs)); ltol=last ? tol : itol; lnmax=last ? nmax : inmax; rf=last ? endrefresh : 1e9
        r=DNLF.newton_flow!(x,Bn,DNLF.smoothed_law(net,tolls,fr*tmean),d;inner=isym,build_solver=bs,
            tol=ltol,nmax=lnmax,anderson=anderson,refresh=rf,H=H,SC=SC,ST=ST,setups=setups,GG=GG); tot+=r.steps
    end
    r=DNLF.newton_flow!(x,Bn,DNLF.rectified_law(net,tolls),d;inner=isym,build_solver=bs,
        energy=DNLF.ue_energy(net,d,tolls),tol=tol,nmax=nmax,anderson=anderson,refresh=endrefresh,
        H=H,SC=SC,ST=ST,setups=setups,GG=GG); tot+=r.steps
    (steps=tot,setups=setups[],rr=relres(net,d,x,tolls))
end
net,d=rand_net(2000); run(net,d); run(net,d;endrefresh=0.5)  # compile
@printf("%-6s %-8s | %-22s | %-22s | %-7s\n","n","m","FROZEN (steps/setups/s/res)","REFRESH0.5 (steps/setups/s/res)","speedup")
for n in (4000, 8000, 16000)
    net,d=rand_net(n)
    tf=@elapsed (F=run(net,d;endrefresh=1e9)); tr=@elapsed (R=run(net,d;endrefresh=0.5))
    @printf("%-6d %-8d | %4d /%3d /%6.1f /%.0e | %4d /%3d /%6.1f /%.0e | %.1fx\n",
            n, net.m, F.steps,F.setups,tf,F.rr, R.steps,R.setups,tr,R.rr, tf/tr)
end
