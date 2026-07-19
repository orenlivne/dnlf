# Cleanest integration path: keep the loose continuation, but run the POLISH as full Newton by forcing
# newton_flow! to rebuild every step (refresh=0) -> its "frozen apply" becomes a true current-J solve.
# No new solver code, no PCG, no NLF changes. Verify robust across sizes AND seeds: reaches 1e-9, bounded
# steps/setups, ~10x faster than the frozen polish.
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

# loose continuation (frozen intermediate levels) + full-Newton polish (newton_flow! refresh=prf)
function loose_fullnewton(net,d; prf=0.0, pnmax=200)
    tolls=zeros(net.m); Bn=-net.B; bs=DNLF.approxchol_builder()
    H,SC,ST,setups,GG=Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0)
    x=zeros(net.n); tmean=sum(net.t0)/net.m; ctot=0
    for (i,fr) in enumerate(DNLF.DFRACS)
        H[]=nothing
        r=DNLF.newton_flow!(x,Bn,DNLF.smoothed_law(net,tolls,fr*tmean),d;inner=:multigrid,build_solver=bs,
            tol=3e-2,nmax=6,anderson=8,refresh=1e9,H=H,SC=SC,ST=ST,setups=setups,GG=GG); ctot+=r.steps
    end
    cset=setups[]
    pr=DNLF.newton_flow!(x,Bn,DNLF.rectified_law(net,tolls),d;inner=:multigrid,build_solver=bs,
        energy=DNLF.ue_energy(net,d,tolls),tol=1e-9,nmax=pnmax,anderson=8,refresh=prf,
        H=H,SC=SC,ST=ST,setups=setups,GG=GG)
    (cont=ctot, polish=pr.steps, setups=setups[], polsetups=setups[]-cset, rr=relres(net,d,x,tolls), conv=pr.converged)
end
net,d=rand_net(2000); loose_fullnewton(net,d)  # compile
@printf("%-6s %-6s %-9s %-8s %-9s %-8s %-9s\n","n","seed","cont+pol","setups","time(s)","conv?","rel.res")
for n in (4000, 8000, 16000), sd in (1,2)
    net,d=rand_net(n; seed=sd); t=@elapsed (R=loose_fullnewton(net,d))
    @printf("%-6d %-6d %d+%-7d %-8d %-9.1f %-8s %.0e\n", n, sd, R.cont,R.polish, R.setups, t, R.conv, R.rr)
end
