# Option C: shrink the semismooth-polish step count (hence the ~100 factorizations and the m^1.2) by giving it a
# TIGHT warm start. Solve the last `tighten` SMOOTH continuation levels with full Newton (quadratic, few steps)
# to `smooth_tol`, so the active set is nearly settled before the exact-law polish. Measure steps/setups/time and
# FIT the exponent across sizes (isolated — no parallel jobs).
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
rrf(net,d,x,tolls)=(f=[DNLF.rho(net,a,(x[net.ini[a]]-x[net.ter[a]])-tolls[a])[1] for a in 1:net.m]; norm(net.B*f.+d)/norm(d))
function solve_C(net,d; tighten=2, smooth_tol=1e-7)
    tolls=zeros(net.m); Bn=-net.B; bs=DNLF.approxchol_builder()
    H,SC,ST,setups,GG=Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0)
    x=zeros(net.n); tmean=sum(net.t0)/net.m; L=length(DNLF.DFRACS); ctot=0; polish=0
    for (i,fr) in enumerate(DNLF.DFRACS)
        if i <= L - tighten
            H[]=nothing
            r=DNLF.newton_flow!(x,Bn,DNLF.smoothed_law(net,tolls,fr*tmean),d;inner=:multigrid,build_solver=bs,
                tol=3e-2,nmax=6,anderson=8,refresh=1e9,H=H,SC=SC,ST=ST,setups=setups,GG=GG); ctot+=r.steps
        else
            ctot += DNLF.ssn_polish!(x,Bn,DNLF.smoothed_law(net,tolls,fr*tmean),d; build_solver=DNLF.approxchol_builder(), tol=smooth_tol, setups=setups)
        end
    end
    polish = DNLF.ssn_polish!(x,Bn,DNLF.rectified_law(net,tolls),d; build_solver=DNLF.approxchol_builder(), tol=1e-9, setups=setups)
    (steps=ctot+polish, polish=polish, setups=setups[], rr=rrf(net,d,x,tolls))
end
net,d=rand_net(2000); solve_C(net,d)  # compile
NS=(4000,8000,16000,32000,64000); ms=Float64[]; ts=Float64[]
@printf("%-6s %-8s %-9s %-7s %-9s %-9s\n","n","m","polish","setups","time(s)","rel.res")
for n in NS
    net,d=rand_net(n); t=@elapsed (R=solve_C(net,d))
    push!(ms,net.m); push!(ts,t)
    @printf("%-6d %-8d %-9d %-7d %-9.1f %.0e\n", n, net.m, R.polish, R.setups, t, R.rr)
end
x=log10.(ms); y=log10.(ts); xm=sum(x)/length(x); ym=sum(y)/length(y)
@printf("C exponent: t ~ m^%.2f\n", sum((x.-xm).*(y.-ym))/sum((x.-xm).^2))
