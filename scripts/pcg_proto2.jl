# PCG polish v2: FREEZE the approxChol preconditioner (build once), let PCG absorb Jacobian drift; rebuild
# only on a genuine stall (line-search failure or PCG hitting its cap). Goal: setups back to ~14, all sizes
# reach true 1e-9, ~10x speedup preserved. Verify across sizes.
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
zm!(v)=(v.-=sum(v)/length(v); v)
lapclean(J)=(off=J-spdiagm(0=>diag(J)); off=(off+off')/2; dropzeros!(off); off+spdiagm(0=>-vec(sum(off,dims=2))))
function pcg(Jmv, rhs, P; tol=1e-2, maxit=500)
    x=zeros(length(rhs)); r=zm!(copy(rhs)); z=zm!(P(r)); p=copy(z); rz=dot(r,z); bn=max(norm(rhs),1e-30); its=0
    for it in 1:maxit
        its=it; Jp=zm!(Jmv(p)); pJp=dot(p,Jp); pJp<=0 && break
        α=rz/pJp; x.+=α.*p; r.-=α.*Jp; zm!(r); norm(r)<=tol*bn && break
        z=zm!(P(r)); rzn=dot(r,z); β=rzn/rz; rz=rzn; p.=z.+β.*p
    end
    zm!(x), its
end
function newton_pcg!(x, N, tolls, d; tol=1e-9, nmax=150, cgtol=1e-2, cgmax=500)
    Bn=-N.B; law! = DNLF.rectified_law(N,tolls); energy=DNLF.ue_energy(N,d,tolls); bs=DNLF.approxchol_builder()
    f=zeros(N.m); dρ=zeros(N.m); bnrm=max(norm(d),1.0); setups=0; cgtot=0; nsteps=0
    g=Bn'*x; law!(f,dρ,g); dρf=max.(dρ,1e-12*maximum(dρ))          # build the frozen preconditioner ONCE
    mkP()=(SC=maximum(dρf); bs(lapclean((Bn*Diagonal(dρf)*Bn')./SC)))
    P=mkP(); setups+=1
    for it in 1:nmax
        nsteps=it; g=Bn'*x; law!(f,dρ,g); r=Bn*f.-d; nr=norm(r)
        nr<tol*bnrm && break
        dρf=max.(dρ,1e-12*maximum(dρ)); Jmv = v -> Bn*(dρf .* (Bn'*v))
        δ,cgits=pcg(Jmv,-Vector(r),P; tol=cgtol,maxit=cgmax); cgtot+=cgits
        E0=energy(x); gTd=dot(Vector(r),δ); τ=1.0; ok=false
        for _ in 1:60; xt=zm!(x.+τ.*δ); if energy(xt)<=E0+1e-4*τ*gTd; copyto!(x,xt); ok=true; break; end; τ*=0.5; end
        if !ok || cgits>=cgmax                                     # genuine stall: refresh preconditioner once, retry
            P=mkP(); setups+=1
            δ,cgits=pcg(Jmv,-Vector(r),P; tol=cgtol,maxit=cgmax); cgtot+=cgits
            E0=energy(x); gTd=dot(Vector(r),δ); τ=1.0; ok=false
            for _ in 1:60; xt=zm!(x.+τ.*δ); if energy(xt)<=E0+1e-4*τ*gTd; copyto!(x,xt); ok=true; break; end; τ*=0.5; end
            ok || break
        end
    end
    (nsteps=nsteps, setups=setups, cgtot=cgtot)
end
function loose_plus_pcg(net,d)
    tolls=zeros(net.m); Bn=-net.B; bs=DNLF.approxchol_builder()
    H,SC,ST,setups,GG=Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0)
    x=zeros(net.n); tmean=sum(net.t0)/net.m; ctot=0
    for (i,fr) in enumerate(DNLF.DFRACS)
        H[]=nothing
        r=DNLF.newton_flow!(x,Bn,DNLF.smoothed_law(net,tolls,fr*tmean),d;inner=:multigrid,build_solver=bs,
            tol=3e-2,nmax=6,anderson=8,refresh=1e9,H=H,SC=SC,ST=ST,setups=setups,GG=GG); ctot+=r.steps
    end
    pr=newton_pcg!(x,net,tolls,d)
    (cont=ctot, newton=pr.nsteps, cg=pr.cgtot, setups=setups[]+pr.setups, rr=relres(net,d,x,tolls))
end
net,d=rand_net(2000); loose_plus_pcg(net,d)  # compile
@printf("%-6s %-8s %-8s %-8s %-8s %-9s %-9s\n","n","m","cont+nwt","cg","setups","time(s)","rel.res")
for n in (4000, 8000, 16000, 32000)
    net,d=rand_net(n); t=@elapsed (B=loose_plus_pcg(net,d))
    @printf("%-6d %-8d %d+%-6d %-8d %-8d %-9.1f %.0e\n", n, net.m, B.cont,B.newton, B.cg, B.setups, t, B.rr)
end
