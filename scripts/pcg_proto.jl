# PROTOTYPE (no solver changes yet): keep the cheap loose delta-continuation, but replace the STALLING
# frozen-chord tight end-solve with a Newton loop whose inner solve is PCG on the CURRENT Jacobian,
# preconditioned by the (frozen) approxChol factorization. Verify: bounded Newton steps, stable low setups,
# true 1e-9, faster than the current frozen mode -- across sizes. Compare A) current loose vs B) loose+PCG.
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

# preconditioned CG on singular SPD Laplacian J x = rhs (rhs zero-mean); P(z) ~ J^{-1} z
function pcg(Jmv, rhs, P; tol=1e-2, maxit=400)
    x=zeros(length(rhs)); r=zm!(copy(rhs)); z=zm!(P(r)); p=copy(z); rz=dot(r,z); bn=max(norm(rhs),1e-30); its=0
    for it in 1:maxit
        its=it; Jp=zm!(Jmv(p)); pJp=dot(p,Jp); pJp<=0 && break
        α=rz/pJp; x.+=α.*p; r.-=α.*Jp; zm!(r)
        norm(r)<=tol*bn && break
        z=zm!(P(r)); rzn=dot(r,z); β=rzn/rz; rz=rzn; p.=z.+β.*p
    end
    zm!(x), its
end

# Newton with PCG inner solve, approxChol preconditioner frozen (rebuilt only if PCG struggles/stalls)
function newton_pcg!(x, N, tolls, d; tol=1e-9, nmax=60, cgtol=1e-2, refresh=0.05)
    Bn=-N.B; law! = DNLF.rectified_law(N,tolls); energy=DNLF.ue_energy(N,d,tolls); bs=DNLF.approxchol_builder()
    f=zeros(N.m); dρ=zeros(N.m); bnrm=max(norm(d),1.0); P=nothing; nr_prev=Inf; setups=0; cgtot=0; nsteps=0
    for it in 1:nmax
        nsteps=it; g=Bn'*x; law!(f,dρ,g); r=Bn*f.-d; nr=norm(r)
        nr<tol*bnrm && break
        dρf=max.(dρ, 1e-12*maximum(dρ))
        if P===nothing || nr>refresh*nr_prev
            SC=maximum(dρf); P=bs(lapclean((Bn*Diagonal(dρf)*Bn')./SC)); setups+=1
        end
        nr_prev=nr
        Jmv = v -> Bn*(dρf .* (Bn'*v))
        δ,cgits = pcg(Jmv, -Vector(r), P; tol=cgtol); cgtot+=cgits
        E0=energy(x); gTd=dot(Vector(r),δ); τ=1.0; ok=false
        for _ in 1:60
            xt=zm!(x.+τ.*δ)
            if energy(xt)<=E0+1e-4*τ*gTd; copyto!(x,xt); ok=true; break; end
            τ*=0.5
        end
        ok || break
    end
    (nsteps=nsteps, setups=setups, cgtot=cgtot)
end

# B) loose continuation (13 smoothed levels, cheap) then PCG polish on exact law
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
    (contsteps=ctot, newton=pr.nsteps, cg=pr.cgtot, setups=setups[]+pr.setups, rr=relres(net,d,x,tolls))
end
# A) current frozen loose mode
function loose_frozen(net,d)
    Hp=(Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    (x,f,s,su)=DNLF.solve_flow(net,d,zeros(net.m); inner=:approxchol, itol=3e-2, inmax=6, tol=1e-9, Hpack=Hp)
    (steps=s, setups=su, rr=relres(net,d,x,zeros(net.m)))
end

net,d=rand_net(2000); loose_frozen(net,d); loose_plus_pcg(net,d)  # compile
@printf("%-6s %-8s | %-22s | %-34s\n","n","m","A) FROZEN steps/setups/s/res","B) LOOSE+PCG cont+newton(cg)/setups/s/res")
for n in (4000, 8000, 16000)
    net,d=rand_net(n)
    tA=@elapsed (A=loose_frozen(net,d)); tB=@elapsed (B=loose_plus_pcg(net,d))
    @printf("%-6d %-8d | %4d /%3d /%6.1f /%.0e | %d+%d(cg%d) /%3d /%6.1f /%.0e  [%.1fx]\n",
        n, net.m, A.steps,A.setups,tA,A.rr, B.contsteps,B.newton,B.cg,B.setups,tB,B.rr, tA/tB)
end
