# #2: semismooth Newton polish for the exact dead-zone law. Key vs the stalling energy-Armijo polish:
#  - generalized Jacobian J = B diag(rho'_active) B' (rho'=0 on dead arcs, tiny floor for SPD), rebuilt each step
#  - inner solve J d = -F by PCG preconditioned with the fresh approxChol factorization (~1-2 CG iters)
#  - MERIT line search on psi = 1/2 ||F||^2 (the semismooth globalizer), not the energy
# Verify: reaches 1e-9 robustly across sizes x seeds, bounded steps, ~10x faster than frozen chord.
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
zm!(v)=(v.-=sum(v)/length(v); v)
lapclean(J)=(off=J-spdiagm(0=>diag(J)); off=(off+off')/2; dropzeros!(off); off+spdiagm(0=>-vec(sum(off,dims=2))))
function pcg(Jmv, rhs, P; tol=1e-3, maxit=200)
    x=zeros(length(rhs)); r=zm!(copy(rhs)); z=zm!(P(r)); p=copy(z); rz=dot(r,z); bn=max(norm(rhs),1e-30); its=0
    for it in 1:maxit
        its=it; Jp=zm!(Jmv(p)); pJp=dot(p,Jp); pJp<=0 && break
        α=rz/pJp; x.+=α.*p; r.-=α.*Jp; zm!(r); norm(r)<=tol*bn && break
        z=zm!(P(r)); rzn=dot(r,z); β=rzn/rz; rz=rzn; p.=z.+β.*p
    end
    zm!(x), its
end
Fval(Bn,N,tolls,d,x)=(f=similar(x,N.m); dρ=similar(f); g=Bn'*x; DNLF.rectified_law(N,tolls)(f,dρ,g); (Bn*f.-d, dρ, f))
function ssn_polish!(x, N, tolls, d; tol=1e-9, nmax=150, cgtol=1e-8)
    Bn=-N.B; bs=DNLF.approxchol_builder(); bnrm=max(norm(d),1.0); setups=0; cgtot=0; nsteps=0
    for it in 1:nmax
        nsteps=it
        F,dρ,_ = Fval(Bn,N,tolls,d,x); nr=norm(F)
        nr < tol*bnrm && break
        dρf=max.(dρ,1e-12*maximum(dρ)); SC=maximum(dρf)
        P=bs(lapclean((Bn*Diagonal(dρf)*Bn')./SC)); setups+=1
        Jmv = v -> Bn*(dρf .* (Bn'*v))
        δ,cg=pcg(Jmv,-Vector(F),P; tol=cgtol); cgtot+=cg
        ψ0=0.5*nr^2; τ=1.0; ok=false            # MERIT (1/2||F||^2) Armijo line search
        for _ in 1:50
            xt=zm!(x.+τ.*δ); Ft,_,_=Fval(Bn,N,tolls,d,xt)
            if 0.5*norm(Ft)^2 <= ψ0*(1-2e-4*τ); copyto!(x,xt); ok=true; break; end
            τ*=0.5
        end
        ok || break
    end
    (nsteps=nsteps, setups=setups, cg=cgtot)
end
function solve_ssn(net,d)
    tolls=zeros(net.m); Bn=-net.B; bs=DNLF.approxchol_builder()
    H,SC,ST,setups,GG=Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0)
    x=zeros(net.n); tmean=sum(net.t0)/net.m; ctot=0
    for (i,fr) in enumerate(DNLF.DFRACS)
        H[]=nothing
        r=DNLF.newton_flow!(x,Bn,DNLF.smoothed_law(net,tolls,fr*tmean),d;inner=:multigrid,build_solver=bs,
            tol=3e-2,nmax=6,anderson=8,refresh=1e9,H=H,SC=SC,ST=ST,setups=setups,GG=GG); ctot+=r.steps
    end
    cs=setups[]; pr=ssn_polish!(x,net,tolls,d)
    F,_,_=Fval(Bn,net,tolls,d,x)
    (cont=ctot, polish=pr.nsteps, setups=setups[]+pr.setups, rr=norm(F)/max(norm(d),1.0))
end
net,d=rand_net(2000); solve_ssn(net,d)  # compile
@printf("%-6s %-5s %-9s %-7s %-9s %-9s\n","n","seed","cont+pol","setups","time(s)","rel.res")
for n in (32000, 64000), sd in (1,)
    net,d=rand_net(n; seed=sd); t=@elapsed (R=solve_ssn(net,d))
    conv = R.rr < 1e-8 ? "OK" : "**"
    @printf("%-6d %-5d %d+%-7d %-7d %-9.1f %.0e %s\n", n, sd, R.cont,R.polish, R.setups, t, R.rr, conv)
end
