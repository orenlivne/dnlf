# Coupling-aware (Brandt distribution-matrix) preconditioner experiment for the multicommodity Newton
# system. Confirms: (1) the reduced-potential-space mean/deviation preconditioner FAILS (variable
# coupling direction); (2) the FLOW-SPACE / saddle realization (exact per-arc Sherman-Morrison coupling +
# aggregate-congestion Laplacian on the mean mode) CONVERGES γ-boundedly, but does NOT beat the simple
# block-diagonal preconditioner at or near equilibrium. Hence block-diagonal is retained (paper §6).
#   Usage:  julia --project=. scripts/coupling_precond_mc.jl
# system [H, A^T; A, 0][df;dφ]=[0;-b] (equivalent to reduced J dφ = b) by preconditioned MINRES with block
# preconditioner diag(H_exact, Ŝ): H applied EXACTLY per-arc (Sherman-Morrison, O(K) — the constant-1 coupling
# handled exactly in flow space), Ŝ = block-diagonal per-commodity Laplacians on the potential block.
# Compare CG iters (block-diagonal reduced PCG) vs MINRES iters (saddle) vs γ. Falsifiable: MINRES γ-flat.
using DNLF, Laplacians, LinearAlgebra, SparseArrays, Random, Printf, Statistics
const D=DNLF
function rand_net(n; deg=5, seed=1)
    rng=MersenneTwister(seed); E=Set{Tuple{Int,Int}}()
    for u in 1:n,_ in 1:deg; v=rand(rng,1:n); v==u&&continue; push!(E,(min(u,v),max(u,v))); end
    es=collect(E); ini=Int[];ter=Int[]; for (u,v) in es; push!(ini,u);push!(ter,v);push!(ini,v);push!(ter,u); end
    m=length(ini); B=sparse([ini;ter],[1:m;1:m],[fill(-1.0,m);fill(1.0,m)],n,m)
    D.DirectedNetwork(n,m,ini,ter,1000 .*(0.5 .+rand(rng,m)),1 .+rand(rng,m),fill(0.15,m),fill(4.0,m),B)
end
function demand(N,rng,mode; D0=3.0e3, S=6)
    v=zeros(N.n); if mode==:single; v[rand(rng,1:N.n)]+=D0; v[rand(rng,1:N.n)]-=D0
    else; for _ in 1:S; v[rand(rng,1:N.n)]+=D0/S; v[rand(rng,1:N.n)]-=D0/S; end; end; v.-=sum(v)/N.n; v
end
function ac(L)
    A=-(L-spdiagm(0=>diag(L))); dropzeros!(A)
    for p in (nothing, Laplacians.ApproxCholParams(:deg))
        try; f= p===nothing ? Laplacians.approxchol_lap(A;tol=1e-10) : Laplacians.approxchol_lap(A;tol=1e-10,params=p)
            return r->(x=f(r.-sum(r)/length(r)); x.-sum(x)/length(x)); catch; end; end
    n=size(L,1); k=2:n; F=ldlt(Symmetric(L[k,k]+1e-10*maximum(diag(L))*I)); r->(x=zeros(n); x[k]=F\(r[k].-sum(r)/n); x.-=sum(x)/n; x)
end
lap(B,w)=B*spdiagm(0=>max.(w,1e-10*maximum(w)))*B'

# Saddle solve of reduced J dφ = b via preconditioned MINRES. Returns (dφ, iters).
function saddle_minres(N,fk,fa,γ,b; tol=1e-7, maxit=400)
    m,n,K=N.m,N.n,size(fk,2); tp=[D.dcost(N,a,fa[a]) for a in 1:m]; Bt=sparse(N.B')
    # exact per-arc H^{-1}: H_a=t'11^T+γ diag(1/f^k); H_a^{-1}v = (f/γ).*v - t'/(1+t'fa/γ)·(f/γ)·((f/γ)·v)
    Hinv(V)=begin  # V is m×K
        FdV=(fk./γ); s=vec(sum(FdV.*V,dims=2)); coef=tp./(1 .+ tp.*fa./γ)
        (fk./γ).*V .- (coef.*s).*(fk./γ) end
    Sinv=D.mc_blk_precond(N,fk,fa,γ)                                   # Schur precond on potential block
    # saddle apply K[df;dφ] = [H df + A^T dφ ; A df]  (df: m×K, dφ: n×K)
    Kap(df,dφ)=begin
        s=vec(sum(df,dims=2)); Hdf=(tp.*s).*ones(1,K).+γ.*(df./fk)     # H df per-arc
        AtP=hcat([Bt*dφ[:,k] for k in 1:K]...)                         # A^T dφ (m×K)
        Adf=hcat([N.B*df[:,k] for k in 1:K]...)                        # A df (n×K)
        (Hdf.+AtP, Adf) end
    Pinv(rf,rc)=(Hinv(rf), Sinv(rc))                                   # block-diag preconditioner
    ip((a1,a2),(b1,b2))=sum(a1.*b1)+sum(a2.*b2)
    # RHS = [0; -b]
    rf=zeros(m,K); rc=-b
    # preconditioned MINRES (Paige–Saunders), preconditioner SPD
    xf=zeros(m,K); xc=zeros(n,K)
    v1f,v1c=copy(rf),copy(rc); z1f,z1c=Pinv(v1f,v1c); β1=sqrt(ip((v1f,v1c),(z1f,z1c))); (β1<1e-30||!isfinite(β1)) && return xc,maxit  # degenerate/NaN RHS = breakdown, not "0 iters"
    v1f./=β1; v1c./=β1; z1f./=β1; z1c./=β1
    v0f,v0c=zeros(m,K),zeros(n,K); β=β1; η=β1
    c0,c1=1.0,1.0; s0,s1=0.0,0.0
    w0f,w0c=zeros(m,K),zeros(n,K); w1f,w1c=zeros(m,K),zeros(n,K)
    its=maxit
    for it in 1:maxit
        # Lanczos step on preconditioned operator: p = K z
        pf,pc=Kap(z1f,z1c); α=ip((pf,pc),(z1f,z1c))
        pf.-=α.*v1f.+β.*v0f; pc.-=α.*v1c.+β.*v0c
        zf,zc=Pinv(pf,pc); βn=sqrt(max(ip((pf,pc),(zf,zc)),0.0))
        # QR
        δ=c1*α-c0*s1*β; γr=sqrt(δ^2+βn^2); ρ1=s1*α+c0*c1*β; ρ2=s0*β
        c=γr<1e-30 ? 1.0 : δ/γr; s=γr<1e-30 ? 0.0 : βn/γr
        wf=(z1f.-ρ1.*w1f.-ρ2.*w0f)./(γr<1e-30 ? 1.0 : γr); wc=(z1c.-ρ1.*w1c.-ρ2.*w0c)./(γr<1e-30 ? 1.0 : γr)
        xf.+=(c*η).*wf; xc.+=(c*η).*wc; η=-s*η
        (abs(η)/β1<tol) && (its=it; break)
        v0f,v0c=v1f,v1c; w0f,w0c=w1f,w1c; w1f,w1c=wf,wc
        if βn>1e-30; v1f,v1c=pf./βn,pc./βn; z1f,z1c=zf./βn,zc./βn; else; its=it; break; end
        β=βn; c0,c1=c1,c; s0,s1=s1,s
    end
    all(isfinite,xc) ? (xc,its) : (xc,maxit)             # non-finite iterate = breakdown (report as maxit)
end

N=rand_net(1500)
for mode in (:spread,:single)
    @printf("\n=== %s demand, K=4: block-PCG vs saddle-MINRES vs γ ===\n%-7s %-10s %-12s\n","$mode","γ","block-PCG","saddle-MINRES")
    for γ in (0.2,0.1,0.05,0.02,0.01)
        rng=MersenneTwister(7); dk=[demand(N,rng,mode) for _ in 1:4]
        _,fk,fa,_=D.solve_sue(N,dk,γ)
        e=randn(N.n,4); for k in 1:4; e[:,k].-=sum(e[:,k])/N.n; end; b=D.mc_jac(N,fk,fa,γ,e)
        # block-diagonal reduced PCG
        Minv=D.mc_blk_precond(N,fk,fa,γ); nb=norm(b); x=zeros(N.n,4); r=b.-D.mc_jac(N,fk,fa,γ,x); z=Minv(r); p=copy(z); rz=sum(r.*z); ib=400
        for it in 1:400; Jp=D.mc_jac(N,fk,fa,γ,p); a=rz/sum(p.*Jp); x.+=a.*p; r.-=a.*Jp; norm(r)/nb<1e-7 && (ib=it;break); z=Minv(r); rz2=sum(r.*z); p.=z.+(rz2/rz).*p; rz=rz2; end
        _,is=saddle_minres(N,fk,fa,γ,b)
        @printf("%-7.2g %-10d %-12d\n",γ,ib,is)
    end
end

# DECISIVE: far from equilibrium (γ=0.02, single-OD) — the cold-transient regime where block-diagonal degrades
println("\n=== FAR FROM EQUILIBRIUM (γ=0.02, single-OD): iters vs ‖perturbation‖ ===")
@printf("%-8s %-10s %-12s\n","α","block-PCG","saddle-MINRES")
let γ=0.02, rng=MersenneTwister(7)
    dk=[demand(N,rng,:single) for _ in 1:4]; φ,_,_,_=D.solve_sue(N,dk,γ)
    for α in (0.0,0.5,1.0,2.0,4.0)
        φp=φ.+α.*randn(size(φ)); fkp,fap=D.mc_flows(N,φp,γ)
        e=randn(N.n,4); for k in 1:4; e[:,k].-=sum(e[:,k])/N.n; end; b=D.mc_jac(N,fkp,fap,γ,e)
        Minv=D.mc_blk_precond(N,fkp,fap,γ); nb=norm(b); x=zeros(N.n,4); r=b.-D.mc_jac(N,fkp,fap,γ,x); z=Minv(r); p=copy(z); rz=sum(r.*z); ib=400
        for it in 1:400; Jp=D.mc_jac(N,fkp,fap,γ,p); a=rz/sum(p.*Jp); x.+=a.*p; r.-=a.*Jp; norm(r)/nb<1e-7&&(ib=it;break); z=Minv(r); rz2=sum(r.*z); p.=z.+(rz2/rz).*p; rz=rz2; end
        _,is=saddle_minres(N,fkp,fap,γ,b)
        @printf("%-8.1f %-10d %-12d\n",α,ib,is)
    end
end
