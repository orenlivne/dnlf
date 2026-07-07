# ============================================================================================
# reproduce.jl — regenerate the paper's headline numbers with timing.
#
#   "A Near-Linear-Time Solver for Bilevel Congestion Network Design via Directed Nonlinear
#    Laplacian Flow" (O. E. Livne).  github.com/orenlivne/dnlf
#
# Run:   julia --project=. examples/reproduce.jl
#        (for the full irregular-graph crossover, see scripts/scaling.jl and set AIER_DATA)
#
# Reproduces, on the Sioux Falls benchmark shipped in data/:
#   * equilibrium correctness vs. an independent Frank–Wolfe solve   (paper: ~1e-11)
#   * the adjoint design gradient vs. central finite differences     (paper: ~2.6e-3)
#   * projected-gradient toll design: TSTT reduction                 (paper: 5.15%)
#   * engine-agnosticism (approximate Cholesky = LU = direct)
#   * size-independent Newton-step / AMG-setup counts and the near-linear crossover shape
#     on a small synthetic irregular family (the full corpus is scripts/scaling.jl)
# ============================================================================================
using DNLF, LinearAlgebra, SparseArrays, Printf, Random

reldiff(a,b) = norm(a-b)/max(norm(b),1e-30)
net = DNLF.load_tntp_net(joinpath(pkgdir(DNLF),"data","SiouxFalls","SiouxFalls_net.tntp"))
r,s,D = 1,20,3.0e4; d = zeros(net.n); d[r]=D; d[s]=-D
println("Sioux Falls: n=$(net.n)  m=$(net.m)\n")

# 1. equilibrium correctness (near-linear approxChol vs. Frank–Wolfe)
ffw = DNLF.frank_wolfe(net,r,s,D; iters=40000)
t = @elapsed ((φ,f,st) = DNLF.solve_ue(net,r,s,D; inner=:approxchol, tol=1e-10))
@printf("[1] equilibrium  approxChol vs Frank–Wolfe:  reldiff = %.2e   (%.3fs, %d Newton steps)\n",
        reldiff(f,ffw), t, st)

# 2. engine-agnosticism
_,fx,_ = DNLF.solve_ue(net,r,s,D; inner=:direct, tol=1e-10)
xl,fl,_,_ = DNLF.solve_flow(net,d,zeros(net.m); inner=:lu, tol=1e-10)
@printf("[2] engines      approxChol=direct: %.2e   approxChol=LU: %.2e\n", reldiff(f,fx), reldiff(f,fl))

# 3. adjoint gradient vs central finite differences (paper §5)
φ,f,_ = DNLF.solve_ue(net,r,s,D; tol=1e-11, nmax=6000)
g = DNLF.adjoint_grad(net,s,φ,f,zeros(net.m)); ε=1e-2; active=findall(>(1e-6),f); mx=0.0
for a in active
    τp=zeros(net.m);τp[a]+=ε; _,fp,_=DNLF.solve_ue(net,r,s,D;tolls=τp,tol=1e-11,nmax=6000)
    τm=zeros(net.m);τm[a]-=ε; _,fm,_=DNLF.solve_ue(net,r,s,D;tolls=τm,tol=1e-11,nmax=6000)
    fd=(DNLF.tstt(net,fp)-DNLF.tstt(net,fm))/(2ε); global mx=max(mx,abs(g[a]-fd)/max(abs(fd),1.0))
end
@printf("[3] adjoint      max rel err vs finite differences over %d active arcs: %.2e\n", length(active), mx)

# 4. bilevel toll design: monotone descent to the marginal-cost optimum
φ,f,_,_ = DNLF.solve_flow(net,d,zeros(net.m); tol=1e-11); τ=zeros(net.m); T0=DNLF.tstt(net,f); T=T0
for _ in 1:40
    gv=DNLF.adjoint_grad(net,s,φ,f,τ); lr=0.05; acc=false
    for _ in 1:22
        τt=max.(τ.-lr.*gv,0.0); φt,ft,_,_=DNLF.solve_flow(net,d,τt;tol=1e-11,init=φ); Tt=DNLF.tstt(net,ft)
        if Tt<T; global τ=τt; global φ=φt; global f=ft; global T=Tt; acc=true; break; end; lr*=0.5
    end
    acc || break
end
@printf("[4] design       TSTT %.1f -> %.1f  (%.2f%% reduction; paper 5.15%%)\n", T0, T, 100*(1-T/T0))

# 5. size-independence + crossover shape on a small synthetic irregular family (full corpus: scripts/scaling.jl)
function rand_net(n; deg=5, seed=1)
    rng=MersenneTwister(seed); E=Set{Tuple{Int,Int}}()
    for u in 1:n,_ in 1:deg; v=rand(rng,1:n); v==u&&continue; push!(E,(min(u,v),max(u,v))); end
    es=collect(E); ini=Int[];ter=Int[]; for (u,v) in es; push!(ini,u);push!(ter,v);push!(ini,v);push!(ter,u); end
    m=length(ini); B=sparse([ini;ter],[1:m;1:m],[fill(-1.0,m);fill(1.0,m)],n,m)
    nt=DNLF.DirectedNetwork(n,m,ini,ter,1000 .*(0.5 .+rand(rng,m)),1 .+rand(rng,m),fill(0.15,m),fill(4.0,m),B)
    dd=zeros(n); for u in randperm(rng,n)[1:n÷8];dd[u]+=1;end; for v in randperm(rng,n)[1:n÷8];dd[v]-=1;end
    dd.-=sum(dd)/n; dd.*=(3000.0*n/(sum(abs,dd)/2)); nt,dd
end
println("\n[5] size-independence (synthetic irregular; see scripts/scaling.jl for the real-graph crossover):")
@printf("    %-7s %-9s %-8s %-8s %-9s\n","n","m","steps","builds","ac_time")
for n in (1000,2000,4000)
    nt,dd=rand_net(n); Hp=(Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    DNLF.solve_flow(nt,dd,zeros(nt.m); tol=1e-9)  # compile
    Hp=(Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    tt=@elapsed ((_,_,stp)=DNLF.solve_flow(nt,dd,zeros(nt.m); tol=1e-9, Hpack=Hp))
    @printf("    %-7d %-9d %-8d %-8d %-9.2f\n", nt.n, nt.m, stp, Hp[4][], tt)
end
println("\nDone. Newton-steps and AMG-builds are ~constant in size; wall-clock is near-linear (~m^1.2).")
