# Diagnostic: where do the ~625 chord-Newton steps go? Replicates solve_flow's loose-mode continuation loop
# (itol=3e-2, inmax=6, tol=1e-9) but records steps PER delta-level and for the final polish separately.
using DNLF, SparseArrays, Random, Printf
function rand_net(n; deg=5, seed=1)
    rng=MersenneTwister(seed); E=Set{Tuple{Int,Int}}()
    for u in 1:n,_ in 1:deg; v=rand(rng,1:n); v==u&&continue; push!(E,(min(u,v),max(u,v))); end
    es=collect(E); ini=Int[];ter=Int[]; for (u,v) in es; push!(ini,u);push!(ter,v);push!(ini,v);push!(ter,u); end
    m=length(ini); B=sparse([ini;ter],[1:m;1:m],[fill(-1.0,m);fill(1.0,m)],n,m)
    nt=DNLF.DirectedNetwork(n,m,ini,ter,1000 .*(0.5 .+rand(rng,m)),1 .+rand(rng,m),fill(0.15,m),fill(4.0,m),B)
    dd=zeros(n); for u in randperm(rng,n)[1:n÷8];dd[u]+=1;end; for v in randperm(rng,n)[1:n÷8];dd[v]-=1;end
    dd.-=sum(dd)/n; dd.*=(3000.0*n/(sum(abs,dd)/2)); nt,dd
end

function breakdown(net, d; itol=3e-2, inmax=6, tol=1e-9, nmax=300, anderson=8)
    tolls = zeros(net.m); Bn = -net.B; bs = DNLF.approxchol_builder(); isym = :multigrid
    H, SC, ST, setups, GG = Ref{Any}(nothing), Ref(1.0), Ref(false), Ref(0), Ref(1.0)
    x = zeros(net.n); tmean = sum(net.t0)/net.m; dfracs = DNLF.DFRACS
    @printf("%-6s %-10s %-7s %-8s\n","level","delta","steps","setups")
    tot = 0
    for (i,fr) in enumerate(dfracs)
        H[] = nothing; last = (i==length(dfracs))
        ltol = last ? tol : itol; lnmax = last ? nmax : inmax
        res = DNLF.newton_flow!(x, Bn, DNLF.smoothed_law(net,tolls,fr*tmean), d; inner=isym,
                 build_solver=bs, tol=ltol, nmax=lnmax, anderson=anderson, refresh=1e9,
                 H=H, SC=SC, ST=ST, setups=setups, GG=GG)
        tot += res.steps
        @printf("%-6d %-10.4g %-7d %-8d%s\n", i, fr*tmean, res.steps, setups[], last ? "  <- final level (tight 1e-9)" : "")
    end
    res = DNLF.newton_flow!(x, Bn, DNLF.rectified_law(net,tolls), d; inner=isym, build_solver=bs,
             energy=DNLF.ue_energy(net,d,tolls), tol=tol, nmax=nmax, anderson=anderson, refresh=1e9,
             H=H, SC=SC, ST=ST, setups=setups, GG=GG)
    tot += res.steps
    @printf("%-6s %-10s %-7d %-8d  <- polish (exact law, tight 1e-9)\n","polish","", res.steps, setups[])
    @printf("TOTAL steps=%d  setups=%d\n", tot, setups[])
end

net,d = rand_net(8000)
DNLF.solve_flow(net,d,zeros(net.m); inner=:approxchol, itol=3e-2, inmax=6)  # compile
println("=== per-phase step breakdown (rand_net 8000, itol=3e-2, inmax=6, tol=1e-9) ===")
breakdown(net, d)
