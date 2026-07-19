# =====================================================================================================
# STRONGER direct baseline for the crossover study (referee ask): supernodal Cholesky (CHOLMOD) under a
# METIS nested-dissection ordering — the fair direct incumbent for the SPD graph-Laplacian J — versus the
# unsymmetric UMFPACK/COLAMD LU used before, and versus the near-linear NLF engine (approxChol). Same
# synthetic random-degree-5 family, same generator/seed, same frozen-once-per-level discipline. Fits each
# engine's wall-clock exponent in-run and reports how far each direct factorization reaches before its
# time/RAM wall. Serial + uncontended (run alone).
#
#   Usage:  julia --project=. scripts/scaling_cholmod.jl
#   Emits:  per-size rows for NLF / UMFPACK-LU / CHOLMOD-METIS, fitted exponents, crossover, scaling_cholmod.csv
# =====================================================================================================
using DNLF, LinearAlgebra, SparseArrays, Printf, Random

function rand_net(n; deg=5, seed=1)                       # SAME generator/seed as scaling_synthetic.jl
    rng=MersenneTwister(seed); E=Set{Tuple{Int,Int}}()
    for u in 1:n,_ in 1:deg; v=rand(rng,1:n); v==u&&continue; push!(E,(min(u,v),max(u,v))); end
    es=collect(E); ini=Int[];ter=Int[]; for (u,v) in es; push!(ini,u);push!(ter,v);push!(ini,v);push!(ter,u); end
    m=length(ini); B=sparse([ini;ter],[1:m;1:m],[fill(-1.0,m);fill(1.0,m)],n,m)
    nt=DNLF.DirectedNetwork(n,m,ini,ter,1000 .*(0.5 .+rand(rng,m)),1 .+rand(rng,m),fill(0.15,m),fill(4.0,m),B)
    dd=zeros(n); for u in randperm(rng,n)[1:n÷8];dd[u]+=1;end; for v in randperm(rng,n)[1:n÷8];dd[v]-=1;end
    dd.-=sum(dd)/n; dd.*=(3000.0*n/(sum(abs,dd)/2)); nt,dd
end
solve(net,d;inner) = DNLF.solve_flow(net,d,zeros(net.m); inner=inner, itol=3e-2, inmax=6)
function loglogfit(ms, ts)
    x = log10.(ms); y = log10.(ts); xm = sum(x)/length(x); ym = sum(y)/length(y)
    b = sum((x .- xm).*(y .- ym)) / sum((x .- xm).^2); b, ym - b*xm
end

# time one engine on (net,d) with a wall cap; returns time (s) or NaN if it threw / exceeded the cap
function timed(net,d,inner; cap=Inf)
    try
        solve(net,d; inner=inner)                          # compile / warm
        t0=time(); solve(net,d; inner=inner); t=time()-t0
        return t > cap ? NaN : t
    catch e
        @printf("    [%s failed at m=%d: %s]\n", inner, net.m, sprint(showerror,e)); flush(stdout)
        return NaN
    end
end

SIZES  = (1000, 2000, 4000, 8000, 16000, 32000, 64000)     # push both direct solvers to their wall
LUCAP  = 300.0                                              # per-solve wall cap (s) for the direct baselines
CHCAP  = 300.0
ms=Float64[]; tac=Float64[]
mlu=Float64[]; tlu=Float64[]; mch=Float64[]; tch=Float64[]
@printf("%-7s %-9s | %-10s | %-12s | %-12s\n","n","m","NLF(AC)","UMFPACK-LU","CHOLMOD-METIS"); flush(stdout)
for n in SIZES
    net,d = rand_net(n)
    ta = timed(net,d,:approxchol)
    isnan(ta) || (push!(ms,net.m); push!(tac,ta))
    tl = n<=32000 ? timed(net,d,:lu;      cap=LUCAP) : NaN   # skip LU past 32k (known RAM wall) to save time
    isnan(tl) || (push!(mlu,net.m); push!(tlu,tl))
    tc = timed(net,d,:cholmod; cap=CHCAP)
    isnan(tc) || (push!(mch,net.m); push!(tch,tc))
    @printf("%-7d %-9d | %-10.2f | %-12s | %-12s\n", net.n, net.m, ta,
            isnan(tl) ? "-" : @sprintf("%.2f",tl), isnan(tc) ? "-" : @sprintf("%.2f",tc)); flush(stdout)
end

pac,aac = loglogfit(ms, tac)
@printf("\nNLF (approxChol):  t ~ m^%.3f\n", pac)
if length(mlu)>=3; plu,alu=loglogfit(mlu,tlu); @printf("UMFPACK-LU:        t ~ m^%.3f   (crossover m* ~ %.2e)\n", plu, 10^((alu-aac)/(pac-plu))); end
if length(mch)>=3; pch,ach=loglogfit(mch,tch); @printf("CHOLMOD-METIS:     t ~ m^%.3f   (crossover m* ~ %.2e)\n", pch, 10^((ach-aac)/(pac-pch))); end
open(joinpath(@__DIR__,"scaling_cholmod.csv"),"w") do io
    println(io,"m,nlf_ac_s,umfpack_lu_s,cholmod_metis_s")
    for m in sort(unique([ms;mlu;mch]))
        f(mv,tv)=(i=findfirst(==(m),mv); i===nothing ? "" : @sprintf("%.3f",tv[i]))
        println(io, @sprintf("%d,%s,%s,%s", Int(m), f(ms,tac), f(mlu,tlu), f(mch,tch)))
    end
end
println("wrote scaling_cholmod.csv"); flush(stdout)
