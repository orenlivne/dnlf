# =====================================================================================================
# Reproducible SYNTHETIC scaling study (paper Table 1 / Figure 1): controlled size-scaled family of
# irregular (poorly-separable) random graphs. Times the near-linear engine (approximate Cholesky) against
# the FAIR frozen-per-level direct baseline (LU, one factorization per continuation level, reused within),
# loose-intermediate smoothing homotopy, and FITS the wall-clock exponents in-run (no hand-typed numbers).
# The crossover m* is the intersection of the two fitted log-log lines.
#
# Usage:  julia --project=. scripts/scaling_synthetic.jl
# Emits:  the Table 1 rows, the fitted exponents t ~ m^p, and the crossover m*, plus scaling_points.csv.
# =====================================================================================================
using DNLF, LinearAlgebra, SparseArrays, Printf, Random

function rand_net(n; deg=5, seed=1)                       # same generator as reproduce.jl / baseline
    rng=MersenneTwister(seed); E=Set{Tuple{Int,Int}}()
    for u in 1:n,_ in 1:deg; v=rand(rng,1:n); v==u&&continue; push!(E,(min(u,v),max(u,v))); end
    es=collect(E); ini=Int[];ter=Int[]; for (u,v) in es; push!(ini,u);push!(ter,v);push!(ini,v);push!(ter,u); end
    m=length(ini); B=sparse([ini;ter],[1:m;1:m],[fill(-1.0,m);fill(1.0,m)],n,m)
    nt=DNLF.DirectedNetwork(n,m,ini,ter,1000 .*(0.5 .+rand(rng,m)),1 .+rand(rng,m),fill(0.15,m),fill(4.0,m),B)
    dd=zeros(n); for u in randperm(rng,n)[1:n÷8];dd[u]+=1;end; for v in randperm(rng,n)[1:n÷8];dd[v]-=1;end
    dd.-=sum(dd)/n; dd.*=(3000.0*n/(sum(abs,dd)/2)); nt,dd
end

solve(net,d;inner) = DNLF.solve_flow(net,d,zeros(net.m); inner=inner, itol=3e-2, inmax=6)

# least-squares slope/intercept of log10 t vs log10 m, CENTERED (numerically stable; the uncentered
# normal-equation form n*Sxx-(Sx)^2 subtracts two large near-equal numbers and can lose digits here).
function loglogfit(ms, ts)
    x = log10.(ms); y = log10.(ts); xm = sum(x)/length(x); ym = sum(y)/length(y)
    b = sum((x .- xm).*(y .- ym)) / sum((x .- xm).^2)
    b, ym - b*xm                                          # slope (exponent), intercept
end

NS_BOTH = (1000, 2000, 4000, 8000, 16000)                 # direct feasible (m up to ~1.4e5)
NS_AC   = (32000, 64000)                                  # approxChol only (m up to ~5.7e5)
LUCAP   = 160_000

ms=Float64[]; tac=Float64[]; stp=Int[]; bld=Int[]; mlu=Float64[]; tlu=Float64[]
@printf("%-8s %-9s | %-10s %-8s %-8s | %-10s\n","n","m","approxChol","steps","setups","direct")
for n in (NS_BOTH..., NS_AC...)
    net,d = rand_net(n)
    solve(net,d; inner=:approxchol)                       # compile
    Hp=(Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
    ta = @elapsed ((_,_,s)=DNLF.solve_flow(net,d,zeros(net.m); inner=:approxchol, itol=3e-2, inmax=6, Hpack=Hp))
    push!(ms,net.m); push!(tac,ta); push!(stp,s); push!(bld,Hp[4][])
    tl = NaN
    if net.m <= LUCAP
        solve(net,d; inner=:lu)                           # compile
        tl = @elapsed solve(net,d; inner=:lu)
        push!(mlu,net.m); push!(tlu,tl)
    end
    @printf("%-8d %-9d | %-10.2f %-8d %-8d | %-10s\n", net.n, net.m, ta, s, Hp[4][],
            isnan(tl) ? "-" : @sprintf("%.2f",tl))
end

pac,aac = loglogfit(ms, tac); plu,alu = loglogfit(mlu, tlu)
mstar = 10^((alu-aac)/(pac-plu))                          # crossover: aac+pac*x = alu+plu*x
@printf("\napproxChol:  t ~ m^%.2f      direct:  t ~ m^%.2f      crossover m* ~ %.2e\n", pac, plu, mstar)
open(joinpath(@__DIR__,"scaling_points.csv"),"w") do io
    println(io,"m,approxchol_s,direct_s,steps,setups")
    for i in eachindex(ms)
        di = findfirst(==(ms[i]), mlu)
        println(io, @sprintf("%d,%.3f,%s,%d,%d", Int(ms[i]), tac[i],
                di===nothing ? "" : @sprintf("%.3f",tlu[di]), stp[i], bld[i]))
    end
end
println("wrote scaling_points.csv")
