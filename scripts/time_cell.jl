# Time ONE (engine, size) cell of the crossover study in an ISOLATED process, so an out-of-memory crash in a
# direct factorization (which on macOS kills the whole process) documents itself instead of cascading into the
# other cells. Same synthetic random-degree-5 generator/seed and same loose-homotopy solve as scaling_cholmod.jl.
#   Usage:  julia --project=. scripts/time_cell.jl <n> <approxchol|lu|cholmod>
#   Prints: "RESULT <m> <engine> <seconds>"  on success (a driver greps this; absence ⇒ OOM/crash).
using DNLF, LinearAlgebra, SparseArrays, Printf, Random
function rand_net(n; deg=5, seed=1)
    rng=MersenneTwister(seed); E=Set{Tuple{Int,Int}}()
    for u in 1:n,_ in 1:deg; v=rand(rng,1:n); v==u&&continue; push!(E,(min(u,v),max(u,v))); end
    es=collect(E); ini=Int[];ter=Int[]; for (u,v) in es; push!(ini,u);push!(ter,v);push!(ini,v);push!(ter,u); end
    m=length(ini); B=sparse([ini;ter],[1:m;1:m],[fill(-1.0,m);fill(1.0,m)],n,m)
    nt=DNLF.DirectedNetwork(n,m,ini,ter,1000 .*(0.5 .+rand(rng,m)),1 .+rand(rng,m),fill(0.15,m),fill(4.0,m),B)
    dd=zeros(n); for u in randperm(rng,n)[1:n÷8];dd[u]+=1;end; for v in randperm(rng,n)[1:n÷8];dd[v]-=1;end
    dd.-=sum(dd)/n; dd.*=(3000.0*n/(sum(abs,dd)/2)); nt,dd
end
solve(net,d;inner) = DNLF.solve_flow(net,d,zeros(net.m); inner=inner, itol=3e-2, inmax=6)
n   = parse(Int, ARGS[1]); eng = Symbol(ARGS[2])
net,d = rand_net(n)
@printf("cell n=%d m=%d engine=%s\n", net.n, net.m, eng); flush(stdout)
solve(net,d; inner=eng)                        # compile / warm
t0=time(); solve(net,d; inner=eng); t=time()-t0
@printf("RESULT %d %s %.3f\n", net.m, eng, t); flush(stdout)
