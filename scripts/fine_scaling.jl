# Does the FIXED solver (fine schedule + accurate mode, reaching true 1e-9) still scale near-linearly?
# Size-scaled synthetic family, measure steps/setups/time/exact-residual and fit the wall-clock exponent.
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
exresid(net,d,x) = (f=[DNLF.rho(net,a,(x[net.ini[a]]-x[net.ter[a]]))[1] for a in 1:net.m];
                    min(norm(net.B*f .- d), norm(net.B*f .+ d))/norm(d))
sched(δmin,ratio) = tuple((0.5 .* ratio .^ (0:ceil(Int, log(δmin/0.5)/log(ratio))))...)
DF = sched(1e-8, 0.85)
net,d=rand_net(2000); DNLF.solve_flow(net,d,zeros(net.m); tol=1e-9, dfracs=DF)  # compile
@printf("fine-accurate fix, %d-level schedule\n", length(DF))
@printf("%-6s %-8s %-8s %-8s %-9s %-10s\n","n","m","steps","setups","time(s)","resid")
ms=Float64[]; ts=Float64[]
for n in (4000, 8000, 16000, 32000, 64000)
    net,d=rand_net(n)
    t=@elapsed ((x,f,s,su)=DNLF.solve_flow(net,d,zeros(net.m); tol=1e-9, dfracs=DF))
    push!(ms,net.m); push!(ts,t)
    @printf("%-6d %-8d %-8d %-8d %-9.1f %-10.1e\n",n,net.m,s,su,t,exresid(net,d,x)); flush(stdout)
end
x=log10.(ms); y=log10.(ts); xm=sum(x)/length(x); ym=sum(y)/length(y)
@printf("exponent: t ~ m^%.2f\n", sum((x.-xm).*(y.-ym))/sum((x.-xm).^2))
