# Diagnosis test: the stall is the jump from the continuation floor (delta=1e-4) to the exact (delta=0)
# nonsmooth law. Fix = continue delta ALL THE WAY DOWN to delta_min ~ target tol, so the SMOOTHED solution
# already equals the exact equilibrium to O(delta_min) -- every solve stays smooth, no nonsmooth polish.
# Measure the EXACT-law residual reached, plus steps/setups/time, on the real SNAP graphs.
using DNLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))    # read_mtx_lcc, build_net, datadir
exresid(net,d,x,tolls) = (f=[DNLF.rho(net,a,(x[net.ini[a]]-x[net.ter[a]])-tolls[a])[1] for a in 1:net.m];
                          min(norm(net.B*f .- d), norm(net.B*f .+ d))/norm(d))   # HARD-law residual
sched(δmin) = tuple((10 .^ range(log10(0.5), log10(δmin); length=max(13, ceil(Int, log2(0.5/δmin))+1)))...)
graphs = ("SNAP__as-735","SNAP__p2p-Gnutella08","SNAP__as-caida")
let (n,e)=read_mtx_lcc(joinpath(datadir(),"SNAP__as-735.mtx")); (nt,dd)=build_net(n,e)
    DNLF.solve_flow(nt,dd,zeros(nt.m); itol=3e-2,inmax=6,dfracs=sched(1e-6)); end   # compile
for δmin in (1e-4, 1e-6, 1e-8, 1e-10)
    df = sched(δmin)
    @printf("\n=== delta_min=%.0e  (%d levels) ===\n", δmin, length(df))
    @printf("%-24s %-8s %-8s %-8s %-10s\n","graph","m","steps","setups","exact resid")
    for g in graphs
        p = joinpath(datadir(), g*".mtx"); isfile(p) || continue
        n,e = read_mtx_lcc(p); net,d = build_net(n,e); tolls=zeros(net.m)
        Hp=(Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
        x,f,s,su = DNLF.solve_flow(net,d,zeros(net.m); itol=3e-2, inmax=6, tol=1e-9, dfracs=df, Hpack=Hp)
        @printf("%-24s %-8d %-8d %-8d %-10.1e\n", g, net.m, s, su, exresid(net,d,x,tolls))
        flush(stdout)
    end
end
