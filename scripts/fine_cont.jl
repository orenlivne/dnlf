# Test a PROPER continuation: fine geometric schedule + every level solved tight (accurate mode, adaptive
# refresh), so no single solve is far from converged. Does the exact-law residual reach 1e-9? Also probe
# whether a finer schedule tames the stall. On the real SNAP graphs.
using DNLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))
exresid(net,d,x,tolls) = (f=[DNLF.rho(net,a,(x[net.ini[a]]-x[net.ter[a]])-tolls[a])[1] for a in 1:net.m];
                          min(norm(net.B*f .- d), norm(net.B*f .+ d))/norm(d))
sched(δmin,ratio) = tuple((0.5 .* ratio .^ (0:ceil(Int, log(δmin/0.5)/log(ratio))))...)
graphs = ("SNAP__as-735","SNAP__p2p-Gnutella08","SNAP__as-caida")
let (n,e)=read_mtx_lcc(joinpath(datadir(),"SNAP__as-735.mtx")); (nt,dd)=build_net(n,e)
    DNLF.solve_flow(nt,dd,zeros(nt.m); dfracs=sched(1e-6,0.7)); end   # compile accurate
for (δmin,ratio) in ((1e-8,0.7), (1e-8,0.85))
    df = sched(δmin,ratio)
    @printf("\n=== ACCURATE, fine schedule: delta_min=%.0e ratio=%.2f (%d levels) ===\n", δmin, ratio, length(df))
    @printf("%-24s %-8s %-8s %-8s %-9s %-10s\n","graph","m","steps","setups","time(s)","exact resid")
    for g in graphs
        p = joinpath(datadir(), g*".mtx"); isfile(p) || continue
        n,e = read_mtx_lcc(p); net,d = build_net(n,e)
        Hp=(Ref{Any}(nothing),Ref(1.0),Ref(false),Ref(0),Ref(1.0))
        t=@elapsed ((x,f,s,su)=DNLF.solve_flow(net,d,zeros(net.m); tol=1e-9, dfracs=df, Hpack=Hp))  # accurate (no itol)
        @printf("%-24s %-8d %-8d %-8d %-9.1f %-10.1e\n", g, net.m, s, su, t, exresid(net,d,x,zeros(net.m)))
        flush(stdout)
    end
end
