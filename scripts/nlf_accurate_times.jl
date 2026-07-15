# Lean: NLF accurate-mode wall-clock + achieved residual on the sec 5.6 instances (FISTA numbers already in
# the paper). For the honest #3 restatement of the first-order comparison.
using DNLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))    # read_mtx_lcc, build_net, datadir
resid(net,d,f) = min(norm(net.B*f .- d), norm(net.B*f .+ d)) / norm(d)
let (n,e)=read_mtx_lcc(joinpath(datadir(),"SNAP__as-735.mtx")); (nt,dd)=build_net(n,e)
    DNLF.solve_flow(nt,dd,zeros(nt.m); tol=1e-9); end                       # compile
@printf("%-24s %-8s %-10s %-10s\n","graph","m","accur(s)","resid")
for g in ("SNAP__as-735","SNAP__p2p-Gnutella08","SNAP__Oregon-1","SNAP__as-caida")
    p = joinpath(datadir(), g*".mtx"); isfile(p) || (println("  (missing $g)"); continue)
    n,e = read_mtx_lcc(p); net,d = build_net(n,e)
    local f
    t = @elapsed ((_,f,_,_) = DNLF.solve_flow(net,d,zeros(net.m); tol=1e-9))   # accurate mode
    @printf("%-24s %-8d %-10.2f %-10.1e\n", g, net.m, t, resid(net,d,f)); flush(stdout)
end
