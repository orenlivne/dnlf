# Ground-truth check for the #3 reframe: what relative residual does the DEFAULT (loose, frozen-chord) solve
# actually reach on the real SNAP instances the paper cites (sec 5.6 / tab:corpus)? Compare loose vs accurate.
using DNLF, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "scaling.jl"))   # read_mtx_lcc, build_net, datadir
rr(net,d,f) = min(norm(net.B*f .- d), norm(net.B*f .+ d)) / norm(d)
graphs = ("SNAP__as-735", "SNAP__p2p-Gnutella08", "SNAP__Oregon-1", "SNAP__as-caida")
# compile
let (n,e)=read_mtx_lcc(joinpath(datadir(),"SNAP__as-735.mtx")); (nt,dd)=build_net(n,e);
    DNLF.solve_flow(nt,dd,zeros(nt.m); itol=3e-2,inmax=6); DNLF.solve_flow(nt,dd,zeros(nt.m)); end
let (n,e)=read_mtx_lcc(joinpath(datadir(),"SNAP__as-735.mtx")); (nt,dd)=build_net(n,e);
    DNLF.solve_flow(nt,dd,zeros(nt.m); itol=3e-2,inmax=6,polish=:ssn); end   # compile ssn
@printf("%-24s %-8s %-12s %-12s %-12s\n","graph","m","loose resid","accurate resid","ssn resid")
for g in graphs
    p = joinpath(datadir(), g*".mtx"); isfile(p) || (println("  (missing $g)"); continue)
    n,e = read_mtx_lcc(p); net,d = build_net(n,e)
    _,fL,_,_ = DNLF.solve_flow(net,d,zeros(net.m); itol=3e-2, inmax=6, tol=1e-9)               # loose (paper config)
    _,fA,_,_ = DNLF.solve_flow(net,d,zeros(net.m); tol=1e-9)                                    # accurate
    _,fS,_,_ = DNLF.solve_flow(net,d,zeros(net.m); itol=3e-2, inmax=6, tol=1e-9, polish=:ssn)   # semismooth polish
    @printf("%-24s %-8d %-12.1e %-12.1e %-12.1e\n", g, net.m, rr(net,d,fL), rr(net,d,fA), rr(net,d,fS))
end
