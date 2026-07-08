# Real-network validation (paper §6): destination-bundled logit-SUE congestion-toll design on the two
# canonical TNTP road networks with their PUBLISHED origin-destination matrices — Sioux Falls and Anaheim.
# Confirms the multicommodity solver + adjoint toll design run end-to-end on real topology, real BPR curves
# (b, power from the .tntp), and real OD demand (one commodity per destination zone), not just synthetic
# random graphs. Reports network size, #commodities K, TSTT reduction, inner-iteration count, tolled fraction.
#   Usage:  julia --project=. scripts/design_study_tntp.jl
using DNLF, LinearAlgebra, SparseArrays, Printf

# Adjoint projected-gradient toll design (identical scheme to the synthetic study): each outer step takes one
# O(K·m) adjoint solve for ∇_τ TSTT, then a backtracking projected step accepted only if it lowers TSTT.
function toll_design(N, dk, γ, fk, fa; steps = 15, lr0 = 5e-4)
    τ = zeros(N.m); T = DNLF.mc_tstt(N, fa)
    for _ in 1:steps
        g = DNLF.mc_adjoint(N, fk, fa, γ); lr = lr0; acc = false
        for _ in 1:25
            τt = max.(τ .- lr .* g, 0.0); _, fkt, fat, _ = solve_sue(N, dk, γ; tolls = τt)
            if DNLF.mc_tstt(N, fat) < T; τ = τt; fk = fkt; fa = fat; T = DNLF.mc_tstt(N, fat); acc = true; break; end
            lr /= 2
        end
        acc || break
    end
    τ, T
end

function run_instance(name, netfile, tripsfile, γ; scale = 1.0)
    N   = load_tntp_net(joinpath(pkgdir(DNLF), "data", name, netfile))
    od  = scale .* load_tntp_trips(joinpath(pkgdir(DNLF), "data", name, tripsfile))
    dk  = destination_demands(od, N.n)
    _, fk, fa, its = solve_sue(N, dk, γ); T0 = DNLF.mc_tstt(N, fa)
    τ, T = toll_design(N, dk, γ, fk, fa)
    thr = 1e-3 * maximum(τ; init = 0.0)
    @printf("%-12s n=%-4d m=%-5d K=%-3d  γ=%-4.1f  TSTT-red=%5.2f%%  inner=%-3d  tolled=%4.1f%%  maxV/C=%.2f\n",
            name, N.n, N.m, length(dk), γ, 100*(1 - T/T0), its, 100*count(>(thr), τ)/N.m, maximum(fa ./ N.cap))
end

# compile (tiny warmup, silent) then report
let Nw = load_tntp_net(joinpath(pkgdir(DNLF), "data", "SiouxFalls", "SiouxFalls_net.tntp")),
    odw = load_tntp_trips(joinpath(pkgdir(DNLF), "data", "SiouxFalls", "SiouxFalls_trips.tntp"))
    solve_sue(Nw, destination_demands(odw, Nw.n), 5.0)
end
println("=== Real TNTP networks, published OD matrices, destination-bundled logit-SUE toll design ===")
run_instance("SiouxFalls", "SiouxFalls_net.tntp", "SiouxFalls_trips.tntp", 2.0)
run_instance("SiouxFalls", "SiouxFalls_net.tntp", "SiouxFalls_trips.tntp", 5.0)
run_instance("Anaheim",    "Anaheim_net.tntp",    "Anaheim_trips.tntp",    2.0)
run_instance("Anaheim",    "Anaheim_net.tntp",    "Anaheim_trips.tntp",    5.0)
