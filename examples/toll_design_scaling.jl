# Scaling of the design-gradient cost with the number of tollable links k.
# Given the base equilibrium (which both methods need once), the cost to obtain ∇_τ TSTT is:
#   DNLF adjoint        : ONE symmetric-Laplacian solve   (independent of k)
#   derivative-free loop: 2k user-equilibrium re-solves   (one per toll, finite difference)
# So the speedup grows linearly with k — modest on a toy, orders of magnitude at city scale.
using DNLF, LinearAlgebra, Printf

function design_cost(name, netpath, r, s, D)
    N = load_tntp_net(netpath); k = N.m
    φ0, f0, ueSteps = solve_ue(N, r, s, D)                 # base equilibrium (shared by both)
    τ0 = zeros(N.m)
    adjoint_grad(N, s, φ0, f0, τ0)                          # warm up
    t_adj = @elapsed adjoint_grad(N, s, φ0, f0, τ0)         # DNLF: one adjoint solve
    h = 1e-2
    t_fd = @elapsed for a in 1:k                            # derivative-free: 2k warm UE re-solves
        τp = copy(τ0); τp[a] += h; solve_ue(N, r, s, D; tolls = τp, init = φ0)
        τm = copy(τ0); τm[a] -= h; solve_ue(N, r, s, D; tolls = τm, init = φ0)
    end
    @printf("%-12s k=%-5d | adjoint(1 solve) %9.5f s | derivative-free(%5d solves) %8.3f s | SPEEDUP %7.0f×\n",
            name, k, t_adj, 2k, t_fd, t_fd/t_adj)
end

println("Marginal cost of one full design gradient (∇ over all k tollable links), given the base UE:\n")
design_cost("SiouxFalls", joinpath(@__DIR__, "..", "data", "SiouxFalls", "SiouxFalls_net.tntp"), 1, 20, 30000.0)
design_cost("Anaheim",    joinpath(@__DIR__, "..", "data", "Anaheim", "Anaheim_net.tntp"),       1, 400, 20000.0)
println("\nThe adjoint is ONE Laplacian solve regardless of k; the derivative-free loop is 2k UE solves.")
println("Speedup ∝ k → orders of magnitude when k is thousands of links (city networks) or millions")
println("of OD entries (matrix calibration) — exactly the regime real toll/NDP/OD-estimation studies run.")
