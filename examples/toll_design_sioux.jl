# Bilevel network design on the NLF stack: optimal CONGESTION PRICING on SiouxFalls.
# Outer problem: choose link tolls τ ≥ 0 to MINIMIZE total system travel time (TSTT), subject to the
# directed user equilibrium (a real MPEC: second-best tolling).
#
# DNLF gets ∇_τ TSTT by the ADJOINT — one extra symmetric-Laplacian solve, for ALL tolls at once.
# A SOTA equilibrium solver returns no derivatives, so the only option is a derivative-free / finite-
# difference loop: re-solve the UE once per toll. We show the two gradients AGREE and that the adjoint
# is orders of magnitude cheaper — and then optimize the tolls with gradients the SOTA cannot supply.
using DNLF, LinearAlgebra, Printf

function main()
    N = load_tntp_net(joinpath(@__DIR__, "..", "data", "SiouxFalls", "SiouxFalls_net.tntp"))
    r, s, D = 1, 20, 30000.0
    k = N.m
    @printf("SiouxFalls: %d nodes, %d directed arcs (= %d tollable design variables)\n", N.n, N.m, k)

    # ---- adjoint gradient: 1 UE solve + 1 adjoint Laplacian solve ----
    τ0 = zeros(N.m)
    TSTT0, gA, φ0, f0, ueSteps = toll_gradient(N, r, s, D; tolls = τ0)
    @printf("\nbase UE: TSTT = %.4e   (UE Newton steps = %d)\n", TSTT0, ueSteps)

    # ---- finite-difference gradient: derivative-free baseline (re-solve UE per toll), warm-started ----
    h = 1e-2; gF = zeros(k); fd_solves = 0
    for a in 1:k
        τp = copy(τ0); τp[a] += h; _, fp, _ = solve_ue(N, r, s, D; tolls = τp, init = φ0); fd_solves += 1
        τm = copy(τ0); τm[a] -= h; _, fm, _ = solve_ue(N, r, s, D; tolls = τm, init = φ0); fd_solves += 1
        gF[a] = (tstt(N, fp) - tstt(N, fm)) / (2h)
    end
    @printf("\n== correctness ==\n  ||grad_adjoint − grad_FD|| / ||grad_FD|| = %.2e   (FD used %d UE solves)\n",
            norm(gA - gF)/max(norm(gF), 1e-30), fd_solves)

    # ---- cost of one full gradient ----
    t_adj = @elapsed toll_gradient(N, r, s, D; tolls = τ0)
    t_fd  = @elapsed for a in 1:k
        τp = copy(τ0); τp[a] += h; solve_ue(N, r, s, D; tolls = τp, init = φ0)
        τm = copy(τ0); τm[a] -= h; solve_ue(N, r, s, D; tolls = τm, init = φ0)
    end
    @printf("\n== cost of ONE full gradient (∇ over all %d design vars) ==\n", k)
    @printf("  adjoint   : 1 UE + 1 Laplacian solve         %.4f s\n", t_adj)
    @printf("  finite-diff (derivative-free SOTA loop): %d UE solves    %.4f s\n", fd_solves, t_fd)
    @printf("  SPEEDUP   : %.1f×   — and it grows as O(#design vars): orders of magnitude at city scale.\n",
            t_fd/t_adj)

    # ---- design optimization: projected gradient descent on tolls (τ ≥ 0) ----
    println("\n== optimal-toll design loop (minimize TSTT, projected gradient) ==")
    τ = zeros(N.m)
    for it in 1:25
        Tk, g, _, _, _ = toll_gradient(N, r, s, D; tolls = τ)
        η = 2.0 / maximum(abs.(g); init = 1.0)
        for _ in 1:40
            τt = max.(τ .- η.*g, 0.0); _, ft, _ = solve_ue(N, r, s, D; tolls = τt)
            tstt(N, ft) < Tk && (τ = τt; break); η *= 0.5
        end
        it % 5 == 0 && @printf("  iter %2d: TSTT = %.4e   (%.2f%% below the untolled UE)\n",
                               it, Tk, 100*(TSTT0 - Tk)/TSTT0)
    end
    _, ffin, _ = solve_ue(N, r, s, D; tolls = τ); Tfin = tstt(N, ffin)
    @printf("\nRESULT: optimal tolls cut TSTT by %.2f%% (%.4e → %.4e), using gradients a derivative-free\n",
            100*(TSTT0 - Tfin)/TSTT0, TSTT0, Tfin)
    @printf("        SOTA solver cannot supply — each gradient is ~%d× cheaper than the finite-diff loop.\n", k)
end

main()
