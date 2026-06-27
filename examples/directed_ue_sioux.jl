# Directed single-commodity user-equilibrium on SiouxFalls, on the NLF stack (DNLF → NLF → LAMG+).
# Solves it two ways and compares: (A) DNLF's rectified-law Newton; (B) an independent Frank–Wolfe
# oracle. Agreement + Wardrop conditions = the directed extension works.
using DNLF, LinearAlgebra, Printf

N = load_tntp_net(joinpath(@__DIR__, "..", "data", "SiouxFalls", "SiouxFalls_net.tntp"))
@printf("SiouxFalls: %d nodes, %d directed arcs\n", N.n, N.m)

r, s, D = 1, 20, 30000.0
println("\nRouting D=$(Int(D)) from node $r to node $s (single commodity).")
φ, fN, steps = solve_ue(N, r, s, D)
fF = frank_wolfe(N, r, s, D)

active = count(>(1e-6), fN)
keep = setdiff(1:N.n, s)
consv = norm((-(N.B*fN))[keep] - [i == r ? D : 0.0 for i in keep])
redc = [DNLF.tcost(N, a, fN[a]) - (φ[N.ini[a]] - φ[N.ter[a]]) for a in 1:N.m]
w_used   = maximum(abs(redc[a]) for a in 1:N.m if fN[a] > 1e-6; init = 0.0)
w_unused = minimum(redc[a] for a in 1:N.m if fN[a] <= 1e-6; init = 0.0)
relflow = norm(fN - fF)/max(norm(fF), 1e-30)
costN = sum(DNLF.tcost(N, a, fN[a])*fN[a] for a in 1:N.m)
costF = sum(DNLF.tcost(N, a, fF[a])*fF[a] for a in 1:N.m)

@printf("\n== DNLF rectified Newton ==\n  Newton steps: %d   active arcs: %d / %d   conservation: %.2e\n",
        steps, active, N.m, consv)
@printf("  Wardrop:  max|reduced cost| USED = %.2e   min reduced cost UNUSED = %.3f (>=0 ok)\n",
        w_used, w_unused)
@printf("\n== vs Frank–Wolfe oracle ==\n  ||f_DNLF − f_FW|| / ||f_FW|| = %.2e   total-cost rel.diff = %.2e\n",
        relflow, abs(costN - costF)/costF)
@printf("\nVERDICT: %s\n", (relflow < 1e-3 && w_used < 1e-4 && w_unused > -1e-4) ?
        "directed UE solved on the NLF stack; matches FW; Wardrop satisfied." : "mismatch — inspect.")
