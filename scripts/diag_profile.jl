# Where does the ~40s/level go on soc-Epinions1 (811k arcs)? Thinning the homotopy didn't help (all ratios
# time out at ~same low level count), so the cost is PER-LEVEL, not level count. Profile the primitive ops:
# smoothed-law evaluation (rho_smooth does an 80-iter per-arc softplus inversion -> prime suspect), AMG
# hierarchy build, and AMG solve. This tells us which primitive dominates and hence what to actually fix.
using DNLF, NLF, LinearAlgebra, SparseArrays, Printf
import LAMG: LAMGOptions, setup, solve
include(joinpath(@__DIR__, "scaling.jl"))

g = "SNAP__soc-Epinions1"
path = joinpath(datadir(), g*".mtx")
n, edges = read_mtx_lcc(path); net, d = build_net(n, edges)
@printf("graph=%s n=%d m=%d\n", g, net.n, net.m); flush(stdout)

Bn = -net.B
tmean = sum(net.t0)/net.m
δ = 0.5 * tmean                      # level-1 delta (largest; representative)
law = DNLF.smoothed_law(net, zeros(net.m), δ)

# warm up compile
x = randn(net.n); f = zeros(net.m); dρ = zeros(net.m)
g_ = Bn' * x; law(f, dρ, g_)

# 1) smoothed-law evaluation cost (the per-arc 80-iter softplus inversion)
nlaw = 5
tlaw = @elapsed for _ in 1:nlaw; g_ = Bn'*x; law(f, dρ, g_); end
@printf("smoothed_law eval:      %.3fs each  (x%d)\n", tlaw/nlaw, nlaw); flush(stdout)

# 2) AMG hierarchy build cost
function laplacian_clean(J)
    off = J - spdiagm(0 => diag(J)); off = (off + off')/2; dropzeros!(off)
    off + spdiagm(0 => -vec(sum(off, dims=2)))
end
dρf = max.(dρ, 1e-12*maximum(dρ)); SC = maximum(dρf)
Lc = laplacian_clean((Bn * Diagonal(dρf) * Bn') ./ SC)
tbuild = @elapsed H = setup(Lc; options=LAMGOptions())
@printf("AMG hierarchy build:    %.3fs\n", tbuild); flush(stdout)

# 3) AMG solve cost
rhs = randn(net.n); rhs .-= sum(rhs)/net.n
nsolve = 5
tsolve = @elapsed for _ in 1:nsolve; solve(H, rhs; options=LAMGOptions(tol=0.05)); end
@printf("AMG solve (tol=0.05):   %.3fs each  (x%d)\n", tsolve/nsolve, nsolve); flush(stdout)

# 4) Jacobian assembly cost (B * Diagonal * B')
tjac = @elapsed for _ in 1:3; laplacian_clean((Bn * Diagonal(dρf) * Bn') ./ SC); end
@printf("Jacobian assembly:      %.3fs each\n", tjac/3); flush(stdout)

@printf("\nper-level ~= 1 build + 1 jac + ~6 steps x (1 solve + ~L law-evals in line search)\n")
@printf("If law-eval dominates (>>solve), the fix is a cheaper/analytic smoothed law, not fewer levels.\n")
println("DONE")
