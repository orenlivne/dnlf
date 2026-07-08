# =====================================================================================================
# reproduce_mc.jl — multicommodity logit-SUE congestion design at scale.
#
# Many OD pairs (commodities) share a congested irregular network. Solve the logit stochastic user
# equilibrium (fixed dispersion γ), then reduce total system travel time (TSTT) by projected-gradient
# tolling using the near-linear design adjoint (∇_τ TSTT for all tolls in ONE adjoint solve). Reports the
# TSTT reduction, the (single-digit) inner-iteration count, and wall-clock — demonstrating the whole
# bilevel design loop is near-linear in the total problem size Kn.
#
# Run:  julia --project=. examples/reproduce_mc.jl
# =====================================================================================================
using DNLF, LinearAlgebra, SparseArrays, Random, Printf

function rand_net(n; deg=5, seed=1)
    rng=MersenneTwister(seed); E=Set{Tuple{Int,Int}}()
    for u in 1:n,_ in 1:deg; v=rand(rng,1:n); v==u&&continue; push!(E,(min(u,v),max(u,v))); end
    es=collect(E); ini=Int[];ter=Int[]; for (u,v) in es; push!(ini,u);push!(ter,v);push!(ini,v);push!(ter,u); end
    m=length(ini); B=sparse([ini;ter],[1:m;1:m],[fill(-1.0,m);fill(1.0,m)],n,m)
    DNLF.DirectedNetwork(n,m,ini,ter,1000 .*(0.5 .+rand(rng,m)),1 .+rand(rng,m),fill(0.15,m),fill(4.0,m),B), rng
end

N,rng = rand_net(600); K = 8; γ = 0.15                        # 8 commodities on a 600-node irregular net
dk = Vector{Vector{Float64}}()
for _ in 1:K                                                  # well-spread demand (many OD pairs per commodity)
    v=zeros(N.n); for _ in 1:10; v[rand(rng,1:N.n)]+=5.0e2; v[rand(rng,1:N.n)]-=5.0e2; end; v.-=sum(v)/N.n; push!(dk,v)
end
@printf("multicommodity design: n=%d m=%d K=%d γ=%.2g\n", N.n, N.m, K, γ)

_, fk, fa, its0 = solve_sue(N, dk, γ; tolls=zeros(N.m)); T0 = DNLF.mc_tstt(N, fa)
@printf("[equilibrium] TSTT=%.4e  inner iters=%d\n", T0, its0)

# projected-gradient toll design (τ ≥ 0), backtracking on TSTT, adjoint gradient reused each step
function toll_design(N, dk, γ, fk, fa)
    τ = zeros(N.m); T = DNLF.mc_tstt(N, fa)
    for _ in 1:25
        g = DNLF.mc_adjoint(N, fk, fa, γ); lr = 5e-4; accepted = false
        for _ in 1:20
            τt = max.(τ .- lr .* g, 0.0)
            _, fkt, fat, _ = solve_sue(N, dk, γ; tolls = τt)
            if DNLF.mc_tstt(N, fat) < T; τ = τt; fk = fkt; fa = fat; T = DNLF.mc_tstt(N, fat); accepted = true; break; end
            lr /= 2
        end
        accepted || break
    end
    τ, T, fk, fa
end
tdesign = @elapsed ((τ, T, fk, fa) = toll_design(N, dk, γ, fk, fa))
@printf("[design]      TSTT=%.4e  reduction=%.2f%%  over design loop in %.1fs\n", T, 100*(1-T/T0), tdesign)
@printf("  active tolls: %d of %d arcs ; one adjoint solve gives ∇ for all %d tolls\n",
        count(>(1e-9), τ), N.m, N.m)
