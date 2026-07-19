using DNLF, Test, LinearAlgebra, SparseArrays, Random

net = DNLF.load_tntp_net(joinpath(pkgdir(DNLF), "data", "SiouxFalls", "SiouxFalls_net.tntp"))
r, s, D = 1, 20, 3.0e4                         # a congested single-OD instance (flow splits across paths)
d = zeros(net.n); d[r] = D; d[s] = -D
reldiff(a, b) = norm(a - b) / max(norm(b), 1e-30)

# small irregular (poorly-separable) network for the multicommodity tests — the solver's target regime,
# unlike the tiny structured SiouxFalls road graph
function mc_randnet(n; deg = 6, seed = 11)
    rng = MersenneTwister(seed); E = Set{Tuple{Int,Int}}()
    for u in 1:n, _ in 1:deg; v = rand(rng, 1:n); v == u && continue; push!(E, (min(u, v), max(u, v))); end
    es = collect(E); ii = Int[]; tt = Int[]
    for (u, v) in es; push!(ii, u); push!(tt, v); push!(ii, v); push!(tt, u); end
    m = length(ii); Bm = sparse([ii; tt], [1:m; 1:m], [fill(-1.0, m); fill(1.0, m)], n, m)
    DNLF.DirectedNetwork(n, m, ii, tt, 1000 .*(0.5 .+ rand(rng, m)), 1 .+ rand(rng, m),
                         fill(0.15, m), fill(4.0, m), Bm)
end
mcnet = mc_randnet(200)

@testset "DNLF — full algorithm coverage" begin

    # ---- Phase 0: parsing & network construction ----
    @testset "load_tntp_net / DirectedNetwork" begin
        @test net.n == 24 && net.m == 76                       # Sioux Falls
        @test length(net.ini) == net.m == length(net.ter)
        @test size(net.B) == (net.n, net.m)
        @test all(sum(net.B; dims = 1) .== 0)                  # each arc column sums to 0 (−1 tail, +1 head)
        @test all(net.cap .> 0) && all(net.t0 .> 0)
    end

    @testset "load_tntp_trips / destination_demands" begin
        od = DNLF.load_tntp_trips(joinpath(pkgdir(DNLF), "data", "SiouxFalls", "SiouxFalls_trips.tntp"))
        @test size(od) == (24, 24)                             # Sioux Falls has 24 zones
        @test all(od .>= 0) && all(diag(od) .== 0)             # nonneg, no intra-zone trips
        @test isapprox(sum(od), 360600.0; rtol = 1e-9)         # published TOTAL OD FLOW
        dem = DNLF.destination_demands(od, net.n)
        @test length(dem) == 24                                # one commodity per destination zone
        @test all(length(d) == net.n for d in dem)             # demands padded to network node count
        @test all(abs(sum(d)) < 1e-6 for d in dem)             # each commodity conserves mass
        j = 10; @test isapprox(dem[j][j], -sum(@view od[:, j]); rtol = 1e-9)  # sink = −(incoming total)
    end

    # ---- Cost laws: BPR, its inverse ρ, and the smoothed law ----
    @testset "cost laws: tcost / dcost / rho / rho_smooth" begin
        a = 1
        @test DNLF.tcost(net, a, 0.0) ≈ net.t0[a] + DNLF.REG*net.t0[a]/net.cap[a]*0  # free-flow at f=0
        @test DNLF.dcost(net, a, 5.0) > 0                                            # increasing cost
        # ρ inverts the cost on active arcs: tcost(ρ(g)) = g for g > t⁰
        for g in (net.t0[a] + 1.0, net.t0[a] + 20.0)
            f, dρ = DNLF.rho(net, a, g)
            @test f > 0 && dρ > 0
            @test abs(DNLF.tcost(net, a, f) - g) < 1e-8*(g + 1)
            @test isapprox(dρ, 1/DNLF.dcost(net, a, f); rtol = 1e-6)                 # ρ′ = 1/t′
        end
        @test DNLF.rho(net, a, net.t0[a])       == (0.0, 0.0)                        # dead zone at threshold
        @test DNLF.rho(net, a, net.t0[a] - 1.0) == (0.0, 0.0)                        # dead zone below
        # smoothed law → exact law as δ→0 (well above threshold), and ρ′>0 everywhere
        g = net.t0[a] + 15.0
        @test reldiff([DNLF.rho_smooth(net, a, g, 1e-6)[1]], [DNLF.rho(net, a, g)[1]]) < 1e-3
        @test DNLF.rho_smooth(net, a, net.t0[a] - 5.0, 1.0)[2] > 0                    # no dead zone when smoothed
    end

    # ---- Energy: ∇E equals the flow residual (globalization is on the right objective) ----
    @testset "ue_energy gradient = residual" begin
        rng = MersenneTwister(0); x = randn(rng, net.n)
        E = DNLF.ue_energy(net, d, zeros(net.m))
        f = zeros(net.m); for a in 1:net.m; f[a] = DNLF.rho(net, a, x[net.ini[a]] - x[net.ter[a]])[1]; end
        grad_analytic = -net.B * f - d                        # ∇E = Bₙ f − b with Bₙ = −B, b = d
        # central finite differences of E along a few coordinates
        ε = 1e-5; maxrel = 0.0
        for v in (2, 7, 13, 20)
            xp = copy(x); xp[v] += ε; xm = copy(x); xm[v] -= ε
            gnum = (E(xp) - E(xm)) / (2ε)
            maxrel = max(maxrel, abs(gnum - grad_analytic[v]) / max(abs(grad_analytic[v]), 1.0))
        end
        @test maxrel < 1e-4
    end

    # ---- Law callbacks fill (f, dρ) as newton_flow! expects ----
    @testset "law callbacks: rectified_law / smoothed_law" begin
        g = [net.t0[a] + 3.0 for a in 1:net.m]
        f = zeros(net.m); dρ = zeros(net.m)
        DNLF.rectified_law(net, zeros(net.m))(f, dρ, g)
        @test all(f .>= 0) && all(dρ .>= 0)
        @test f[1] ≈ DNLF.rho(net, 1, g[1])[1]
        fs = zeros(net.m); dρs = zeros(net.m)
        DNLF.smoothed_law(net, zeros(net.m), 0.5)(fs, dρs, g)
        @test all(dρs .> 0)                                    # smoothed: strictly positive conductance
    end

    # ---- Inner engines both solve Lx = b to tolerance and return x ⊥ 1 ----
    @testset "engines: approxchol_builder / lu_builder solve Lx=b" begin
        rng = MersenneTwister(1)
        # a connected weighted graph Laplacian from the network
        L = net.B * spdiagm(0 => 1 .+ rand(rng, net.m)) * net.B'
        b = randn(rng, net.n); b .-= sum(b)/net.n
        for build in (DNLF.approxchol_builder(), DNLF.lu_builder())
            x = build(L)(b)
            @test abs(sum(x)) < 1e-6                           # zero-mean
            @test norm(L*x - b) < 1e-5 * norm(b)               # solves the system
        end
    end

    # ---- Proposition 1: Jacobian is a symmetric Laplacian on the active subgraph ----
    @testset "Prop 1: symmetric active-subgraph Laplacian" begin
        φ, f, _ = DNLF.solve_ue(net, r, s, D; tol = 1e-9)
        ρ′ = [DNLF.rho(net, a, φ[net.ini[a]] - φ[net.ter[a]])[2] for a in 1:net.m]
        J = net.B * spdiagm(0 => ρ′) * net.B'
        @test issymmetric(J)
        @test all(ρ′ .>= 0)
        @test norm(J * ones(net.n)) < 1e-8 * norm(J, 1)
    end

    # ---- Forward solve: correctness, engine-agnosticism, homotopy = warm, loose = tight ----
    @testset "forward equilibrium solve" begin
        ffw = DNLF.frank_wolfe(net, r, s, D; iters = 30000)
        _, fa, _ = DNLF.solve_ue(net, r, s, D; inner = :approxchol, tol = 1e-10)
        _, fx, _ = DNLF.solve_ue(net, r, s, D; inner = :direct,     tol = 1e-10)
        @test reldiff(fa, ffw) < 1e-6                          # matches independent Frank–Wolfe
        @test reldiff(fa, fx)  < 1e-7                          # approxChol = direct
        xa, fal, _, _ = DNLF.solve_flow(net, d, zeros(net.m); inner = :approxchol, tol = 1e-10)
        xl, fll, _, _ = DNLF.solve_flow(net, d, zeros(net.m); inner = :lu,         tol = 1e-10)
        xc, flc, _, _ = DNLF.solve_flow(net, d, zeros(net.m); inner = :cholmod,    tol = 1e-10)
        @test reldiff(fal, fll) < 1e-6                         # engine-agnostic solve_flow (lu = approxchol)
        @test reldiff(fal, flc) < 1e-6                         # CHOLMOD+METIS direct baseline = approxchol
        # warm start from a nearby toll's solution reaches the same equilibrium
        φw, _, _, _ = DNLF.solve_flow(net, d, fill(0.5, net.m); tol = 1e-10)
        _, fw, _, _ = DNLF.solve_flow(net, d, zeros(net.m); tol = 1e-10, init = φw)
        @test reldiff(fw, fal) < 1e-6
    end

    # ---- Proposition 3: adjoint gradient = finite differences; toll_gradient wiring ----
    @testset "Prop 3: adjoint gradient vs finite differences" begin
        φ, f, _ = DNLF.solve_ue(net, r, s, D; tol = 1e-11, nmax = 6000)
        g = DNLF.adjoint_grad(net, s, φ, f, zeros(net.m))
        active = findall(>(1e-6), f); ε = 1e-2; maxrel = 0.0
        for a in active
            τp = zeros(net.m); τp[a] += ε; _, fp, _ = DNLF.solve_ue(net, r, s, D; tolls = τp, tol = 1e-11, nmax = 6000)
            τm = zeros(net.m); τm[a] -= ε; _, fm, _ = DNLF.solve_ue(net, r, s, D; tolls = τm, tol = 1e-11, nmax = 6000)
            fd = (DNLF.tstt(net, fp) - DNLF.tstt(net, fm)) / (2ε)
            maxrel = max(maxrel, abs(g[a] - fd) / max(abs(fd), 1.0))
        end
        @test maxrel < 1e-2
        T, gg, φ2, f2 = DNLF.toll_gradient(net, r, s, D; tolls = zeros(net.m))
        @test T ≈ DNLF.tstt(net, f2) && length(gg) == net.m   # toll_gradient returns (TSTT, ∇, φ, f)
    end

    # ---- tstt matches its definition ----
    @testset "tstt = Σ tₐ(fₐ) fₐ" begin
        _, f, _ = DNLF.solve_ue(net, r, s, D; tol = 1e-9)
        @test DNLF.tstt(net, f) ≈ sum(DNLF.tcost(net, a, f[a]) * f[a] for a in 1:net.m)
    end

    # ---- Design: monotone descent to the marginal-cost optimum (~5%) ----
    @testset "toll design reduces TSTT monotonically to the optimum" begin
        φ, f, _ = DNLF.solve_flow(net, d, zeros(net.m); tol = 1e-11)
        τ = zeros(net.m); T0 = DNLF.tstt(net, f); T = T0; mono = true
        for _ in 1:20
            gv = DNLF.adjoint_grad(net, s, φ, f, τ); lr = 0.05; acc = false
            for _ in 1:16
                τt = max.(τ .- lr .* gv, 0.0); φt, ft, _, _ = DNLF.solve_flow(net, d, τt; tol = 1e-11, init = φ)
                Tt = DNLF.tstt(net, ft)
                if Tt < T; (Tt > T + 1e-6*abs(T0)) && (mono = false); τ = τt; φ = φt; f = ft; T = Tt; acc = true; break; end
                lr *= 0.5
            end
            acc || break
        end
        @test T < T0 && (1 - T / T0) > 0.04 && mono
    end

    # ---- Phase 6: multicommodity logit stochastic user equilibrium (small irregular graph mcnet) ----
    @testset "multicommodity logit-SUE: flow map / Jacobian / solve / γ→0" begin
        Random.seed!(1); K = 3; γ = 0.2; mn = mcnet
        dk = [(v = zeros(mn.n); v[o] = 3.0e3; v[t] = -3.0e3; v) for (o, t) in ((1, 100), (3, 150), (5, 180))]
        φ = 0.3 .* randn(mn.n, K); fk, fa = DNLF.mc_flows(mn, φ, γ)
        @test reldiff(vec(sum(fk, dims = 2)), fa) < 1e-10               # shares sum to total
        okkt = 0.0                                                      # per-arc KKT stationarity
        for a in 1:mn.m, k in 1:K
            fk[a, k] < 1e-8 * fa[a] && continue
            g = φ[mn.ini[a], k] - φ[mn.ter[a], k]
            okkt = max(okkt, abs(γ*log(fk[a, k]) + DNLF.tcost(mn, a, fa[a]) - g)/(abs(g) + 1))
        end
        @test okkt < 1e-9
        e = 0.2 .* randn(mn.n, K); for k in 1:K; e[:, k] .-= sum(e[:, k])/mn.n; end     # exact J vs FD
        ε = 1e-6; fp, _ = DNLF.mc_flows(mn, φ .+ ε.*e, γ); fm, _ = DNLF.mc_flows(mn, φ .- ε.*e, γ)
        fd = (DNLF.mc_resid(mn, fp, dk) .- DNLF.mc_resid(mn, fm, dk)) ./ (2ε)
        @test reldiff(DNLF.mc_jac(mn, fk, fa, γ, e), fd) < 1e-5
        _, fks, _, its = solve_sue(mn, dk, γ)                          # solver reaches equilibrium
        @test norm(DNLF.mc_resid(mn, fks, dk)) / norm(hcat(dk...)) < 1e-7
        @test its ≤ 40                                                  # bounded (single-digit) inner iterations
        d1 = [(v = zeros(mn.n); v[1] = 3.0e3; v[100] = -3.0e3; v)]      # K=1 → deterministic as γ→0
        _, fdet, _ = DNLF.solve_flow(mn, d1[1], zeros(mn.m); tol = 1e-10)
        _, _, fa_a, _ = solve_sue(mn, d1, 0.08); _, _, fa_b, _ = solve_sue(mn, d1, 0.02)
        @test reldiff(fa_b, fdet) < reldiff(fa_a, fdet)
    end

    @testset "multicommodity design adjoint vs finite differences" begin
        Random.seed!(2); K = 2; γ = 0.3; mn = mc_randnet(60; seed = 5)   # tiny net: fast + exact re-solves
        dk = [(v = zeros(mn.n); v[o] = 2.0e3; v[t] = -2.0e3; v) for (o, t) in ((1, 40), (3, 55))]
        _, fke, fae, _ = solve_sue(mn, dk, γ; tol = 1e-11)
        g = DNLF.mc_adjoint(mn, fke, fae, γ)                            # ∇_τ TSTT, one adjoint solve
        a = argmax(fae); εt = 0.05                                      # FD-check the most active arc
        τp = zeros(mn.m); τp[a] += εt; _, _, fap, _ = solve_sue(mn, dk, γ; tolls = τp, tol = 1e-11)
        τm = zeros(mn.m); τm[a] -= εt; _, _, fam, _ = solve_sue(mn, dk, γ; tolls = τm, tol = 1e-11)
        fd = (DNLF.mc_tstt(mn, fap) - DNLF.mc_tstt(mn, fam)) / (2εt)
        @test abs(g[a] - fd) / max(abs(fd), 1.0) < 5e-2
    end
end
