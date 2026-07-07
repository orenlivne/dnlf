using DNLF, Test, LinearAlgebra, SparseArrays, Random

net = DNLF.load_tntp_net(joinpath(pkgdir(DNLF), "data", "SiouxFalls", "SiouxFalls_net.tntp"))
r, s, D = 1, 20, 3.0e4                         # a congested single-OD instance (flow splits across paths)
d = zeros(net.n); d[r] = D; d[s] = -D
reldiff(a, b) = norm(a - b) / max(norm(b), 1e-30)

@testset "DNLF — full algorithm coverage" begin

    # ---- Phase 0: parsing & network construction ----
    @testset "load_tntp_net / DirectedNetwork" begin
        @test net.n == 24 && net.m == 76                       # Sioux Falls
        @test length(net.ini) == net.m == length(net.ter)
        @test size(net.B) == (net.n, net.m)
        @test all(sum(net.B; dims = 1) .== 0)                  # each arc column sums to 0 (−1 tail, +1 head)
        @test all(net.cap .> 0) && all(net.t0 .> 0)
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
        @test reldiff(fal, fll) < 1e-6                         # engine-agnostic solve_flow (lu = approxchol)
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
end
