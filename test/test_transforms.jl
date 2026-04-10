using Test
using FFTW

@testset "Transform Round-Trips" begin

    @testset "Chebyshev: physical -> coefficients -> physical" begin
        N = 32
        x = chebyshev_points(N)
        f = @. sin(π * x)
        c = chebyshev_coefficients(f)
        f_reconstructed = chebyshev_evaluate(c, x)
        @test maximum(abs.(f_reconstructed - f)) < 1e-10
    end

    @testset "Chebyshev: known polynomial" begin
        N = 16
        x = chebyshev_points(N)
        # f(x) = 3 + 2x - x^2 = 3*T_0 + 2*T_1 + (-1/2)*T_0 + (-1/2)*T_2
        #       = 2.5*T_0 + 2*T_1 - 0.5*T_2
        f = @. 3.0 + 2.0 * x - x^2
        c = chebyshev_coefficients(f)
        @test abs(c[1] - 2.5) < 1e-12
        @test abs(c[2] - 2.0) < 1e-12
        @test abs(c[3] - (-0.5)) < 1e-12
        for k in 4:N
            @test abs(c[k]) < 1e-12
        end
    end

    @testset "Fourier: physical -> coefficients -> physical" begin
        N = 32
        L = 2π
        x = fourier_points(N, L)
        f = @. sin(x) + 0.5 * cos(3x)
        c = to_coefficients(f, :fourier)
        f_reconstructed = to_physical(c, :fourier)
        @test maximum(abs.(f_reconstructed - f)) < 1e-10
    end

    @testset "chebyshev_evaluate with Clenshaw" begin
        # T_0 = 1 everywhere
        c = [1.0, 0.0, 0.0, 0.0]
        x = [-1.0, -0.5, 0.0, 0.5, 1.0]
        @test chebyshev_evaluate(c, x) ≈ ones(5)

        # T_1 = x
        c = [0.0, 1.0, 0.0, 0.0]
        @test chebyshev_evaluate(c, x) ≈ x

        # T_2 = 2x^2 - 1
        c = [0.0, 0.0, 1.0, 0.0]
        @test chebyshev_evaluate(c, x) ≈ @. 2x^2 - 1
    end

    @testset "differentiate: Chebyshev on [-1,1]" begin
        domain = Domain(z = Chebyshev(N=32, lower=-1.0, upper=1.0))
        z = gridpoints(domain, :z)

        # d/dz(z^3) = 3z^2
        f = z .^ 3
        df = differentiate(f, domain, :z)
        @test maximum(abs.(df - 3.0 .* z .^ 2)) < 1e-10

        # d²/dz²(z^3) = 6z
        d2f = differentiate(f, domain, :z; order=2)
        @test maximum(abs.(d2f - 6.0 .* z)) < 1e-8
    end

    @testset "differentiate: Chebyshev on [0,1]" begin
        domain = Domain(z = Chebyshev(N=32, lower=0.0, upper=1.0))
        z = gridpoints(domain, :z)

        # d/dz(sin(πz)) = π*cos(πz)
        f = sin.(π .* z)
        df = differentiate(f, domain, :z)
        @test maximum(abs.(df - π .* cos.(π .* z))) < 1e-8
    end

    @testset "differentiate: Fourier (no filter)" begin
        domain = Domain(y = Fourier(N=32, L=2π))
        y = gridpoints(domain, :y)

        # d/dy(sin(y)) = cos(y)
        f = sin.(y)
        df = differentiate(f, domain, :y)
        @test maximum(abs.(df - cos.(y))) < 1e-10

        # d²/dy²(sin(y)) = -sin(y)
        d2f = differentiate(f, domain, :y; order=2)
        @test maximum(abs.(d2f - (-sin.(y)))) < 1e-10
    end

    @testset "differentiate: Fourier with filters" begin
        domain = Domain(y = Fourier(N=32, L=2π))
        y = gridpoints(domain, :y)
        f = sin.(y)

        # Exponential filter — should still be accurate for well-resolved sin(y)
        df_exp = differentiate(f, domain, :y; filter=:exp)
        @test maximum(abs.(df_exp - cos.(y))) < 1e-8

        # 2/3 rule — sin(y) is mode 1, well within 2/3 cutoff
        df_23 = differentiate(f, domain, :y; filter=Symbol("2/3"))
        @test maximum(abs.(df_23 - cos.(y))) < 1e-10

        # Custom filter — identity (no effect)
        df_custom = differentiate(f, domain, :y; filter=(k, kmax) -> 1.0)
        @test maximum(abs.(df_custom - cos.(y))) < 1e-10
    end

end
