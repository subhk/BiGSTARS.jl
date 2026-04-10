using Test

# This test file is meant to be run via the full module (runtests.jl)
# where BiGSTARS is loaded. It tests the end-to-end DSL workflow.

@testset "End-to-End Integration" begin

    @testset "1D eigenvalue: -u'' = sigma*u" begin
        domain = Domain(z = Chebyshev(N=32, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)

        @equation prob sigma * u == -dz(dz(u))
        @bc prob left(u) == 0
        @bc prob right(u) == 0

        cache = discretize(prob)
        A, B = assemble(cache, 0.0)

        # Direct solve for verification
        lambdas = eigvals(Matrix(A), Matrix(B))
        real_pos = sort(filter(l -> isfinite(l) && abs(imag(l)) < 1e-6 && real(l) > 0.1, lambdas) .|> real)

        @test length(real_pos) >= 1
        @test abs(real_pos[1] - (π / 2)^2) < 0.01
    end

    @testset "Wavenumber sweep: sigma*u = -dx(dx(u)) - dz(dz(u))" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=32, lower=-1.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)

        @equation prob sigma * u == -dx(dx(u)) - dz(dz(u))
        @bc prob left(u) == 0
        @bc prob right(u) == 0

        cache = discretize(prob)

        # For each k, smallest eigenvalue should be (pi/2)^2 + k^2
        for k in [0.0, 1.0, 2.0]
            A, B = assemble(cache, k)
            lambdas = eigvals(Matrix(A), Matrix(B))
            real_pos = sort(filter(l -> isfinite(l) && abs(imag(l)) < 1e-6 && real(l) > 0.1, lambdas) .|> real)

            expected = (π / 2)^2 + k^2
            @test length(real_pos) >= 1
            @test abs(real_pos[1] - expected) / expected < 0.01
        end
    end

    @testset "Substitution in equation" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=16, lower=-1.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)

        @substitution prob D2(A) = dz(dz(A))
        @equation prob sigma * u == -D2(u)
        @bc prob left(u) == 0
        @bc prob right(u) == 0

        cache = discretize(prob)
        A, B = assemble(cache, 0.0)

        lambdas = eigvals(Matrix(A), Matrix(B))
        real_pos = sort(filter(l -> isfinite(l) && abs(imag(l)) < 1e-6 && real(l) > 0.1, lambdas) .|> real)

        @test length(real_pos) >= 1
        @test abs(real_pos[1] - (π / 2)^2) < 0.1
    end

end
