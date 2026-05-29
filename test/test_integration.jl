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

    @testset "Sparse solve path matches dense path" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=24, lower=-1.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)
        @equation prob sigma * u == -dx(dx(u)) - dz(dz(u))
        @bc prob left(u) == 0
        @bc prob right(u) == 0
        cache = discretize(prob)
        ks = [0.5, 1.5]

        rd = solve(cache, ks; sigma_0=3.0, method=:Arnoldi, sparse=false)
        rs = solve(cache, ks; sigma_0=3.0, method=:Arnoldi, sparse=true)

        for i in eachindex(ks)
            @test rd[i].converged && rs[i].converged
            @test abs(rd[i].eigenvalues[1] - rs[i].eigenvalues[1]) < 1e-6
        end
    end

    @testset "Variable-coefficient field multiply stays banded (Olver–Townsend)" begin
        mkc(N) = begin
            d = Domain(x=FourierTransformed(), z=Chebyshev(N=N, lower=-1.0, upper=1.0))
            p = EVP(d, variables=[:u], eigenvalue=:sigma)
            z = gridpoints(d, :z); p[:U] = @. tanh(4z)
            @equation p sigma * u == U * dz(u) - dz(dz(u))
            @bc p left(u) == 0
            @bc p right(u) == 0
            discretize(p)
        end

        # Banded C^(λ) multiplication: assembled fill falls with N and routes
        # sparse at high N. The old dense S·M·S⁻¹ form plateaued near 0.26.
        @test BiGSTARS._assembled_density(mkc(768)) < 0.12

        # Physics preserved: smallest positive-real eigenvalue is unchanged.
        A, B = assemble(mkc(128), 1.0)
        ev = eigvals(Matrix(A), Matrix(B))
        λ = sort(filter(e -> isfinite(e) && real(e) > 1e-3, ev), by=abs)[1]
        @test abs(λ - 1.6880521668) < 1e-6
    end

    @testset "Density gate: sparse for banded, dense for filled" begin
        # constant-coefficient 2D Laplacian → banded/block → very sparse
        db = Domain(x=FourierTransformed(), y=Fourier(16, [0, 1.0]), z=Chebyshev(12, [0, 1.0]))
        pb = EVP(db, variables=[:u], eigenvalue=:sigma)
        @equation pb sigma * u == -dx(dx(u)) - dy(dy(u)) - dz(dz(u))
        @bc pb left(u) == 0
        @bc pb right(u) == 0
        cb = discretize(pb)
        @test BiGSTARS._assembled_density(cb) < 0.10        # → auto-selects sparse

        # rich variable coefficient → fills in → above threshold
        df = Domain(x=FourierTransformed(), z=Chebyshev(N=64, lower=-1.0, upper=1.0))
        pf = EVP(df, variables=[:u], eigenvalue=:sigma)
        z = gridpoints(df, :z); pf[:U] = @. tanh(4z)
        @equation pf sigma * u == U * dz(u) - dz(dz(u))
        @bc pf left(u) == 0
        @bc pf right(u) == 0
        cf = discretize(pf)
        @test BiGSTARS._assembled_density(cf) > 0.10        # → auto-selects dense
    end

end
