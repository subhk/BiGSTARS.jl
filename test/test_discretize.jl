using Test
using SparseArrays
using LinearAlgebra
using BiGSTARS: conversion_operator, differentiation_operator, get_conversion_operator,
    total_grid_size, discretize, assemble, allocate_workspace, assemble!,
    DiscretizationCache, AssemblyWorkspace

@testset "Discretize and Assemble" begin

    @testset "1D: -u'' = sigma*u, u(-1)=u(1)=0" begin
        domain = Domain(z = Chebyshev(N=32, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)

        @equation prob sigma * u == -dz(dz(u))
        @bc prob left(u) == 0
        @bc prob right(u) == 0

        cache = discretize(prob)
        A, B = assemble(cache, 0.0)

        lambdas = eigvals(Matrix(A), Matrix(B))
        real_pos = sort(filter(l -> isfinite(l) && abs(imag(l)) < 1e-6 && real(l) > 0.1, lambdas) .|> real)

        @test length(real_pos) >= 1
        @test abs(real_pos[1] - (π / 2)^2) < 0.01
    end

    @testset "k-separation caching" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=32, lower=-1.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)

        @equation prob sigma * u == -dx(dx(u)) - dz(dz(u))
        @bc prob left(u) == 0
        @bc prob right(u) == 0

        cache = discretize(prob)

        # Should have k^0 and k^2 components in A
        @test haskey(cache.A_components, 0)
        @test haskey(cache.A_components, 2)

        # B should only have k^0
        @test haskey(cache.B_components, 0)

        # A at different k values should differ
        A1, B1 = assemble(cache, 1.0)
        A2, B2 = assemble(cache, 2.0)
        @test !(A1 ≈ A2)
        @test B1 ≈ B2
    end

    @testset "1D on non-standard interval [0, 1]" begin
        domain = Domain(z = Chebyshev(N=32, lower=0.0, upper=1.0))
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)

        @equation prob sigma * u == -dz(dz(u))
        @bc prob left(u) == 0
        @bc prob right(u) == 0

        cache = discretize(prob)
        A, B = assemble(cache, 0.0)

        lambdas = eigvals(Matrix(A), Matrix(B))
        real_pos = sort(filter(l -> isfinite(l) && abs(imag(l)) < 1e-6 && real(l) > 0.1, lambdas) .|> real)

        # On [0,1]: eigenvalues are (n*pi)^2 for n=1,2,...
        @test length(real_pos) >= 1
        @test abs(real_pos[1] - π^2) / π^2 < 0.01
    end

    @testset "Scalar parameter in equation" begin
        domain = Domain(z = Chebyshev(N=32, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)
        prob[:E] = 2.0

        # sigma*u = -E*dz(dz(u))  → eigenvalues should be E*(n*pi/2)^2
        @equation prob sigma * u == -E * dz(dz(u))
        @bc prob left(u) == 0
        @bc prob right(u) == 0

        cache = discretize(prob)
        A, B = assemble(cache, 0.0)

        lambdas = eigvals(Matrix(A), Matrix(B))
        real_pos = sort(filter(l -> isfinite(l) && abs(imag(l)) < 1e-6 && real(l) > 0.1, lambdas) .|> real)

        expected = 2.0 * (π / 2)^2
        @test length(real_pos) >= 1
        @test abs(real_pos[1] - expected) / expected < 0.01
    end

    @testset "Field parameter × derivative" begin
        N = 32
        domain = Domain(z = Chebyshev(N=N, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)

        # U = constant 1 field → same as no parameter
        prob[:U] = ones(N)

        @equation prob sigma * u == -U * dz(dz(u))
        @bc prob left(u) == 0
        @bc prob right(u) == 0

        cache = discretize(prob)
        A, B = assemble(cache, 0.0)

        lambdas = eigvals(Matrix(A), Matrix(B))
        real_pos = sort(filter(l -> isfinite(l) && abs(imag(l)) < 1e-6 && real(l) > 0.1, lambdas) .|> real)

        @test length(real_pos) >= 1
        @test abs(real_pos[1] - (π / 2)^2) / (π / 2)^2 < 0.01
    end

    @testset "Negative scalar parameter" begin
        domain = Domain(z = Chebyshev(N=32, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)
        prob[:nu] = 0.5

        # sigma*u = -nu * dz(dz(u)) with nu=0.5
        # eigenvalues should be 0.5 * (nπ/2)^2
        @equation prob sigma * u == -nu * dz(dz(u))
        @bc prob left(u) == 0
        @bc prob right(u) == 0

        cache = discretize(prob)
        A, B = assemble(cache, 0.0)

        lambdas = eigvals(Matrix(A), Matrix(B))
        real_pos = sort(filter(l -> isfinite(l) && abs(imag(l)) < 1e-6 && real(l) > 0.1, lambdas) .|> real)

        expected = 0.5 * (π / 2)^2
        @test length(real_pos) >= 1
        @test abs(real_pos[1] - expected) / expected < 0.01
    end

    @testset "Coupled 2-variable system" begin
        domain = Domain(z = Chebyshev(N=16, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:u, :v], eigenvalue=:sigma)

        # Decoupled system: sigma*u = -dz(dz(u)), sigma*v = -dz(dz(v))
        # Eigenvalues should be doubled: each of (n*pi/2)^2 appears twice
        @equation prob sigma * u == -dz(dz(u))
        @equation prob sigma * v == -dz(dz(v))
        @bc prob left(u) == 0
        @bc prob right(u) == 0
        @bc prob left(v) == 0
        @bc prob right(v) == 0

        cache = discretize(prob)
        A, B = assemble(cache, 0.0)

        @test cache.N_total == 32  # 16 per variable * 2

        lambdas = eigvals(Matrix(A), Matrix(B))
        real_pos = sort(filter(l -> isfinite(l) && abs(imag(l)) < 1e-6 && real(l) > 0.1, lambdas) .|> real)

        # First eigenvalue should still be (pi/2)^2, appearing at least twice
        @test length(real_pos) >= 2
        @test abs(real_pos[1] - (π / 2)^2) / (π / 2)^2 < 0.01
        @test abs(real_pos[2] - (π / 2)^2) / (π / 2)^2 < 0.01
    end

    @testset "In-place assembly (assemble!)" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=16, lower=-1.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)

        @equation prob sigma * u == -dx(dx(u)) - dz(dz(u))
        @bc prob left(u) == 0
        @bc prob right(u) == 0

        cache = discretize(prob)
        ws = allocate_workspace(cache)

        for k in [0.0, 1.0, 2.0]
            A_alloc, B_alloc = assemble(cache, k)
            assemble!(ws, cache, k)

            @test ws.A ≈ Matrix(A_alloc)
            @test ws.B ≈ Matrix(B_alloc)
        end
    end

    @testset "Mixed BCs: Dirichlet + Neumann" begin
        domain = Domain(z = Chebyshev(N=32, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)

        # -u'' = sigma*u with u(-1)=0, u'(1)=0
        # Exact eigenvalues: ((2n-1)*pi/4)^2 for n=1,2,...
        @equation prob sigma * u == -dz(dz(u))
        @bc prob left(u) == 0
        @bc prob right(dz(u)) == 0

        cache = discretize(prob)
        A, B = assemble(cache, 0.0)

        lambdas = eigvals(Matrix(A), Matrix(B))
        real_pos = sort(filter(l -> isfinite(l) && abs(imag(l)) < 1e-6 && real(l) > 0.1, lambdas) .|> real)

        expected = (π / 4)^2
        @test length(real_pos) >= 1
        @test abs(real_pos[1] - expected) / expected < 0.01
    end

end
