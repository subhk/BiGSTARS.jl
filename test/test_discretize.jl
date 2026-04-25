using Test
using SparseArrays
using LinearAlgebra
using BiGSTARS: conversion_operator, differentiation_operator, get_conversion_operator,
    total_grid_size, discretize, assemble, allocate_workspace, assemble!,
    DiscretizationCache, AssemblyWorkspace, ParamNode, VarNode, BinaryOpNode,
    DerivedVarCache

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

    @testset "@compute-style field parameter multiplication" begin
        N = 8
        domain = Domain(z = Chebyshev(N=N, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)
        prob[:U] = ones(N)

        cache = DiscretizationCache(
            Dict{Int,SparseMatrixCSC{ComplexF64,Int}}(),
            Dict{Int,SparseMatrixCSC{ComplexF64,Int}}(),
            Dict{Symbol,DerivedVarCache}(),
            N, N, 1, domain
        )

        expr = BinaryOpNode(:*, ParamNode(:U), VarNode(:u))
        @test BiGSTARS._evaluate_expr(expr, prob, cache, ones(ComplexF64, N), 0.0) ≈ ones(ComplexF64, N)
    end

    @testset "@compute-style coefficient field multiplication" begin
        N = 8
        domain = Domain(z = Chebyshev(N=N, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)
        cache = DiscretizationCache(
            Dict{Int,SparseMatrixCSC{ComplexF64,Int}}(),
            Dict{Int,SparseMatrixCSC{ComplexF64,Int}}(),
            Dict{Symbol,DerivedVarCache}(),
            N, N, 1, domain
        )

        coeffs = zeros(ComplexF64, N)
        coeffs[1] = 1
        coeffs[3] = 0.5
        expr = BinaryOpNode(:*, VarNode(:u), VarNode(:u))
        result = BiGSTARS._evaluate_expr(expr, prob, cache, coeffs, 0.0)
        expected = BiGSTARS._complex_multiplication_operator(coeffs, N) * coeffs
        @test result ≈ expected
    end

    @testset "@compute-style 2D coefficient field multiplication" begin
        N_z = 4
        N_y = 4
        domain = Domain(y = Fourier(N=N_y, L=2π), z = Chebyshev(N=N_z, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)
        cache = DiscretizationCache(
            Dict{Int,SparseMatrixCSC{ComplexF64,Int}}(),
            Dict{Int,SparseMatrixCSC{ComplexF64,Int}}(),
            Dict{Symbol,DerivedVarCache}(),
            N_z * N_y, N_z * N_y, 1, domain
        )

        coeffs = zeros(ComplexF64, N_z * N_y)
        coeffs[1] = 1
        coeffs[N_z + 2] = 0.25
        expr = BinaryOpNode(:*, VarNode(:u), VarNode(:u))
        result = BiGSTARS._evaluate_expr(expr, prob, cache, coeffs, 0.0)
        expected = BiGSTARS._build_2d_coeff_multiply(coeffs, domain, :z, :y) * coeffs
        @test result ≈ expected
    end

    @testset "2D field parameter in T-basis operator discretization" begin
        N_z = 4
        N_y = 4
        domain = Domain(y = Fourier(N=N_y, L=2π), z = Chebyshev(N=N_z, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)
        prob[:U] = ones(N_z * N_y)

        M = BiGSTARS._discretize_operator(ParamNode(:U), nothing, prob, N_z * N_y)
        @test size(M) == (N_z * N_y, N_z * N_y)
        @test M * ones(ComplexF64, N_z * N_y) ≈ ones(ComplexF64, N_z * N_y)
    end

    @testset "multiple transformed direction assembly uses per-direction powers" begin
        domain = Domain(
            x = FourierTransformed(),
            y = FourierTransformed(),
            z = Chebyshev(N=32, lower=-1.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)

        @equation prob sigma * u == -dx(dx(u)) - dy(dy(u)) - dz(dz(u))
        @bc prob left(u) == 0
        @bc prob right(u) == 0

        cache = discretize(prob)
        @test haskey(cache.A_kcomponents, (:k_x => 2,))
        @test haskey(cache.A_kcomponents, (:k_y => 2,))

        A, B = assemble(cache; k_x=1.0, k_y=0.5)
        lambdas = eigvals(Matrix(A), Matrix(B))
        real_pos = sort(filter(l -> isfinite(l) && abs(imag(l)) < 1e-6 && real(l) > 0.1, lambdas) .|> real)

        expected = 1.0^2 + 0.5^2 + (π / 2)^2
        @test length(real_pos) >= 1
        @test abs(real_pos[1] - expected) / expected < 0.01
        @test_throws ErrorException assemble(cache; kx=1.0, k_y=0.5)
        @test_throws ErrorException assemble(cache; x=1.0, k_x=2.0, k_y=0.5)
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
