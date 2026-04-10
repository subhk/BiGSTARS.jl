using Test
using SparseArrays
using LinearAlgebra
using BiGSTARS
using BiGSTARS: VarNode, DerivNode, BinaryOpNode, ConstNode, ParamNode,
    chebyshev_boundary_row, count_bc_deriv_order

@testset "Boundary Conditions" begin

    @testset "Dirichlet at left: T_n(-1) = (-1)^n" begin
        N = 8
        row = chebyshev_boundary_row(:left, 0, N)
        expected = [(-1.0)^(n - 1) for n in 1:N]
        @test row ≈ expected
    end

    @testset "Dirichlet at right: T_n(1) = 1" begin
        N = 8
        row = chebyshev_boundary_row(:right, 0, N)
        expected = ones(N)
        @test row ≈ expected
    end

    @testset "Neumann at left: T'_n(-1)" begin
        N = 8
        row = chebyshev_boundary_row(:left, 1, N)
        @test abs(row[1]) < 1e-14
        @test row[2] ≈ 1.0
        @test row[3] ≈ -4.0
    end

    @testset "Neumann at right: T'_n(1) = n^2" begin
        N = 8
        row = chebyshev_boundary_row(:right, 1, N)
        @test abs(row[1]) < 1e-14
        for n in 1:N-1
            @test row[n+1] ≈ Float64(n^2)
        end
    end

    @testset "Domain-scaled Neumann on [0,1]" begin
        N = 8
        row = chebyshev_boundary_row(:right, 1, N; a=0.0, b=1.0)
        @test abs(row[1]) < 1e-14
        @test row[2] ≈ 2.0   # 2 * 1^2
        @test row[3] ≈ 8.0   # 2 * 2^2
    end

    @testset "Second derivative BC at boundary" begin
        N = 8
        row = chebyshev_boundary_row(:right, 2, N)
        @test abs(row[1]) < 1e-14
        @test abs(row[2]) < 1e-14
        # T_n^(2)(1) = n^2 * (n^2-1) / 3
        @test row[3] ≈ 4.0 * (4 - 1) / 3  # n=2: 4
        @test row[4] ≈ 9.0 * (9 - 1) / 3  # n=3: 24
    end

    @testset "count_bc_deriv_order" begin
        psi = VarNode(:psi)
        @test count_bc_deriv_order(psi) == 0
        @test count_bc_deriv_order(DerivNode(psi, :z)) == 1
        @test count_bc_deriv_order(DerivNode(DerivNode(psi, :z), :z)) == 2
    end

    @testset "Robin BC: left(3*psi + dz(psi)) == 0" begin
        N = 8
        domain = Domain(z = Chebyshev(N=N, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)

        eval_row_0 = chebyshev_boundary_row(:left, 0, N)
        eval_row_1 = chebyshev_boundary_row(:left, 1, N)
        expected = 3.0 .* eval_row_0 .+ eval_row_1

        robin_expr = BinaryOpNode(:+,
            BinaryOpNode(:*, ConstNode(3.0), VarNode(:psi)),
            DerivNode(VarNode(:psi), :z)
        )
        row_vec = zeros(ComplexF64, N)
        BiGSTARS._build_bc_row_1d!(row_vec, robin_expr, :left, :z, prob)

        @test row_vec ≈ expected
    end

    @testset "Coupled BC: left(psi + b) == 0 (1D row)" begin
        # _build_bc_row_1d! produces a single N-length row with combined contributions
        # Multi-variable placement is handled by build_bc_rows
        N = 8
        domain = Domain(z = Chebyshev(N=N, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:psi, :b], eigenvalue=:sigma)

        eval_row = chebyshev_boundary_row(:left, 0, N)

        # psi + b: both are VarNodes, each contributes eval_row to the same 1D vector
        coupled_expr = BinaryOpNode(:+, VarNode(:psi), VarNode(:b))
        row_vec = zeros(ComplexF64, N)
        BiGSTARS._build_bc_row_1d!(row_vec, coupled_expr, :left, :z, prob)

        # Both variables add the same eval row → 2× the row
        @test row_vec ≈ 2.0 .* eval_row
    end

    @testset "Coupled BC with derivatives: left(dz(psi) - Ri*b) (1D row)" begin
        N = 8
        domain = Domain(z = Chebyshev(N=N, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:psi, :b], eigenvalue=:sigma)
        prob[:Ri] = 2.0

        eval_row_0 = chebyshev_boundary_row(:left, 0, N)
        eval_row_1 = chebyshev_boundary_row(:left, 1, N)

        # dz(psi) - Ri*b: deriv row for psi + (-Ri) * eval row for b
        coupled_expr = BinaryOpNode(:-,
            DerivNode(VarNode(:psi), :z),
            BinaryOpNode(:*, ParamNode(:Ri), VarNode(:b))
        )
        row_vec = zeros(ComplexF64, N)
        BiGSTARS._build_bc_row_1d!(row_vec, coupled_expr, :left, :z, prob)

        # Both contributions go into the same 1D row
        @test row_vec ≈ eval_row_1 .- 2.0 .* eval_row_0
    end

    @testset "Derivative of scaled variable: left(dz(3*psi)) == 0" begin
        N = 8
        domain = Domain(z = Chebyshev(N=N, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)

        eval_row_1 = chebyshev_boundary_row(:left, 1, N)
        expected = 3.0 .* eval_row_1

        # dz(3*psi) → should distribute to 3*dz(psi)
        expr = DerivNode(BinaryOpNode(:*, ConstNode(3.0), VarNode(:psi)), :z)
        row_vec = zeros(ComplexF64, N)
        BiGSTARS._build_bc_row_1d!(row_vec, expr, :left, :z, prob)

        @test row_vec ≈ expected
    end

    @testset "Higher-order: left(dz(dz(psi))) == 0" begin
        N = 8
        domain = Domain(z = Chebyshev(N=N, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)

        eval_row_2 = chebyshev_boundary_row(:left, 2, N)

        expr = DerivNode(DerivNode(VarNode(:psi), :z), :z)
        row_vec = zeros(ComplexF64, N)
        BiGSTARS._build_bc_row_1d!(row_vec, expr, :left, :z, prob)

        @test row_vec ≈ eval_row_2
    end

    @testset "Higher-order scaled: left(dz(dz(3*psi))) == 0" begin
        N = 8
        domain = Domain(z = Chebyshev(N=N, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)

        eval_row_2 = chebyshev_boundary_row(:left, 2, N)
        expected = 3.0 .* eval_row_2

        expr = DerivNode(DerivNode(BinaryOpNode(:*, ConstNode(3.0), VarNode(:psi)), :z), :z)
        row_vec = zeros(ComplexF64, N)
        BiGSTARS._build_bc_row_1d!(row_vec, expr, :left, :z, prob)

        @test row_vec ≈ expected
    end

    @testset "Inhomogeneous BC errors for eigenvalue problems" begin
        domain = Domain(z = Chebyshev(N=8, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)

        @equation prob sigma * u == -dz(dz(u))
        @bc prob left(u) == 0
        @bc prob right(u) == 1.0  # inhomogeneous

        @test_throws ErrorException discretize(prob)
    end

end
