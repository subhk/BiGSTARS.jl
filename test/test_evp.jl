using Test
using BiGSTARS: ExprNode, VarNode, ParamNode, ConstNode, EigenvalueNode,
    WavenumberNode, DerivNode, BinaryOpNode, SubstitutionNode,
    Substitution, Equation, BoundaryCondition,
    add_equation!, add_bc!, add_substitution!, add_derived!, add_derived_bc!,
    first_chebyshev_coord

@testset "EVP Problem Type" begin

    @testset "Construction" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=10, lower=0.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:psi, :b], eigenvalue=:sigma)

        @test prob.domain === domain
        @test prob.variables == [:psi, :b]
        @test prob.eigenvalue == :sigma
        @test isempty(prob.parameters)
        @test isempty(prob.equations)
        @test isempty(prob.bcs)
        @test isempty(prob.substitutions)
    end

    @testset "Parameter setting" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=10, lower=0.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)

        prob[:E] = 1e-12
        @test prob[:E] == 1e-12

        z = gridpoints(domain, :z)
        prob[:U] = z .- 0.5
        @test prob[:U] ≈ z .- 0.5
    end

    @testset "Name conflict checks" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=10, lower=0.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)

        @test_throws ArgumentError (prob[:psi] = 1.0)
        @test_throws ArgumentError (prob[:sigma] = 1.0)
    end

    @testset "add_equation! and add_bc!" begin
        domain = Domain(z = Chebyshev(N=10, lower=0.0, upper=1.0))
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)

        lhs = BinaryOpNode(:*, EigenvalueNode(:sigma), VarNode(:u))
        rhs = DerivNode(DerivNode(VarNode(:u), :z), :z)
        add_equation!(prob, lhs, rhs)
        @test length(prob.equations) == 1

        add_bc!(prob, :left, :z, VarNode(:u), 0.0)
        add_bc!(prob, :right, :z, VarNode(:u), 0.0)
        @test length(prob.bcs) == 2
        @test prob.bcs[1].side == :left
        @test prob.bcs[2].side == :right
    end

    @testset "add_substitution!" begin
        domain = Domain(z = Chebyshev(N=10, lower=0.0, upper=1.0))
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)

        body = BinaryOpNode(:+,
            DerivNode(DerivNode(VarNode(:A), :x), :x),
            DerivNode(DerivNode(VarNode(:A), :z), :z)
        )
        add_substitution!(prob, :Lap, [:A], body)

        @test haskey(prob.substitutions, :Lap)
        @test prob.substitutions[:Lap].arg_names == [:A]
    end

    @testset "first_chebyshev_coord" begin
        domain = Domain(
            x = FourierTransformed(),
            y = Fourier(N=16, L=1.0),
            z = Chebyshev(N=10, lower=0.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)
        @test first_chebyshev_coord(prob) == :z
    end

    @testset "derived BC preserves derivative order" begin
        domain = Domain(z = Chebyshev(N=10, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)

        add_derived!(prob, :v, :Op, VarNode(:u))
        add_derived_bc!(prob, :v, :left, :z, 2, 0.0)

        @test BiGSTARS.count_bc_deriv_order(prob.derived_vars[:v].bcs[1].expr) == 2
    end

end
