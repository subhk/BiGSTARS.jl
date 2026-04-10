using Test
using BiGSTARS: BinaryOpNode, EigenvalueNode, SubstitutionNode

@testset "DSL Macros" begin

    @testset "@substitution" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=10, lower=0.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)

        @substitution prob Lap(A) = dx(dx(A)) + dz(dz(A))

        @test haskey(prob.substitutions, :Lap)
        @test prob.substitutions[:Lap].arg_names == [:A]
        @test prob.substitutions[:Lap].body isa BinaryOpNode
    end

    @testset "@equation" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=10, lower=0.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)
        prob[:U] = ones(10)
        prob[:E] = 1e-12

        @equation prob sigma * psi == U * dx(psi) - E * dz(dz(psi))

        @test length(prob.equations) == 1
        eq = prob.equations[1]
        @test eq.lhs isa BinaryOpNode
        @test eq.lhs.op == :*
        @test eq.lhs.left isa EigenvalueNode
        @test eq.rhs isa BinaryOpNode
    end

    @testset "@equation with substitution" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=10, lower=0.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)
        prob[:U] = ones(10)

        @substitution prob Lap(A) = dx(dx(A)) + dz(dz(A))
        @equation prob sigma * Lap(psi) == U * dx(Lap(psi))

        @test length(prob.equations) == 1
        # LHS should contain SubstitutionNode for Lap (expanded at discretize-time)
        @test prob.equations[1].lhs.right isa SubstitutionNode
    end

    @testset "@bc" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=10, lower=0.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:psi, :b], eigenvalue=:sigma)
        prob[:Ri] = 1.0

        @bc prob left(psi) == 0
        @bc prob right(dz(psi)) == 0
        @bc prob left(b) == 0
        @bc prob right(psi) == 1.0

        @test length(prob.bcs) == 4
        @test prob.bcs[1].side == :left
        @test prob.bcs[1].rhs == 0.0
        @test prob.bcs[2].side == :right
        @test prob.bcs[4].rhs == 1.0
        # All should default to :z coordinate
        @test all(bc.coord == :z for bc in prob.bcs)
    end

    @testset "Coupled system" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=10, lower=0.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:psi, :b], eigenvalue=:sigma)
        prob[:U] = ones(10)
        prob[:E] = 1e-12

        @substitution prob Lap(A) = dx(dx(A)) + dz(dz(A))

        @equation prob sigma * Lap(psi) == U * dx(Lap(psi)) - E * Lap(Lap(psi)) + dz(b)
        @equation prob sigma * b == U * dx(b) - E * Lap(b)

        @test length(prob.equations) == 2
    end

end
