using Test
using BiGSTARS: VarNode, ParamNode, DerivNode, BinaryOpNode, WavenumberNode,
    lower_derivatives, contains_wavenumber, any_deriv_in_direction,
    max_deriv_order, count_chained_derivs, unwrap_chained_derivs, collect_var_names

@testset "Derivative Lowering" begin

    @testset "dx(psi) → im * k * psi" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=10, lower=0.0, upper=1.0)
        )
        input = DerivNode(VarNode(:psi), :x)
        result = lower_derivatives(input, domain)

        @test contains_wavenumber(result)
        @test !any_deriv_in_direction(result, :x)
    end

    @testset "dz(psi) stays as DerivNode" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=10, lower=0.0, upper=1.0)
        )
        input = DerivNode(VarNode(:psi), :z)
        result = lower_derivatives(input, domain)

        @test result isa DerivNode
        @test result.coord == :z
        @test !contains_wavenumber(result)
    end

    @testset "dx(dx(psi)) → double wavenumber" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=10, lower=0.0, upper=1.0)
        )
        input = DerivNode(DerivNode(VarNode(:psi), :x), :x)
        result = lower_derivatives(input, domain)

        @test !any_deriv_in_direction(result, :x)
        @test contains_wavenumber(result)
    end

    @testset "Mixed: dx(dz(psi))" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=10, lower=0.0, upper=1.0)
        )
        input = DerivNode(DerivNode(VarNode(:psi), :z), :x)
        result = lower_derivatives(input, domain)

        # dx should be lowered, dz should remain
        @test contains_wavenumber(result)
        @test any_deriv_in_direction(result, :z)
        @test !any_deriv_in_direction(result, :x)
    end

    @testset "max_deriv_order" begin
        psi = VarNode(:psi)
        @test max_deriv_order(psi, :z) == 0
        @test max_deriv_order(DerivNode(psi, :z), :z) == 1
        @test max_deriv_order(DerivNode(DerivNode(psi, :z), :z), :z) == 2

        expr = BinaryOpNode(:+,
            DerivNode(DerivNode(psi, :z), :z),
            DerivNode(DerivNode(DerivNode(DerivNode(psi, :z), :z), :z), :z)
        )
        @test max_deriv_order(expr, :z) == 4
    end

    @testset "count_chained_derivs and unwrap" begin
        psi = VarNode(:psi)
        d1 = DerivNode(psi, :z)
        d2 = DerivNode(d1, :z)
        d3 = DerivNode(d2, :z)

        @test count_chained_derivs(d1, :z) == 1
        @test count_chained_derivs(d2, :z) == 2
        @test count_chained_derivs(d3, :z) == 3

        @test unwrap_chained_derivs(d3, :z) == psi
    end

    @testset "collect_var_names" begin
        expr = BinaryOpNode(:+,
            BinaryOpNode(:*, ParamNode(:U), VarNode(:psi)),
            DerivNode(VarNode(:b), :z)
        )
        vars = collect_var_names(expr)
        @test vars == Set([:psi, :b])
    end

end
