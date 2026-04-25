using Test
using BiGSTARS: WavenumberNode, VarNode, ConstNode, DerivNode, BinaryOpNode,
    is_k_dependent, separate_additive_terms, extract_k_power, extract_k_powers,
    separate_by_k_power

@testset "Wavenumber Separation" begin

    @testset "is_k_dependent" begin
        @test is_k_dependent(WavenumberNode(:k_x))
        @test !is_k_dependent(VarNode(:psi))
        @test !is_k_dependent(DerivNode(VarNode(:psi), :z))
        @test is_k_dependent(BinaryOpNode(:*, WavenumberNode(:k_x), VarNode(:psi)))
    end

    @testset "separate_additive_terms" begin
        a = VarNode(:a)
        b = VarNode(:b)
        c = VarNode(:c)

        # a + b
        expr = BinaryOpNode(:+, a, b)
        terms = separate_additive_terms(expr)
        @test length(terms) == 2

        # a + b - c → [a, b, -1*c]
        expr = BinaryOpNode(:-, BinaryOpNode(:+, a, b), c)
        terms = separate_additive_terms(expr)
        @test length(terms) == 3
    end

    @testset "extract_k_power" begin
        psi = VarNode(:psi)
        k = WavenumberNode(:k_x)

        # No k: power = 0
        power, reduced = extract_k_power(psi)
        @test power == 0
        @test reduced == psi

        # k * psi: power = 1
        power, reduced = extract_k_power(BinaryOpNode(:*, k, psi))
        @test power == 1

        # k * k * psi: power = 2
        power, reduced = extract_k_power(BinaryOpNode(:*, k, BinaryOpNode(:*, k, psi)))
        @test power == 2

        # im * k * psi: power = 1 (im stays in reduced)
        expr = BinaryOpNode(:*, ConstNode(im), BinaryOpNode(:*, k, psi))
        power, reduced = extract_k_power(expr)
        @test power == 1
    end

    @testset "extract_k_powers preserves wavenumber names" begin
        psi = VarNode(:psi)
        kx = WavenumberNode(:k_x)
        ky = WavenumberNode(:k_y)

        expr = BinaryOpNode(:*, kx, BinaryOpNode(:*, ky, BinaryOpNode(:*, kx, psi)))
        powers, reduced = extract_k_powers(expr)

        @test powers == (:k_x => 2, :k_y => 1)
        @test reduced == psi
    end

    @testset "separate_by_k_power" begin
        psi = VarNode(:psi)
        k = WavenumberNode(:k_x)

        # im*k*psi + dz(dz(psi)) → k^1 + k^0
        term_k1 = BinaryOpNode(:*, ConstNode(im), BinaryOpNode(:*, k, psi))
        term_k0 = DerivNode(DerivNode(psi, :z), :z)
        expr = BinaryOpNode(:+, term_k1, term_k0)

        k_terms = separate_by_k_power(expr)
        @test length(k_terms) == 2

        powers = [kt.k_power for kt in k_terms]
        @test 0 in powers
        @test 1 in powers
    end

    @testset "separate_by_k_power keeps per-direction powers" begin
        psi = VarNode(:psi)
        kx = WavenumberNode(:k_x)
        ky = WavenumberNode(:k_y)

        expr = BinaryOpNode(:+,
            BinaryOpNode(:*, kx, BinaryOpNode(:*, kx, psi)),
            BinaryOpNode(:*, ky, BinaryOpNode(:*, ky, psi))
        )

        k_terms = separate_by_k_power(expr)
        @test (:k_x => 2,) in [kt.k_powers for kt in k_terms]
        @test (:k_y => 2,) in [kt.k_powers for kt in k_terms]
    end

end
