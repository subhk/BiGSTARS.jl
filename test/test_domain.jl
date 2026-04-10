using Test
using SparseArrays
using BiGSTARS: FourierBasisSpec, ChebyshevBasisSpec, get_diff_operator,
    get_conversion_operator, total_grid_size, conversion_operator

@testset "Domain Construction" begin

    @testset "3D domain (x=transformed, y=Fourier, z=Chebyshev)" begin
        domain = Domain(
            x = FourierTransformed(),
            y = Fourier(N=16, L=1.0),
            z = Chebyshev(N=10, lower=0.0, upper=1.0)
        )

        @test length(domain.coords) == 3
        @test domain.coords[:x] isa FourierTransformed
        @test domain.coords[:y] isa FourierBasisSpec
        @test domain.coords[:z] isa ChebyshevBasisSpec

        @test Set(domain.resolved_dims) == Set([:y, :z])
        @test domain.transformed_dims == [:x]

        @test domain.coords[:y].N == 16
        @test domain.coords[:z].N == 10
    end

    @testset "Grid points" begin
        domain = Domain(
            x = FourierTransformed(),
            y = Fourier(N=16, L=2.0),
            z = Chebyshev(N=10, lower=0.0, upper=1.0)
        )

        zpts = gridpoints(domain, :z)
        @test length(zpts) == 10
        @test zpts[1] ≈ 1.0   # Chebyshev descending
        @test zpts[end] ≈ 0.0

        ypts = gridpoints(domain, :y)
        @test length(ypts) == 16
        @test ypts[1] ≈ 0.0

        ypts2, zpts2 = gridpoints(domain, :y, :z)
        @test ypts2 == ypts
        @test zpts2 == zpts
    end

    @testset "2D domain (x=transformed, z=Chebyshev)" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=20, lower=-1.0, upper=1.0)
        )

        @test length(domain.resolved_dims) == 1
        @test domain.resolved_dims == [:z]
        @test total_grid_size(domain) == 20
    end

    @testset "Multiple FourierTransformed directions" begin
        domain = Domain(
            x = FourierTransformed(),
            y = FourierTransformed(),
            z = Chebyshev(N=20, lower=0.0, upper=1.0)
        )

        @test length(domain.transformed_dims) == 2
        @test total_grid_size(domain) == 20
    end

    @testset "Differentiation operators from domain" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=16, lower=0.0, upper=1.0)
        )

        D1 = get_diff_operator(domain, :z, 1)
        @test size(D1) == (16, 16)
        @test D1 isa SparseMatrixCSC

        D2 = get_diff_operator(domain, :z, 2)
        @test size(D2) == (16, 16)
    end

    @testset "Conversion operators from domain" begin
        domain = Domain(z = Chebyshev(N=16, lower=-1.0, upper=1.0))

        S01 = get_conversion_operator(domain, :z, 0, 1)
        @test size(S01) == (16, 16)

        S02 = get_conversion_operator(domain, :z, 0, 2)
        @test size(S02) == (16, 16)

        # S_0->2 should equal S_1 * S_0
        S0 = conversion_operator(0, 16)
        S1 = conversion_operator(1, 16)
        @test S02 ≈ S1 * S0
    end

end
