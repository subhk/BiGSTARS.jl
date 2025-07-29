using Test
# using SpecialFunctions
# using ModelingToolkit
# using NonlinearSolve
# using Parameters
# using StaticArrays

using BiGSTARS

@testset "Chebyshev Differentiation Tests" begin
    n = 16
    domain = [-1.0, 1.0]  # Standard Chebyshev domain
    cd = ChebyshevDiffn(n, domain, 4)  # Need 4th order derivatives
    
    # Test with complex polynomial: 2x^4 - 3x^3 + x^2 - 5x + 7
    f = 2 .* cd.x.^4 .- 3 .* cd.x.^3 .+ cd.x.^2 .- 5 .* cd.x .+ 7
    df_exact = 8 .* cd.x.^3 .- 9 .* cd.x.^2 .+ 2 .* cd.x .- 5
    d2f_exact = 24 .* cd.x.^2 .- 18 .* cd.x .+ 2
    d3f_exact = 48 .* cd.x .- 18
    d4f_exact = fill(48.0, length(cd.x))
    
    df_numerical = cd.D₁ * f
    d2f_numerical = cd.D₂ * f
    d3f_numerical = cd.D₃ * f
    d4f_numerical = cd.D₄ * f
    
    @test maximum(abs.(df_numerical - df_exact))   < 1e-8
    @test maximum(abs.(d2f_numerical - d2f_exact)) < 1e-8
    @test maximum(abs.(d3f_numerical - d3f_exact)) < 1e-8
    @test maximum(abs.(d4f_numerical - d4f_exact)) < 1e-8
end


@testset "Fourier Differentiation Tests" begin
    n = 32
    𝒟 = FourierDiffn(n)
    
    # Test function: sin(x)
    u = sin.(𝒟.x)
    
    # Test up to 6th derivative
    derivatives = [
        u,                    # 0th: sin(x)
        cos.(𝒟.x),           # 1st: cos(x)
        -sin.(𝒟.x),          # 2nd: -sin(x)
        -cos.(𝒟.x),          # 3rd: -cos(x)
        sin.(𝒟.x),           # 4th: sin(x)
        cos.(𝒟.x),           # 5th: cos(x)
        -sin.(𝒟.x)           # 6th: -sin(x)
    ]
    
    for m in 0:6
        computed = 𝒟[m] * u
        expected = derivatives[m+1]
        @test maximum(abs.(computed - expected)) < 1e-8
    end
end

#
@testset "Stone1971" begin
    
     include("Stone1971.jl")

     @test solve_Stone1971(0.1)
end

# @testset "rotating RBC" begin
#
#     include("rRBC.jl")
#
#     @test solve_rRBC(0.0)
# end