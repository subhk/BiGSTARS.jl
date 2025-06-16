using Test
# using SpecialFunctions
# using ModelingToolkit
# using NonlinearSolve
# using Parameters
# using StaticArrays

using BiGSTARS


# @testset "Stone1971" begin

#     include("Stone_1971.jl")

#     @test solve_Stone1971(0.1)
# end

@testset "rotating RBC" begin

    include("rRBC.jl")

    @test solve_rRBC(0.0)
end