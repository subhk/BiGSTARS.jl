using Test
using LinearAlgebra

@testset "Eigen solver interface" begin
    @testset "solve_eigenvalue_problem accepts verbose keyword" begin
        A = ComplexF64.(Diagonal([1.0, 2.0, 3.0, 4.0]))
        B = Matrix{ComplexF64}(I, 4, 4)

        λ, Χ = BiGSTARS.solve_eigenvalue_problem(A, B;
            method=:Arpack, σ₀=2.5, verbose=false, n_tries=0, nev=1)

        @test length(λ) == 1
        @test size(Χ, 1) == 4
    end
end
