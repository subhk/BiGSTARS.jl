using Test
using LinearAlgebra

# The serial eigensolvers (KrylovKit/ArnoldiMethod/Arpack) were removed when
# SLEPc/PETSc became the sole backend. The distributed solve is exercised by the
# complex-PETSc CI job (test/mpi/test_slepc.jl). What remains here are the
# backend-independent spectral utilities that still ship in the package.

@testset "Eigen utilities (retained)" begin
    @testset "sort_eigenvalues! :nearest requires σ" begin
        @test_throws ArgumentError BiGSTARS.sort_eigenvalues!(
            ComplexF64[1.0, 2.0], Matrix{ComplexF64}(I, 2, 2), :nearest)
    end

    @testset "sort_eigenvalues! non-nearest criteria" begin
        λ = ComplexF64[3.0 + 0im, 1.0 + 2im, 2.0 - 1im]
        Χ = Matrix{ComplexF64}(I, 3, 3)
        ls, _ = BiGSTARS.sort_eigenvalues!(copy(λ), copy(Χ), :R; rev=true)
        @test real(ls[1]) == 3.0
        lm, _ = BiGSTARS.sort_eigenvalues!(copy(λ), copy(Χ), :M; rev=true)
        @test abs(lm[1]) ≈ maximum(abs.(λ))
    end

    @testset "eigenvalue utilities (sort/remove/print)" begin
        λ = ComplexF64[3.0 + 0im, 1.0 + 2im, 2.0 - 1im]
        Χ = Matrix{ComplexF64}(I, 3, 3)

        ls, _ = sort_evals(λ, Χ, :R; rev=true)              # Symbol method, real desc
        @test real(ls[1]) == 3.0
        ls2, _ = sort_evals(λ, Χ, "M"; sorting="lm")        # String method, magnitude
        @test abs(ls2[1]) ≈ maximum(abs.(λ))

        lr, χr = remove_evals(λ, Χ, 1.5, 3.5, "R")          # keep real ∈ [1.5,3.5] → drop 1+2im
        @test length(lr) == 2 && all(e -> 1.5 ≤ real(e) ≤ 3.5, lr)
        @test size(χr, 2) == 2
        @test length(remove_evals(λ, Χ, 1.5, 3.5, :R)[1]) == 2   # Symbol convenience

        @test (print_evals(λ); true)                        # runs without error
    end
end
