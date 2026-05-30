using Test
using BiGSTARS
using SparseArrays
using LinearAlgebra

@testset "MPI prep helpers" begin
    @testset "_to_csr returns row-major CSR of a sparse matrix" begin
        # 3x3 with known pattern
        A = sparse(ComplexF64[
            10  0  20;
             0 30   0;
            40  0 50])
        rowptr, colind, vals = BiGSTARS._to_csr(A)

        # CSR: 0-based rowptr length N+1, contiguous per row
        @test rowptr == Int32[0, 2, 3, 5]
        @test colind == Int32[0, 2, 1, 0, 2]            # 0-based column indices, row order
        @test vals == ComplexF64[10, 20, 30, 40, 50]
    end
end
