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

    @testset "_csr_row_block extracts a contiguous global row range" begin
        A = sparse(ComplexF64[
            10  0  20;
             0 30   0;
            40  0 50])
        rowptr, colind, vals = BiGSTARS._to_csr(A)

        # rows [1,3) == global rows 1 and 2 (0-based half-open)
        lrp, lci, lv = BiGSTARS._csr_row_block(rowptr, colind, vals, 1, 3)

        @test lrp == Int32[0, 1, 3]                 # local rowptr, 2 rows
        @test lci == Int32[1, 0, 2]                 # global column indices preserved
        @test lv == ComplexF64[30, 40, 50]
    end

    @testset "_csr_row_block round-trips to the original matrix" begin
        A = sprand(ComplexF64, 12, 12, 0.3)
        rowptr, colind, vals = BiGSTARS._to_csr(A)

        # Partition rows into 3 contiguous blocks and reassemble
        ranges = [(0, 4), (4, 8), (8, 12)]
        I = Int[]; J = Int[]; V = ComplexF64[]
        for (rs, re) in ranges
            lrp, lci, lv = BiGSTARS._csr_row_block(rowptr, colind, vals, rs, re)
            for r in 1:(re - rs)
                for k in (lrp[r] + 1):lrp[r + 1]
                    push!(I, rs + r)                # 1-based global row
                    push!(J, lci[k] + 1)            # 1-based global col
                    push!(V, lv[k])
                end
            end
        end
        Arebuilt = sparse(I, J, V, 12, 12)
        @test Arebuilt ≈ A
    end
end
