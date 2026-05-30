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

    @testset "solve_mpi without the extension throws an install hint" begin
        err = try
            BiGSTARS.solve_mpi(nothing, [0.0]; sigma_0=1.0)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("SlepcWrap", err.msg)
    end

    @testset "_WHICH_OPT maps BiGSTARS which-symbols to SLEPc options" begin
        @test BiGSTARS._WHICH_OPT[:LM] == "target_magnitude"
        @test BiGSTARS._WHICH_OPT[:LR] == "largest_real"
        @test BiGSTARS._WHICH_OPT[:SR] == "smallest_real"
        @test BiGSTARS._WHICH_OPT[:LI] == "largest_imaginary"
        @test BiGSTARS._WHICH_OPT[:SI] == "smallest_imaginary"
    end

    @testset "_eps_options builds the SLEPc options string" begin
        s = BiGSTARS._eps_options(; sigma_0=0.5, nev=5, which=:LM, tol=1e-10,
                                  maxiter=300, ncv=0, mat_solver="mumps",
                                  eps_type="krylovschur")
        @test occursin("-eps_type krylovschur", s)
        @test occursin("-eps_nev 5", s)
        @test occursin("-eps_target 0.5", s)
        @test occursin("-eps_target_magnitude", s)
        @test occursin("-st_type sinvert", s)
        @test occursin("-st_pc_factor_mat_solver_type mumps", s)
        @test !occursin("-eps_ncv", s)                 # ncv=0 omitted

        s2 = BiGSTARS._eps_options(; sigma_0=1.0, nev=2, which=:SR, tol=1e-8,
                                   maxiter=100, ncv=20, mat_solver="superlu_dist",
                                   eps_type="krylovschur")
        @test occursin("-eps_ncv 20", s2)              # ncv>0 included
        @test occursin("-eps_smallest_real", s2)

        # Unsupported `which` is rejected.
        @test_throws ArgumentError BiGSTARS._eps_options(;
            sigma_0=0.0, nev=1, which=:XX, tol=1e-10, maxiter=10, ncv=0,
            mat_solver="mumps", eps_type="krylovschur")
    end
end
