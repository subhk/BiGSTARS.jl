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

    @testset "_csr_block_nnz_split counts diagonal vs off-diagonal nnz" begin
        A = sparse(ComplexF64[
            10  0  20;
             0 30   0;
            40  0 50])
        rowptr, colind, vals = BiGSTARS._to_csr(A)

        # Whole matrix, single rank: every entry is "diagonal" (cols [0,3)).
        d, o = BiGSTARS._csr_block_nnz_split(rowptr, colind, 0, 3, 0, 3)
        @test d == Int32[2, 1, 2]
        @test o == Int32[0, 0, 0]

        # Diagonal columns [0,2): col 2 falls in the off-diagonal block.
        d2, o2 = BiGSTARS._csr_block_nnz_split(rowptr, colind, 0, 3, 0, 2)
        @test d2 == Int32[1, 1, 1]            # row0 col0 | row1 col1 | row2 col0
        @test o2 == Int32[1, 0, 1]            # row0 col2 | row1 —    | row2 col2

        # An owned sub-block of rows [1,3) with diagonal cols [1,3).
        d3, o3 = BiGSTARS._csr_block_nnz_split(rowptr, colind, 1, 3, 1, 3)
        @test d3 == Int32[1, 1]               # row1 col1 | row2 col2
        @test o3 == Int32[0, 1]               # row1 —    | row2 col0

        # d_nnz + o_nnz always equals the per-row total nnz.
        for (rs, re) in ((0, 3), (1, 3), (0, 1))
            dd, oo = BiGSTARS._csr_block_nnz_split(rowptr, colind, rs, re, 1, 2)
            for i in 1:(re - rs)
                g = rs + i                    # 1-based global row
                @test dd[i] + oo[i] == rowptr[g + 1] - rowptr[g]
            end
        end
    end

    @testset "sparse_from_csr is the inverse of _to_csr" begin
        A = sprand(ComplexF64, 10, 10, 0.3)
        csr = BiGSTARS._to_csr(A)
        @test BiGSTARS.sparse_from_csr(csr) ≈ A
    end

    @testset "_sigma_schedule" begin
        s = BiGSTARS._sigma_schedule(1.0, 3, 0.2, 1.2)
        @test s[1] == 1.0
        @test length(s) == 1 + 2 * 3
        @test s[2] ≈ 1.0 + 0.2 * 1.0       # first up step
        @test s[3] ≈ 1.0 + 0.2 * 1.2       # second up step (×incre)
        @test s[5] ≈ 1.0 - 0.2 * 1.0       # first down step
        # schedule scales with |σ₀|; n_tries=0 → just [σ₀]
        @test BiGSTARS._sigma_schedule(-2.0, 1, 0.5, 1.0) == [-2.0, -1.0, -3.0]
        @test BiGSTARS._sigma_schedule(3.0, 0, 0.2, 1.2) == [3.0]
        # σ₀ = 0: base floors to 1.0 so the retry shifts still vary (not all zero)
        @test BiGSTARS._sigma_schedule(0.0, 2, 0.2, 1.2) == [0.0, 0.2, 0.24, -0.2, -0.24]
    end

    @testset "solve without the extension throws an install hint" begin
        # Cross-platform env: PetscWrap/SlepcWrap are NOT imported, so the extension
        # is inactive and the base fallback fires.
        err = try
            solve(nothing, [0.0]; sigma_0=1.0)
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

    @testset "_group_indices round-robin partition" begin
        # 5 wavenumbers, 2 groups
        @test BiGSTARS._group_indices(5, 2, 0) == [1, 3, 5]
        @test BiGSTARS._group_indices(5, 2, 1) == [2, 4]
        # 1 group gets everything
        @test BiGSTARS._group_indices(4, 1, 0) == [1, 2, 3, 4]
        # every index assigned exactly once, across all groups (coverage + disjoint)
        let nk = 7, ng = 3
            all_idx = reduce(vcat, BiGSTARS._group_indices(nk, ng, g) for g in 0:(ng-1))
            @test sort(all_idx) == collect(1:nk)
        end
        # empty group (more groups than wavenumbers)
        @test BiGSTARS._group_indices(2, 4, 3) == Int[]
    end

    @testset "_eps_options builds the SLEPc options string (no numeric target)" begin
        s = BiGSTARS._eps_options(; nev=5, which=:LM, tol=1e-10,
                                  maxiter=300, ncv=0, mat_solver="mumps",
                                  eps_type="krylovschur")
        @test occursin("-eps_type krylovschur", s)
        @test occursin("-eps_gen_non_hermitian", s)    # generalized non-Hermitian
        @test occursin("-eps_nev 5", s)
        @test occursin("-eps_target_magnitude", s)     # which flag stays
        @test !occursin("-eps_target ", s)             # numeric target removed (set via EPSSetTarget)
        @test occursin("-st_type sinvert", s)
        @test occursin("-st_pc_factor_mat_solver_type mumps", s)
        @test !occursin("-eps_ncv", s)                 # ncv=0 omitted

        s2 = BiGSTARS._eps_options(; nev=2, which=:SR, tol=1e-8,
                                   maxiter=100, ncv=20, mat_solver="superlu_dist",
                                   eps_type="krylovschur")
        @test occursin("-eps_ncv 20", s2)              # ncv>0 included
        @test occursin("-eps_smallest_real", s2)

        # Unsupported `which` is rejected.
        @test_throws ArgumentError BiGSTARS._eps_options(;
            nev=1, which=:XX, tol=1e-10, maxiter=10, ncv=0,
            mat_solver="mumps", eps_type="krylovschur")
    end
end
