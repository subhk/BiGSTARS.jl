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

    @testset "_place_in_block_rows matches place_in_block row-slice" begin
    Npv, Nvars = 4, 3
    N = Npv * Nvars
    mat = sparse(ComplexF64[ (10i + j) for i in 1:Npv, j in 1:Npv ])  # dense-ish block
    full = BiGSTARS.place_in_block(mat, 2, 3, Nvars, Npv)             # eq 2, var 3
    for (rs, re) in ((0, N), (Npv, 2Npv), (0, Npv), (5, 7), (2Npv, N))
        got = BiGSTARS._place_in_block_rows(mat, 2, 3, Npv, Nvars, rs, re)
        @test got == full[(rs+1):re, :]
    end
    # block that does not overlap the owned rows → all-zero slice
    z = BiGSTARS._place_in_block_rows(mat, 1, 1, Npv, Nvars, 2Npv, N)
    @test nnz(z) == 0 && size(z) == (N - 2Npv, N)
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

    @testset "restrict_cache_rows + restricted assemble_rows" begin
    function mk_aug()
        dom = Domain(z=Chebyshev(N=12, lower=0.0, upper=1.0))
        p = EVP(dom, variables=[:psi], eigenvalue=:sigma)
        @derive p v dz(dz(v)) = psi
        @derive_bc p v left(v) == 0
        @derive_bc p v right(v) == 0
        @equation p sigma * psi == v
        @bc p left(psi) == 0
        @bc p right(psi) == 0
        discretize(p; augment_derived=true)
    end
    function mk_plain()
        dom = Domain(x=FourierTransformed(), z=Chebyshev(N=12, lower=0.0, upper=1.0))
        p = EVP(dom, variables=[:u], eigenvalue=:sigma)
        @equation p sigma * u == -dx(dx(u)) - dz(dz(u))
        @bc p left(u) == 0
        @bc p right(u) == 0
        discretize(p)
    end
    for (mk, k) in ((mk_plain, 1.3), (mk_aug, 0.0))
        cache = mk(); N = cache.N_total
        @test cache.row_range === nothing
        for (rs, re) in ((0, N), (0, cld(N,2)), (cld(N,2), N), (N-1, N))
            rc = BiGSTARS.restrict_cache_rows(cache, rs, re)
            @test rc.row_range == (rs, re)
            Ar, Br = BiGSTARS.assemble_rows(rc, k, rs, re)                  # restricted (direct sum)
            Af, Bf = BiGSTARS.assemble_rows(cache, k, rs, re)              # full (2a slice)
            @test Ar ≈ Af && Br ≈ Bf
            @test_throws ArgumentError BiGSTARS.assemble_rows(rc, k, 0, re ÷ 2)  # range mismatch
        end
        # double restriction is rejected
        rc = BiGSTARS.restrict_cache_rows(cache, 0, N)
        @test_throws ArgumentError BiGSTARS.restrict_cache_rows(rc, 0, N)
    end
end

    @testset "assemble_rows == assemble row-slice" begin
        function mkcache_plain()
            dom = Domain(x=FourierTransformed(), z=Chebyshev(N=12, lower=0.0, upper=1.0))
            p = EVP(dom, variables=[:u], eigenvalue=:sigma)
            @equation p sigma * u == -dx(dx(u)) - dz(dz(u))
            @bc p left(u) == 0
            @bc p right(u) == 0
            discretize(p)
        end
        function mkcache_augmented()
            dom = Domain(z=Chebyshev(N=12, lower=0.0, upper=1.0))
            p = EVP(dom, variables=[:psi], eigenvalue=:sigma)
            @derive p v dz(dz(v)) = psi
            @derive_bc p v left(v) == 0
            @derive_bc p v right(v) == 0
            @equation p sigma * psi == v
            @bc p left(psi) == 0
            @bc p right(psi) == 0
            discretize(p; augment_derived=true)
        end
        function mkcache_legacy()
            dom = Domain(x=FourierTransformed(), y=Fourier(8,[0.0,1.0]), z=Chebyshev(8,[0.0,1.0]))
            p = EVP(dom, variables=[:w,:zeta], eigenvalue=:sigma)
            @derive p v dx(dx(v)) + dy(dy(v)) = dy(dz(w)) - dx(zeta)
            @equation p sigma * w == v - dz(dz(w))
            @equation p sigma * zeta == dz(w)
            @bc p left(w) == 0; @bc p right(w) == 0
            @bc p left(dz(zeta)) == 0; @bc p right(dz(zeta)) == 0
            discretize(p; augment_derived=false)
        end
        for (mk, k) in ((mkcache_plain, 1.3), (mkcache_augmented, 0.0), (mkcache_legacy, 1.0))
            cache = mk()
            Afull, Bfull = assemble(cache, k)
            N = cache.N_total
            for (rs, re) in ((0, N), (0, 0), (0, cld(N,3)), (cld(N,3), 2*cld(N,3)), (N-1, N))
                A_rows, B_rows = BiGSTARS.assemble_rows(cache, k, rs, re)
                @test size(A_rows) == (re - rs, N) && size(B_rows) == (re - rs, N)
                @test A_rows ≈ Afull[(rs+1):re, :]
                @test B_rows ≈ Bfull[(rs+1):re, :]
            end
            @test BiGSTARS._assemble_B_full(cache, k) ≈ Bfull
        end
    end

    @testset "_keep_by_mass keeps physical, drops near-zero" begin
        @test BiGSTARS._keep_by_mass([1.0, 1e-12, 2.0]) == [1, 3]    # middle is spurious (≈0)
        @test BiGSTARS._keep_by_mass([1.0, 2.0, 3.0]) == [1, 2, 3]   # all physical
        @test BiGSTARS._keep_by_mass([5.0]) == [1]                   # single
        @test BiGSTARS._keep_by_mass(Float64[]) == Int[]            # empty
        @test BiGSTARS._keep_by_mass([0.0, 0.0]) == [1, 2]          # all-zero → keep all (fallback)
        @test BiGSTARS._keep_by_mass([1.0, 0.0, 1.0]) == [1, 3]     # exact-zero dropped
    end

    @testset "_union_block_nnz covers every per-k pattern" begin
        dom = Domain(x=FourierTransformed(), z=Chebyshev(N=12, lower=0.0, upper=1.0))
        p = EVP(dom, variables=[:u], eigenvalue=:sigma)
        @equation p sigma * u == -dx(dx(u)) - dz(dz(u))
        @bc p left(u) == 0; @bc p right(u) == 0
        cache = discretize(p)
        N = cache.N_total
        d, o = BiGSTARS._union_block_nnz(cache.A_components, 0, N, N)
        @test length(d) == N && length(o) == N
        @test all(==(0), o)                                   # single rank ⇒ all columns diagonal
        for k in (0.5, 1.0, 7.0)
            Ak, _ = BiGSTARS.assemble_rows(cache, k, 0, N)
            rp, _, _ = BiGSTARS._to_csr(Ak)
            for r in 1:N
                @test d[r] + o[r] >= rp[r + 1] - rp[r]        # union ⊇ per-k pattern
            end
        end
        Uref = sum(abs.(M) for M in values(cache.A_components))
        rpu, _, _ = BiGSTARS._to_csr(Uref)
        for r in 1:N
            @test d[r] + o[r] == rpu[r + 1] - rpu[r]          # exactly the OR of all components
        end
        # sub-range (multi-rank scenario): exercises the M[rstart+1:rend,:] slice branch and
        # a partial diagonal column block ⇒ some off-diagonal counts must be nonzero.
        rs, re = 0, cld(N, 2)
        ds, os = BiGSTARS._union_block_nnz(cache.A_components, rs, re, N)
        @test length(ds) == re - rs && length(os) == re - rs
        @test any(>(0), os)                                   # owned cols [0,N/2) ⇒ remote cols exist
        for (lr, gr) in enumerate(rs:(re - 1))
            @test ds[lr] + os[lr] == rpu[gr + 2] - rpu[gr + 1]   # union row nnz, sliced row gr (0-based)
        end
    end

    @testset "_eps_options reuse_factorization flag" begin
        base = BiGSTARS._eps_options(; nev=1, which=:LM, tol=1e-10, maxiter=300, ncv=0,
                                     mat_solver="mumps", eps_type="krylovschur")
        @test !occursin("reuse_ordering", base)
        on = BiGSTARS._eps_options(; nev=1, which=:LM, tol=1e-10, maxiter=300, ncv=0,
                                   mat_solver="mumps", eps_type="krylovschur",
                                   reuse_factorization=true)
        @test occursin("-st_pc_factor_reuse_ordering true", on)
        @test occursin("-st_pc_factor_reuse_fill true", on)
    end
end

@testset "_petsc_ownership matches PETSC_DECIDE split" begin
    for (N, P) in ((10,1),(10,2),(10,3),(7,3),(12,4),(5,5),(5,8),(1,1),(0,2))
        ranges = [BiGSTARS._petsc_ownership(N, P, r) for r in 0:(P-1)]
        @test ranges[1][1] == 0                         # starts at 0
        @test ranges[end][2] == N                       # ends at N
        for r in 1:(P-1)
            @test ranges[r][2] == ranges[r+1][1]        # contiguous, no gaps/overlap
        end
        sizes = [re - rs for (rs, re) in ranges]
        @test sum(sizes) == N                           # covers exactly N rows
        @test maximum(sizes) - minimum(sizes) ≤ 1       # balanced to within 1
    end
end
