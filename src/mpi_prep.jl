# ==============================================================================
# mpi_prep.jl - Pure-Julia matrix preparation for the distributed SLEPc backend.
#
# No PETSc/SLEPc/MPI dependency: these helpers convert a serially-assembled
# SparseMatrixCSC into row-major CSR (0-based, PETSc convention) and extract
# contiguous row-blocks for scatter to MPI ranks. Unit-tested without PETSc;
# the package extension (ext/BiGSTARSMPIExt.jl) consumes their output.
# ==============================================================================

"""
    _to_csr(A::SparseMatrixCSC) -> (rowptr::Vector{Int32}, colind::Vector{Int32}, vals::Vector)

Convert `A` to 0-based CSR arrays (PETSc convention). Implemented by transposing
into CSC of `Aᵀ`, whose column storage is exactly the row storage of `A`, so the
per-row column indices come out sorted and contiguous.
"""
function _to_csr(A::SparseMatrixCSC)
    At = sparse(transpose(A))          # CSC of Aᵀ == CSR of A
    rowptr = Int32.(At.colptr .- 1)    # 0-based
    colind = Int32.(At.rowval .- 1)    # 0-based column indices
    vals = copy(At.nzval)
    return rowptr, colind, vals
end

"""
    _csr_row_block(rowptr, colind, vals, rstart::Integer, rend::Integer)
        -> (local_rowptr::Vector{Int32}, local_colind::Vector{Int32}, local_vals)

Extract the contiguous global row range `[rstart, rend)` (0-based, half-open) from
0-based CSR arrays. Column indices stay global (PETSc `MatSetValues` takes global
columns); `local_rowptr` is re-based to start at 0. Sized for one MPI rank's owned
rows, ready to scatter and insert.
"""
function _csr_row_block(rowptr::AbstractVector{<:Integer},
                        colind::AbstractVector{<:Integer},
                        vals::AbstractVector,
                        rstart::Integer, rend::Integer)
    nrows = rend - rstart
    nrows ≥ 0 || throw(ArgumentError("rend ($rend) < rstart ($rstart)"))
    kstart = rowptr[rstart + 1]                 # 0-based offset of first owned entry
    kend = rowptr[rend + 1]                     # 0-based offset past last owned entry
    local_rowptr = Vector{Int32}(undef, nrows + 1)
    @inbounds for r in 0:nrows
        local_rowptr[r + 1] = Int32(rowptr[rstart + r + 1] - kstart)
    end
    local_colind = Int32.(@view colind[(kstart + 1):kend])
    local_vals = collect(@view vals[(kstart + 1):kend])
    return local_rowptr, local_colind, local_vals
end

"""
    _csr_block_nnz_split(rowptr, colind, rstart, rend, cstart, cend)
        -> (d_nnz::Vector{Int32}, o_nnz::Vector{Int32})

Per-row nonzero counts for the owned global row range `[rstart, rend)` (0-based,
half-open), split by the diagonal column-ownership range `[cstart, cend)`:
`d_nnz[i]` counts entries whose column lies in `[cstart, cend)` (the on-process
"diagonal block" of a PETSc `MatMPIAIJ`), `o_nnz[i]` counts the rest (off-diagonal
block). Exact counts — feeding these to `MatMPIAIJSetPreallocation` means PETSc
never has to grow a row mid-insert. For a single rank pass `cstart=0, cend=N`
(every entry is diagonal). Pure-Julia, so it is unit-tested without PETSc.
"""
function _csr_block_nnz_split(rowptr::AbstractVector{<:Integer},
                              colind::AbstractVector{<:Integer},
                              rstart::Integer, rend::Integer,
                              cstart::Integer, cend::Integer)
    nrows = rend - rstart
    nrows ≥ 0 || throw(ArgumentError("rend ($rend) < rstart ($rstart)"))
    d_nnz = Vector{Int32}(undef, nrows)
    o_nnz = Vector{Int32}(undef, nrows)
    @inbounds for i in 1:nrows
        k0 = rowptr[rstart + i] + 1          # 0-based CSR offset → 1-based Julia
        k1 = rowptr[rstart + i + 1]
        d = 0; o = 0
        for k in k0:k1
            c = colind[k]                    # 0-based global column index
            (cstart ≤ c < cend) ? (d += 1) : (o += 1)
        end
        d_nnz[i] = Int32(d); o_nnz[i] = Int32(o)
    end
    return d_nnz, o_nnz
end

"""
    _union_pattern(components, rstart, rend, N) -> SparseMatrixCSC{Float64,Int}

Union sparsity pattern of all k-power `components` (`Dict{KPowerKey,SparseMatrixCSC}`) over
the owned global rows `[rstart,rend)`, as an `nrows×N` real sparse matrix (positive values via
`abs.` accumulation — so additive overlap never cancels a structural nonzero). Every
per-wavenumber matrix is `Σ c_p(k)·components[p]`, whose pattern ⊆ this union. Pure-Julia.
"""
function _union_pattern(components, rstart::Integer, rend::Integer, N::Integer)
    nrows = rend - rstart
    nrows ≥ 0 || throw(ArgumentError("rend ($rend) < rstart ($rstart)"))
    U = spzeros(Float64, nrows, N)
    for (_, M) in components
        block = size(M, 1) == nrows ? M : M[(rstart + 1):rend, :]   # restricted vs full cache
        U = U + abs.(block)
    end
    return U
end

"""
    _union_block_nnz(components, rstart, rend, N) -> (d_nnz, o_nnz)

PETSc `d_nnz`/`o_nnz` for the union pattern (see [`_union_pattern`](@ref)) — so a Mat
preallocated from these counts covers every wavenumber's per-k pattern (no mid-insert growth
when refilling values in place). May over-count by structural zeros (conservative — safe for
preallocation). Pure-Julia.
"""
function _union_block_nnz(components, rstart::Integer, rend::Integer, N::Integer)
    U = _union_pattern(components, rstart, rend, N)
    rowptr, colind, _ = _to_csr(U)
    return _csr_block_nnz_split(rowptr, colind, 0, rend - rstart, rstart, rend)
end

"""
    _union_template(components, rstart, rend, N) -> SparseMatrixCSC{ComplexF64,Int}

The union pattern (see [`_union_pattern`](@ref)) with all-zero `ComplexF64` values. Adding it
to a per-wavenumber owned-row slice forces the slice to carry the FULL union structure
(explicit zeros where that wavenumber's own entries are absent), so an `INSERT_VALUES` refill
overwrites every union slot — no stale values from a previous wavenumber, and no need for
`MatZeroEntries` (which PetscWrap 0.1.5 does not wrap). Pure-Julia.
"""
function _union_template(components, rstart::Integer, rend::Integer, N::Integer)
    U = _union_pattern(components, rstart, rend, N)
    return SparseMatrixCSC(size(U, 1), size(U, 2), U.colptr, U.rowval,
                           zeros(ComplexF64, length(U.nzval)))
end

"""
    _group_indices(nk, ngroups, group_id) -> Vector{Int}

1-based indices of `1:nk` assigned to group `group_id` (0-based) under a
round-robin split into `ngroups` groups: index `i` goes to group `(i-1) % ngroups`.
Pure-Julia, so the across-wavenumber routing is unit-tested without MPI. Returns
an empty vector when a group gets no work (more groups than wavenumbers).
"""
function _group_indices(nk::Integer, ngroups::Integer, group_id::Integer)
    return Int[i for i in 1:nk if (i - 1) % ngroups == group_id]
end

"""
    _petsc_ownership(N, nproc, rank) -> (rstart, rend)

PETSc's default contiguous row split (`PETSC_DECIDE`) for an `N`-row object over
`nproc` ranks: rank `r` owns `N÷nproc` rows, plus one extra for the first
`N % nproc` ranks. Returns the 0-based half-open owned range `[rstart, rend)`.
Pure-Julia replica of PETSc's `PetscSplitOwnership`, so a rank's owned rows can be
computed WITHOUT a PETSc probe (and unit-tested without MPI). The PETSc matrix
built later uses the same split; `assemble_rows` asserts the resulting range
matches the cache's `row_range`.
"""
function _petsc_ownership(N::Integer, nproc::Integer, rank::Integer)
    base = N ÷ nproc
    rem  = N % nproc
    rstart = rank * base + min(rank, rem)
    rend   = rstart + base + (rank < rem ? 1 : 0)
    return rstart, rend
end

"""
    _sigma_schedule(σ₀, n_tries, Δσ₀, incre) -> Vector{Float64}

Adaptive shift schedule: `σ₀` first, then `n_tries` geometrically growing
increments above and below it (`Δσ₀ * incre^(i-1) * base`, where `base = |σ₀|`,
floored to `1.0` when `σ₀ == 0` so the retries actually vary instead of all
re-targeting zero). Pure-Julia, so the adaptive-σ logic is unit-tested without
PETSc. Mirrors the schedule the old serial solver used.
"""
function _sigma_schedule(σ₀::Real, n_tries::Integer, Δσ₀::Real, incre::Real)
    base = abs(σ₀) > 0 ? abs(σ₀) : 1.0
    up = Float64[Δσ₀ * incre^(i - 1) * base for i in 1:n_tries]
    return vcat(Float64(σ₀), σ₀ .+ up, σ₀ .- up)
end

# Map BiGSTARS `which` to a SLEPc EPS option. `:LM` means "nearest the shift",
# matching the existing shift-and-invert convention (target magnitude). Used by
# the MPI extension; kept here (pure-Julia) so it is unit-testable without PETSc.
const _WHICH_OPT = Dict(
    :LM => "target_magnitude",
    :LR => "largest_real",
    :SR => "smallest_real",
    :LI => "largest_imaginary",
    :SI => "smallest_imaginary",
)

"""
    _eps_options(; nev, which, tol, maxiter, ncv, mat_solver, eps_type, reuse_factorization=false) -> String

Build the PETSc/SLEPc options-database string that configures one distributed
solve: Krylov-Schur EPS, shift-and-invert ST, with a parallel LU (direct)
factorization for the inner solves. The numeric target is NOT placed here — it
is set per attempt via `EPSSetTarget`, so the static options stay invariant across
σ attempts and wavenumbers. Pure-Julia (no PETSc), so it is unit-tested while CI
verifies the solve.
"""
function _eps_options(; nev, which, tol, maxiter, ncv, mat_solver, eps_type,
                      reuse_factorization::Bool=false)
    haskey(_WHICH_OPT, which) || throw(ArgumentError("unsupported which=$which"))
    opts = "-eps_type $(eps_type) " *
           "-eps_gen_non_hermitian " *                # generalized non-Hermitian pencil (EPS_GNHEP)
           "-eps_nev $(nev) " *
           "-eps_tol $(tol) " *
           "-eps_max_it $(maxiter) " *
           "-eps_$(_WHICH_OPT[which]) " *
           "-st_type sinvert " *
           "-st_pc_type lu " *
           "-st_pc_factor_mat_solver_type $(mat_solver) "
    ncv > 0 && (opts *= "-eps_ncv $(ncv) ")
    if reuse_factorization
        # Reuse the LU ordering/fill across re-factorizations of a same-pattern operator
        # (the k-sweep keeps the sparsity pattern fixed); MUMPS then redoes numeric only.
        opts *= "-st_pc_factor_reuse_ordering true -st_pc_factor_reuse_fill true "
    end
    return opts
end

"""
    sparse_from_csr(csr) -> SparseMatrixCSC

Rebuild a `SparseMatrixCSC` from a `(rowptr, colind, vals)` CSR tuple produced by
[`_to_csr`](@ref) (0-based indices). Used on rank 0 to recover the mass matrix `B`
for the singular-`B` mode filter. Pure-Julia (no PETSc), so it is CI-testable.
"""
function sparse_from_csr(csr)
    rowptr, colind, vals = csr
    N = length(rowptr) - 1
    I = Int[]; J = Int[]; V = eltype(vals)[]
    for r in 1:N
        for k in (rowptr[r] + 1):rowptr[r + 1]
            push!(I, r); push!(J, colind[k] + 1); push!(V, vals[k])
        end
    end
    return sparse(I, J, V, N, N)
end

"""
    _place_in_block_rows(mat, eq_idx, var_idx, N_per_var, N_vars, rstart, rend) -> SparseMatrixCSC

Row-restricted `place_in_block`: an `(rend-rstart) × (N_per_var*N_vars)` sparse
matrix equal to `place_in_block(mat, eq_idx, var_idx, N_vars, N_per_var)[rstart+1:rend, :]`.
`rstart`/`rend` are 0-based half-open global rows. Only the `eq_idx` block rows that
intersect `[rstart,rend)` carry `mat`'s rows, at the `var_idx` column block.
"""
function _place_in_block_rows(mat::AbstractMatrix, eq_idx::Integer, var_idx::Integer,
                              N_per_var::Integer, N_vars::Integer,
                              rstart::Integer, rend::Integer)
    N_total = N_per_var * N_vars
    nrows = rend - rstart
    bs = (eq_idx - 1) * N_per_var          # 0-based block row start
    be = eq_idx * N_per_var
    cs = (var_idx - 1) * N_per_var         # 0-based block col start
    lo = max(bs, rstart); hi = min(be, rend)
    I = Int[]; J = Int[]; V = ComplexF64[]
    if lo < hi
        sub = mat[(lo - bs + 1):(hi - bs), :]      # mat rows for the overlap (1-based)
        rv = rowvals(sub); nz = nonzeros(sub)
        for col in 1:size(sub, 2)
            for idx in nzrange(sub, col)
                push!(I, rv[idx] + (lo - rstart))  # 1-based row in the output slice
                push!(J, cs + col)                 # global column (1-based)
                push!(V, nz[idx])
            end
        end
    end
    return sparse(I, J, V, nrows, N_total)
end

"""
    restrict_cache_rows(cache, rstart, rend) -> DiscretizationCache

Return a cache whose components (`A_components`/`B_components`) are
sliced to the owned global rows `[rstart,rend)` (0-based half-open), with
`row_range=(rstart,rend)` set. `derived_caches` are kept full (k-dependent `H(k)`).
Errors if `cache` is already restricted. Pure (no MPI/PETSc).
"""
function restrict_cache_rows(cache, rstart::Integer, rend::Integer)
    cache.row_range === nothing ||
        throw(ArgumentError("cache already restricted to $(cache.row_range)"))
    function _slice_rows(d)
        out = empty(d)
        for (kp, M) in d
            out[kp] = M[(rstart + 1):rend, :]
        end
        return out
    end
    return DiscretizationCache(_slice_rows(cache.A_components), _slice_rows(cache.B_components),
        cache.derived_caches, cache.N_total, cache.N_per_var, cache.N_vars,
        cache.domain, cache.derived_var_order, (Int(rstart), Int(rend)))
end

"""
    assemble_rows(cache, k, rstart, rend) -> (A_rows, B_rows)

Owned-row slice of the assembled pencil: each is an `(rend-rstart) × N_total`
`SparseMatrixCSC` equal to `assemble(cache,k)[rstart+1:rend, :]`, built WITHOUT
materializing any full N×N matrix. `rstart`/`rend` are 0-based half-open global rows.
Mirrors `_assemble` (discretize.jl) term-by-term, sliced to the owned rows.
When `cache.row_range` is set (restricted cache), the k-components are already
sliced and are summed directly; the requested range must match exactly.
"""
function assemble_rows(cache, k::Float64,
                       rstart::Integer, rend::Integer)
    N = cache.N_total
    (0 ≤ rstart ≤ rend ≤ N) ||
        throw(ArgumentError("row range [$rstart,$rend) out of [0,$N]"))
    if cache.row_range !== nothing
        cache.row_range == (Int(rstart), Int(rend)) ||
            throw(ArgumentError("requested rows ($rstart,$rend) ≠ cache.row_range $(cache.row_range)"))
    end
    sliced = cache.row_range === nothing          # full cache → slice; restricted → use as-is
    k_vals = _k_values(cache, k)
    nrows = rend - rstart

    A_rows = spzeros(ComplexF64, nrows, N)
    for (kp, Ap) in cache.A_components
        c = _k_coeff(kp, k_vals); c == 0.0 && continue
        A_rows = A_rows + c * (sliced ? Ap[(rstart + 1):rend, :] : Ap)
    end

    B_rows = spzeros(ComplexF64, nrows, N)
    for (kp, Bp) in cache.B_components
        c = _k_coeff(kp, k_vals); c == 0.0 && continue
        B_rows = B_rows + c * (sliced ? Bp[(rstart + 1):rend, :] : Bp)
    end

    for (_, dc) in cache.derived_caches
        isempty(dc.terms) && continue
        op_k = dc.op_k0
        for (kp, mat) in dc.op_k_components
            c = _k_coeff(kp, k_vals); c == 0.0 && continue
            op_k = op_k + c * mat
        end
        H_k = _sparse_block_inverse(op_k, cache.domain; bcs=dc.bcs)
        for (eq_idx, var_idx, total_kp, coeff_mat, rhs_mat) in dc.terms
            w = _k_coeff(total_kp, k_vals); w == 0.0 && continue
            combined = coeff_mat * H_k * rhs_mat
            A_rows = A_rows + w * _place_in_block_rows(combined, eq_idx, var_idx,
                                                       cache.N_per_var, cache.N_vars,
                                                       rstart, rend)
        end
    end
    return A_rows, B_rows
end

"""
    _assemble_B_full(cache, k) -> SparseMatrixCSC

The full mass matrix `B = Σ_p k^p · B_components[p]` (B has no derived terms).
Cheap; built on the group root for the singular-`B` spurious-mode filter when the
distributed path means no rank holds a full `B`.
"""
function _assemble_B_full(cache, k::Float64)
    cache.row_range === nothing ||
        throw(ArgumentError("_assemble_B_full needs a full (unrestricted) cache; got " *
            "row_range=$(cache.row_range). Its B_components are row-sliced — use assemble_rows."))
    k_vals = _k_values(cache, k)
    N = cache.N_total
    B = spzeros(ComplexF64, N, N)
    for (kp, Bp) in cache.B_components
        c = _k_coeff(kp, k_vals); c == 0.0 && continue
        B = B + c * Bp
    end
    return B
end
