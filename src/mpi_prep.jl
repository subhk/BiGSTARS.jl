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
    _eps_options(; sigma_0, nev, which, tol, maxiter, ncv, mat_solver, eps_type) -> String

Build the PETSc/SLEPc options-database string that configures one distributed
solve: Krylov-Schur EPS, shift-and-invert ST targeting `sigma_0`, with a parallel
LU (direct) factorization for the inner solves. Pure-Julia (no PETSc), so the
extension can consume it while CI verifies it.
"""
function _eps_options(; sigma_0, nev, which, tol, maxiter, ncv, mat_solver, eps_type)
    haskey(_WHICH_OPT, which) || throw(ArgumentError("unsupported which=$which"))
    opts = "-eps_type $(eps_type) " *
           "-eps_gen_non_hermitian " *                # generalized non-Hermitian pencil (EPS_GNHEP)
           "-eps_nev $(nev) " *
           "-eps_tol $(tol) " *
           "-eps_max_it $(maxiter) " *
           "-eps_target $(sigma_0) " *
           "-eps_$(_WHICH_OPT[which]) " *
           "-st_type sinvert " *
           "-st_pc_type lu " *
           "-st_pc_factor_mat_solver_type $(mat_solver) "
    ncv > 0 && (opts *= "-eps_ncv $(ncv) ")
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
