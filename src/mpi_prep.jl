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

function solve_mpi end
