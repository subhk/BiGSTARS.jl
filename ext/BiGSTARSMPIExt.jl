module BiGSTARSMPIExt

using BiGSTARS
using BiGSTARS: _to_csr, _csr_row_block, SolverResults, ConvergenceHistory,
                sort_eigenvalues!, _filter_physical_modes,
                _eps_options, sparse_from_csr
using MPI
using PetscWrap
using SlepcWrap
using SparseArrays
using LinearAlgebra: norm

# ==============================================================================
# BiGSTARSMPIExt - distributed (MPI) eigensolver backend over SLEPc/PETSc.
#
# Rank 0 assembles A,B serially with the existing BiGSTARS pipeline and converts
# them to 0-based CSR (BiGSTARS._to_csr). Each rank's owned row-block is scattered
# from rank 0 and inserted into a distributed PETSc MatMPIAIJ. SLEPc's EPS then
# solves the generalized pencil with a shift-and-invert spectral transform across
# all ranks. Eigenvectors are gathered to rank 0, where SolverResults is built.
#
# API target: PetscWrap 0.1.5 / SlepcWrap 0.1.x (the versions SlepcWrap requires).
# These expose the C-style names — MatCreate/MatSetSizes/MatSetValues/
# MatGetOwnershipRange/MatAssemblyBegin/MatCreateVecs (which RETURNS (vr, vi))/
# VecGetArray/EPSCreate/EPSGetEigenpair — matching the maintainer's Helmholtz
# example. There is no programmatic options inserter and no EPSSetDimensions/
# ST setter wrappers, so SLEPc is configured by the options string passed to
# `SlepcInitialize(opts)` at init time; `EPSSetFromOptions` then reads
# nev/tol/sinvert/mat_solver/target from the database. `solve_mpi` performs that
# initialization itself (manage_init).
#
# Every PETSc/SLEPc/MPI call name and signature here has been checked against
# source: PetscWrap 0.1.5 src (MatCreate/MatSetValues/MatGetOwnershipRange/
# MatCreateVecs→(vr,vi)/VecGetArray→(arr,ref)/VecRestoreArray(vec,ref)), SlepcWrap
# 0.1.3 (EPSCreate/EPSSetOperators/EPSGetEigenpair, options via SlepcInitialize),
# and MPI.jl v0.19.2 (positional send/recv, VBuffer Gatherv! with `nothing` on
# non-root). What remains unverified is only runtime numerics (convergence,
# eigenvector layout) — that needs a live complex PETSc/SLEPc, i.e. the CI job.
# ==============================================================================

# Tracks whether this extension initialized SLEPc (so repeated solve_mpi calls
# don't double-initialize). SLEPc/PETSc may only be initialized once per process.
const _SLEPC_INITED = Ref(false)

# ------------------------------------------------------------------------------
# Build the distributed PETSc matrix from scattered CSR row-blocks
# ------------------------------------------------------------------------------

"""
    _build_petsc_mat(A_csr, N, comm) -> PetscMat

Collectively build an N×N distributed `MatMPIAIJ` on `MPI.COMM_WORLD`. PETSc
decides the row ownership; rank 0 (which holds the full CSR `(rowptr, colind,
vals)`) ships each rank exactly its owned rows, which the rank inserts with
global column indices. Non-root ranks pass `nothing` for the CSR.
"""
function _build_petsc_mat(A_csr, N::Integer, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    M = MatCreate()                                # defaults to MPI.COMM_WORLD
    MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, N, N)
    MatSetFromOptions(M)
    MatSetUp(M)
    rstart, rend = MatGetOwnershipRange(M)         # 0-based [rstart, rend)

    # Gather every rank's ownership range to rank 0 (MPI 0.19: positional root).
    starts = MPI.Gather(Int(rstart), 0, comm)
    ends   = MPI.Gather(Int(rend),   0, comm)

    if rank == 0
        rowptr, colind, vals = A_csr
        # Send each non-root rank its CSR row-block; insert rank 0's own block directly.
        for p in 1:(nproc - 1)
            lrp, lci, lv = _csr_row_block(rowptr, colind, vals, starts[p + 1], ends[p + 1])
            MPI.send((lrp, lci, lv), p, 0, comm)            # MPI 0.19: send(obj, dest, tag, comm)
        end
        lrp, lci, lv = _csr_row_block(rowptr, colind, vals, rstart, rend)
        _insert_rows!(M, rstart, lrp, lci, lv)
    else
        (lrp, lci, lv), _ = MPI.recv(0, 0, comm)            # MPI 0.19: recv(src, tag, comm) -> (obj, status)
        _insert_rows!(M, rstart, lrp, lci, lv)
    end

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY)
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY)
    return M
end

"""
    _insert_rows!(M, rstart, local_rowptr, local_colind, local_vals)

Insert one rank's owned rows into the distributed matrix `M` via `MatSetValues`,
one row at a time, using global (0-based) row and column indices. `MatSetValues`
sets the `rows × cols` block, so a single global row index plus its column list
inserts that whole row.
"""
function _insert_rows!(M, rstart::Integer,
                       local_rowptr, local_colind, local_vals)
    nrows = length(local_rowptr) - 1
    for r in 1:nrows
        k0 = local_rowptr[r] + 1
        k1 = local_rowptr[r + 1]
        k1 < k0 && continue                        # empty row
        row  = PetscInt(rstart + r - 1)            # global 0-based row
        cols = PetscInt.(local_colind[k0:k1])      # global 0-based columns
        vs   = PetscScalar.(local_vals[k0:k1])
        MatSetValues(M, [row], cols, vs, INSERT_VALUES)
    end
    return M
end

# ------------------------------------------------------------------------------
# Run the SLEPc EPS solve (options already in the database via SlepcInitialize)
# ------------------------------------------------------------------------------

"""
    _solve_one(A_csr, B_csr, N, comm; sigma_0) -> SolverResults

Distributed solve of one generalized pencil `A x = λ B x`. SLEPc options
(`nev`, `tol`, shift-and-invert ST, target, direct solver) were placed in the
options database at `SlepcInitialize` time, so `EPSSetFromOptions` configures the
solver. Returns a populated `SolverResults` on rank 0 and an empty marker result
on other ranks. `A_csr`/`B_csr` are the rank-0 CSR tuples (or `nothing` off-root).
"""
function _solve_one(A_csr, B_csr, N::Integer, comm::MPI.Comm; sigma_0)
    rank = MPI.Comm_rank(comm)
    t0 = MPI.Wtime()

    A = _build_petsc_mat(A_csr, N, comm)
    B = _build_petsc_mat(B_csr, N, comm)

    eps = EPSCreate()
    EPSSetOperators(eps, A, B)
    EPSSetFromOptions(eps)
    EPSSolve(eps)

    nconv = EPSGetConverged(eps)
    λ, Χ = _gather_eigenpairs(eps, A, nconv, N, comm)

    solve_time = MPI.Wtime() - t0
    result = if rank == 0
        _assemble_results(λ, Χ, B_csr, sigma_0, nconv, solve_time)
    else
        SolverResults(ComplexF64[], zeros(ComplexF64, 0, 0), false,
                      :Slepc, Float64(sigma_0), 0, solve_time, ConvergenceHistory())
    end

    EPSDestroy(eps)
    MatDestroy(A)
    MatDestroy(B)
    return result
end

# ------------------------------------------------------------------------------
# Gather eigenpairs to rank 0 and assemble SolverResults
# ------------------------------------------------------------------------------

"""
    _gather_eigenpairs(eps, A, nconv, N, comm) -> (λ::Vector{ComplexF64}, Χ::Matrix{ComplexF64})

Pull the `nconv` converged eigenpairs onto rank 0. Eigenvalues are scalars
replicated on every rank. Each eigenvector is a distributed `Vec`; its locally
owned slice is gathered to rank 0 with `MPI.Gatherv!` using the matrix row
ownership counts. Non-root ranks get empty arrays.
"""
function _gather_eigenpairs(eps, A, nconv::Integer, N::Integer, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    rstart, rend = MatGetOwnershipRange(A)
    nlocal = rend - rstart
    counts = Cint.(MPI.Allgather(Int(nlocal), comm))   # local sizes per rank, on every rank

    λ = rank == 0 ? Vector{ComplexF64}(undef, nconv) : ComplexF64[]
    Χ = rank == 0 ? Matrix{ComplexF64}(undef, N, nconv) : zeros(ComplexF64, 0, 0)

    vr, vi = MatCreateVecs(A)                      # vectors compatible with A (returns a pair)
    for ie in 0:(nconv - 1)
        vpr, vpi, _, _ = EPSGetEigenpair(eps, ie, vr, vi)
        # PetscWrap 0.1.5 VecGetArray returns (array, ref); the array is
        # PetscScalar (ComplexF64 on a complex build) and aliases PETSc memory,
        # so copy it out before restoring, and restore with the ref (not the array).
        local_arr, local_ref = VecGetArray(vr)     # this rank's owned entries
        sendbuf = Vector{ComplexF64}(local_arr)
        if rank == 0
            recvbuf = Vector{ComplexF64}(undef, N)
            MPI.Gatherv!(sendbuf, MPI.VBuffer(recvbuf, counts), 0, comm)
            λ[ie + 1] = ComplexF64(vpr, vpi)
            Χ[:, ie + 1] = recvbuf
        else
            MPI.Gatherv!(sendbuf, nothing, 0, comm)
        end
        VecRestoreArray(vr, local_ref)
    end
    VecDestroy(vr)
    VecDestroy(vi)
    return λ, Χ
end

"""
    _assemble_results(λ, Χ, B_csr, sigma_0, nconv, solve_time) -> SolverResults

Rank-0 assembly of the final result: drop spurious (singular-B) modes, sort by
distance from the shift, and wrap in `SolverResults`. `converged` is true when at
least one eigenpair was found.
"""
function _assemble_results(λ, Χ, B_csr, sigma_0, nconv, solve_time)
    B = sparse_from_csr(B_csr)                      # rank-0 SparseMatrixCSC for mass filter
    if nconv ≥ 1
        λ, Χ = _filter_physical_modes(λ, Χ, B)
        λ, Χ = sort_eigenvalues!(λ, Χ, :nearest; σ=sigma_0)
    end
    hist = ConvergenceHistory()
    return SolverResults(ComplexF64.(λ), ComplexF64.(Χ), nconv ≥ 1,
                         :Slepc, Float64(sigma_0), Int(nconv), solve_time, hist)
end

# ------------------------------------------------------------------------------
# Public solve_mpi methods, SLEPc init/options, and validation
# ------------------------------------------------------------------------------

"""Throw a clear error if PETSc was built with real scalars (results would be wrong)."""
function _assert_complex_scalars()
    if !(PetscScalar <: Complex)
        error("BiGSTARS solve_mpi requires a complex-scalar PETSc/SLEPc build " *
              "(configure with --with-scalar-type=complex); got PetscScalar=$(PetscScalar).")
    end
end

"""
    solve_mpi(cache, k_values; sigma_0, nev=1, which=:LM, tol=1e-10, maxiter=300,
              ncv=0, mat_solver=:mumps, eps_type=:krylovschur,
              manage_init=true, verbose=false) -> Vector{SolverResults}

Distributed eigensolve for each wavenumber. Collective: every MPI rank must call
it. Rank 0 returns fully populated results; other ranks return empty markers.

SLEPc has no programmatic per-solve options inserter in these wrappers, so the
solver options are placed in the PETSc options database at init time. With
`manage_init=true` (default) `solve_mpi` calls `SlepcInitialize` with the built
options string the first time it runs (finalization is left to SLEPc's atexit
hook). To manage initialization yourself, pass `manage_init=false` and call
`SlepcInitialize(BiGSTARS._eps_options(...))` before `solve_mpi`.
"""
function BiGSTARS.solve_mpi(cache::BiGSTARS.DiscretizationCache,
                            k_values::AbstractVector;
                            sigma_0::Real, nev::Integer=1, which::Symbol=:LM,
                            tol::Real=1e-10, maxiter::Integer=300, ncv::Integer=0,
                            mat_solver::Symbol=:mumps, eps_type::Symbol=:krylovschur,
                            manage_init::Bool=true, verbose::Bool=false)
    opts = _eps_options(; sigma_0=Float64(sigma_0), nev=Int(nev), which=which,
                        tol=Float64(tol), maxiter=Int(maxiter), ncv=Int(ncv),
                        mat_solver=String(mat_solver), eps_type=String(eps_type))

    if manage_init && !_SLEPC_INITED[]
        SlepcInitialize(opts)
        _SLEPC_INITED[] = true
    end
    _assert_complex_scalars()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    results = Vector{SolverResults}(undef, length(k_values))

    for (i, k) in enumerate(k_values)
        # Rank 0 assembles serially with the existing pipeline; others hold nothing.
        A_csr = nothing; B_csr = nothing; N = 0
        if rank == 0
            A, B = BiGSTARS.assemble(cache, Float64(k))
            N = size(A, 1)
            A_csr = _to_csr(A)
            B_csr = _to_csr(B)
        end
        N = MPI.bcast(N, 0, comm)                   # all ranks need the global size

        verbose && rank == 0 &&
            println("solve_mpi: k=$(k)  N=$(N)  σ₀=$(sigma_0)  nev=$(nev)")

        results[i] = _solve_one(A_csr, B_csr, N, comm; sigma_0=Float64(sigma_0))
    end
    return results
end

"""Single-problem overload (no wavenumber sweep)."""
function BiGSTARS.solve_mpi(cache::BiGSTARS.DiscretizationCache; sigma_0::Real, kwargs...)
    return BiGSTARS.solve_mpi(cache, [0.0]; sigma_0=sigma_0, kwargs...)
end

end # module
