module BiGSTARSMPIExt

using BiGSTARS
using BiGSTARS: _to_csr, _csr_row_block, _csr_block_nnz_split, _eps_options,
                _sigma_schedule, sparse_from_csr, SolverResults, ConvergenceHistory,
                sort_eigenvalues!, _filter_physical_modes
using MPI
using PetscWrap
using SlepcWrap
using SparseArrays
using LinearAlgebra: norm
using Printf: @printf

# ==============================================================================
# BiGSTARSMPIExt — distributed eigensolver backend over SLEPc/PETSc, the package's
# sole eigensolve backend (loaded when MPI, PetscWrap, SlepcWrap are all imported).
#
# Rank 0 assembles A,B serially with the BiGSTARS pipeline and converts them to
# 0-based CSR. Each rank's owned row-block is scattered from rank 0 and inserted
# into a distributed PETSc MatMPIAIJ. SLEPc's EPS solves the generalized pencil
# with a shift-and-invert spectral transform; an adaptive-σ loop retargets the
# transform per attempt via EPSSetTarget until the eigenvalue at the shift is
# stable. Eigenvectors are gathered to rank 0, where SolverResults is built.
#
# API target: PetscWrap 0.1.5 / SlepcWrap 0.1.x. The static solver options enter
# the PETSc options database via SlepcInitialize(opts) at init time (no programmatic
# nev/tol/ST setters in these wrappers); only the numeric σ target is set per
# attempt, programmatically, via the wrapped EPSSetTarget. With -st_type sinvert,
# SLEPc uses the EPS target as the shift, so STSetShift (unwrapped) is not needed.
# Runtime numerics (convergence, eigenvector layout) are verified by the mpi.yml
# CI job on a complex PETSc/SLEPc, not locally.
# ==============================================================================

# SLEPc/PETSc may only be initialized once per process. Track init + the options
# string. The static options enter the database at init; the per-attempt σ target
# is set via EPSSetTarget, so the string never changes per solve.
const _SLEPC_INITED = Ref(false)
const _SLEPC_OPTS = Ref{String}("")

"""Throw a clear error if PETSc was built with real scalars (results would be wrong)."""
function _assert_complex_scalars()
    if !(PetscScalar <: Complex)
        error("BiGSTARS.solve requires a complex-scalar PETSc/SLEPc build " *
              "(configure with --with-scalar-type=complex); got PetscScalar=$(PetscScalar).")
    end
end

# ------------------------------------------------------------------------------
# Build the distributed PETSc matrix from scattered CSR row-blocks
# ------------------------------------------------------------------------------

function _build_petsc_mat(A_csr, N::Integer, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    M = MatCreate()                                # defaults to MPI.COMM_WORLD
    MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, PetscInt(N), PetscInt(N))
    MatSetFromOptions(M)
    rstart, rend = MatGetOwnershipRange(M)         # 0-based [rstart, rend)

    starts = MPI.Gather(Int(rstart), 0, comm)
    ends   = MPI.Gather(Int(rend),   0, comm)

    if rank == 0
        rowptr, colind, vals = A_csr
        for p in 1:(nproc - 1)
            ps, pe = starts[p + 1], ends[p + 1]
            lrp, lci, lv = _csr_row_block(rowptr, colind, vals, ps, pe)
            d_nnz, o_nnz = _csr_block_nnz_split(rowptr, colind, ps, pe, ps, pe)
            MPI.send((lrp, lci, lv, d_nnz, o_nnz), p, 0, comm)
        end
        lrp, lci, lv = _csr_row_block(rowptr, colind, vals, rstart, rend)
        d_nnz, o_nnz = _csr_block_nnz_split(rowptr, colind, rstart, rend, rstart, rend)
        _prealloc!(M, nproc, d_nnz, o_nnz) || MatSetUp(M)
        _insert_rows!(M, rstart, lrp, lci, lv)
    else
        (lrp, lci, lv, d_nnz, o_nnz), _ = MPI.recv(0, 0, comm)
        _prealloc!(M, nproc, d_nnz, o_nnz) || MatSetUp(M)
        _insert_rows!(M, rstart, lrp, lci, lv)
    end

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY)
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY)
    return M
end

function _prealloc!(M, nproc::Integer, d_nnz, o_nnz)
    if nproc == 1
        if isdefined(PetscWrap, :MatSeqAIJSetPreallocation)
            PetscWrap.MatSeqAIJSetPreallocation(M, PetscInt(0), PetscInt.(d_nnz))
            return true
        end
    else
        if isdefined(PetscWrap, :MatMPIAIJSetPreallocation)
            PetscWrap.MatMPIAIJSetPreallocation(M, PetscInt(0), PetscInt.(d_nnz),
                                                PetscInt(0), PetscInt.(o_nnz))
            return true
        end
    end
    return false
end

function _insert_rows!(M, rstart::Integer, local_rowptr, local_colind, local_vals)
    nrows = length(local_rowptr) - 1
    for r in 1:nrows
        k0 = local_rowptr[r] + 1
        k1 = local_rowptr[r + 1]
        k1 < k0 && continue                        # empty row
        row  = PetscInt(rstart + r - 1)            # global 0-based row
        cols = PetscInt.(local_colind[k0:k1])      # global 0-based columns
        vs   = PetscScalar.(local_vals[k0:k1])
        MatSetValues(M, PetscInt(1), [row], PetscInt(length(cols)), cols, vs, INSERT_VALUES)
    end
    return M
end

# ------------------------------------------------------------------------------
# Gather eigenpairs to rank 0
# ------------------------------------------------------------------------------

function _gather_eigenpairs(eps, A, nconv::Integer, N::Integer, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    rstart, rend = MatGetOwnershipRange(A)
    nlocal = rend - rstart
    counts = Cint.(MPI.Allgather(Int(nlocal), comm))

    λ = rank == 0 ? Vector{ComplexF64}(undef, nconv) : ComplexF64[]
    Χ = rank == 0 ? Matrix{ComplexF64}(undef, N, nconv) : zeros(ComplexF64, 0, 0)

    vr, vi = MatCreateVecs(A)                      # vectors compatible with A (returns a pair)
    for ie in 0:(nconv - 1)
        # On a complex build PetscScalar==ComplexF64, so vpr is the FULL complex
        # eigenvalue and vpi≈0 — take ComplexF64(vpr), not ComplexF64(vpr, vpi).
        vpr, vpi, _, _ = EPSGetEigenpair(eps, ie, vr, vi)
        local_arr, local_ref = VecGetArray(vr)     # this rank's owned entries (aliases PETSc memory)
        sendbuf = Vector{ComplexF64}(local_arr)
        if rank == 0
            recvbuf = Vector{ComplexF64}(undef, N)
            MPI.Gatherv!(sendbuf, MPI.VBuffer(recvbuf, counts), 0, comm)
            λ[ie + 1] = ComplexF64(vpr)
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

# ------------------------------------------------------------------------------
# Adaptive-σ solve of one pencil (collective across all ranks)
# ------------------------------------------------------------------------------

"""
    _solve_one_adaptive(A_csr, B_csr, N, comm; sigma_0, n_tries, Δσ₀, incre, ϵ, verbose)
        -> SolverResults

Build the distributed pencil once, then sweep the σ schedule. Each attempt
retargets SLEPc via `EPSSetTarget` and re-solves; rank 0 filters spurious modes,
sorts nearest σ, and decides whether to stop (successive |Δλ₁| < ϵ). The stop
flag is BROADCAST so every rank's control flow is identical — required, or the
collective `EPSSolve`/`MatDestroy` calls desync and deadlock.
"""
function _solve_one_adaptive(A_csr, B_csr, N::Integer, comm::MPI.Comm;
                             sigma_0, n_tries, Δσ₀, incre, ϵ, verbose)
    rank = MPI.Comm_rank(comm)
    t0 = MPI.Wtime()
    A = _build_petsc_mat(A_csr, N, comm)
    B = _build_petsc_mat(B_csr, N, comm)
    schedule = _sigma_schedule(sigma_0, n_tries, Δσ₀, incre)

    hist = ConvergenceHistory()
    λ_prev = nothing
    best_λ = ComplexF64[]
    best_Χ = rank == 0 ? zeros(ComplexF64, N, 0) : zeros(ComplexF64, 0, 0)
    final_shift = Float64(sigma_0)

    for σ in schedule
        eps = EPSCreate()
        EPSSetOperators(eps, A, B)
        EPSSetTarget(eps, PetscScalar(σ))
        EPSSetFromOptions(eps)
        EPSSolve(eps)
        nconv = EPSGetConverged(eps)
        λ, Χ = _gather_eigenpairs(eps, A, nconv, N, comm)
        EPSDestroy(eps)

        stop = false
        if rank == 0
            push!(hist.attempts, σ)
            if nconv ≥ 1
                λf, Χf = _filter_physical_modes(λ, Χ, sparse_from_csr(B_csr))
                λf, Χf = sort_eigenvalues!(λf, Χf, :nearest; σ=σ)
                push!(hist.converged, true)
                push!(hist.eigenvalues, λf[1])
                push!(hist.errors, "")
                best_λ = ComplexF64.(λf)
                best_Χ = ComplexF64.(Χf)
                final_shift = σ
                if λ_prev !== nothing && abs(λf[1] - λ_prev) < ϵ
                    stop = true
                end
                λ_prev = λf[1]
                verbose && @printf("  ✓ σ=%.6f  λ₁=%.6f%+.6fi\n", σ, real(λf[1]), imag(λf[1]))
            else
                push!(hist.converged, false)
                push!(hist.eigenvalues, NaN + 0im)
                push!(hist.errors, "no convergence")
                verbose && @printf("  ✗ σ=%.6f  no convergence\n", σ)
            end
        end
        stop = MPI.bcast(stop, 0, comm)   # collective: identical control flow on every rank
        stop && break
    end

    solve_time = MPI.Wtime() - t0
    MatDestroy(A)
    MatDestroy(B)

    if rank == 0
        hist.final_shift = final_shift
        converged = !isempty(best_λ)
        return SolverResults(best_λ, best_Χ, converged, :Slepc, final_shift,
                             length(hist.attempts), solve_time, hist)
    else
        return SolverResults(ComplexF64[], zeros(ComplexF64, 0, 0), false, :Slepc,
                             final_shift, 0, solve_time, ConvergenceHistory())
    end
end

# ------------------------------------------------------------------------------
# Public solve methods + SLEPc init
# ------------------------------------------------------------------------------

function BiGSTARS.solve(cache::BiGSTARS.DiscretizationCache,
                        k_values::AbstractVector;
                        sigma_0::Real, nev::Integer=1, which::Symbol=:LM,
                        tol::Real=1e-10, maxiter::Integer=300, ncv::Integer=0,
                        mat_solver::Symbol=:mumps, eps_type::Symbol=:krylovschur,
                        n_tries::Integer=8, Δσ₀::Real=0.2, incre::Real=1.2, ϵ::Real=1e-5,
                        manage_init::Bool=true, verbose::Bool=false)
    opts = _eps_options(; nev=Int(nev), which=which, tol=Float64(tol),
                        maxiter=Int(maxiter), ncv=Int(ncv),
                        mat_solver=String(mat_solver), eps_type=String(eps_type))

    # MPI.jl populates COMM_WORLD only through its own MPI.Init(); the C-level
    # MPI_Init PETSc runs inside SlepcInitialize does NOT. Init MPI.jl first, then
    # let PETSc reuse the already-initialized MPI.
    MPI.Initialized() || MPI.Init()
    if manage_init
        if !_SLEPC_INITED[]
            SlepcInitialize(opts)
            _SLEPC_INITED[] = true
            _SLEPC_OPTS[] = opts
        elseif opts != _SLEPC_OPTS[]
            error("solve: SLEPc is already initialized in this process with " *
                  "different options, and PETSc/SLEPc options can only be set once " *
                  "per process. Restart Julia for new solver settings, or pass " *
                  "manage_init=false and drive SlepcInitialize yourself.\n" *
                  "  initialized with: $(_SLEPC_OPTS[])\n  requested now:    $(opts)")
        end
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
            println("solve: k=$(k)  N=$(N)  σ₀=$(sigma_0)  nev=$(nev)")
        results[i] = _solve_one_adaptive(A_csr, B_csr, N, comm;
                        sigma_0=Float64(sigma_0), n_tries=Int(n_tries),
                        Δσ₀=Float64(Δσ₀), incre=Float64(incre), ϵ=Float64(ϵ),
                        verbose=verbose)
    end
    return results
end

"""Single-problem overload (no wavenumber sweep)."""
function BiGSTARS.solve(cache::BiGSTARS.DiscretizationCache; sigma_0::Real, kwargs...)
    return BiGSTARS.solve(cache, [0.0]; sigma_0=sigma_0, kwargs...)
end

end # module
