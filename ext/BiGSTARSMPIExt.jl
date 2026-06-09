module BiGSTARSMPIExt

using BiGSTARS
using BiGSTARS: _to_csr, _csr_block_nnz_split, _eps_options,
                _sigma_schedule, _group_indices, _petsc_ownership, assemble_rows,
                SolverResults, ConvergenceHistory, _keep_by_mass,
                sort_eigenvalues!, _discretize_n_total, _union_block_nnz, _union_template
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
# Each rank builds ONLY its owned rows of A,B directly (discretize with row_range —
# no full operator is materialized on any rank, no rank-0 build, no scatter) and inserts them
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

"""Fill an already-created PETSc matrix `M` from a local owned-row slice `rows`
(`nrows × N`, global columns), owned global rows `[rstart,rend)`. No scatter."""
function _fill_mat!(M, rows, rstart::Integer, rend::Integer, comm::MPI.Comm)
    rowptr, colind, vals = _to_csr(rows)                       # CSR of the local slice
    d_nnz, o_nnz = _csr_block_nnz_split(rowptr, colind, 0, rend - rstart, rstart, rend)
    _prealloc!(M, MPI.Comm_size(comm), d_nnz, o_nnz) || MatSetUp(M)
    _insert_rows!(M, rstart, rowptr, colind, vals)            # global rows rstart..rend-1
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY)
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY)
    return M
end

"""Refill an existing PETSc matrix `M`'s values in place for a new wavenumber by INSERT-ing
the owned-row slice `rows`. `rows` MUST carry the full UNION structure (the caller adds a
`_union_template`, so absent-this-wavenumber entries are explicit zeros) — then INSERT_VALUES
overwrites EVERY preallocated slot, leaving no stale values from a previous wavenumber. This
avoids `MatZeroEntries` (unwrapped in PetscWrap 0.1.5). `M` must already be preallocated for
the union pattern (see `_create_union_mat`); no reallocation occurs."""
function _refill_mat!(M, rows, rstart::Integer, rend::Integer)
    rowptr, colind, vals = _to_csr(rows)
    _insert_rows!(M, rstart, rowptr, colind, vals)
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY)
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY)
    return M
end

"""Create a distributed PETSc matrix preallocated ONCE for the union of all `components`'
k-power patterns (constant across the sweep), so values can be refilled per wavenumber with
`_refill_mat!` without growing rows. Returns `(M, rstart, rend)` (owned 0-based row range)."""
function _create_union_mat(components, N::Integer, comm::MPI.Comm)
    M = MatCreate(comm)
    MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, PetscInt(N), PetscInt(N))
    MatSetFromOptions(M)
    rstart, rend = MatGetOwnershipRange(M)
    d_nnz, o_nnz = _union_block_nnz(components, rstart, rend, N)
    _prealloc!(M, MPI.Comm_size(comm), d_nnz, o_nnz) || MatSetUp(M)
    return M, rstart, rend
end

"""Build distributed PETSc A and B for one wavenumber, each rank assembling only
its owned rows locally from its row-restricted cache (no rank-0 full matrix, no scatter)."""
function _build_petsc_mats_local(cache, k, N::Integer, comm::MPI.Comm)
    A = MatCreate(comm)
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, PetscInt(N), PetscInt(N))
    MatSetFromOptions(A)
    rstart, rend = MatGetOwnershipRange(A)                    # 0-based [rstart,rend)
    A_rows, B_rows = assemble_rows(cache, Float64(k), rstart, rend)
    _fill_mat!(A, A_rows, rstart, rend, comm)

    B = MatCreate(comm)
    MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, PetscInt(N), PetscInt(N))
    MatSetFromOptions(B)
    rstartB, rendB = MatGetOwnershipRange(B)
    (rstartB, rendB) == (rstart, rend) ||                    # same N+comm ⇒ same layout
        error("PETSc returned different row ownership for A $((rstart, rend)) and " *
              "B $((rstartB, rendB)) with identical size/comm — cannot assemble the pencil.")
    _fill_mat!(B, B_rows, rstart, rend, comm)
    return A, B
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

function _gather_eigenpairs(eps, A, B, nconv::Integer, N::Integer, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    rstart, rend = MatGetOwnershipRange(A)
    nlocal = rend - rstart
    counts = Cint.(MPI.Allgather(Int(nlocal), comm))

    λ = rank == 0 ? Vector{ComplexF64}(undef, nconv) : ComplexF64[]
    Χ = rank == 0 ? Matrix{ComplexF64}(undef, N, nconv) : zeros(ComplexF64, 0, 0)
    masses = Vector{Float64}(undef, nconv)         # replicated on every rank (via Allreduce)

    vr, vi = MatCreateVecs(A)
    Bvr = VecDuplicate(vr)
    for ie in 0:(nconv - 1)
        vpr, vpi, _, _ = EPSGetEigenpair(eps, ie, vr, vi)
        local_arr, local_ref = VecGetArray(vr)     # this rank's owned entries
        sendbuf = Vector{ComplexF64}(local_arr)
        nv2 = sum(abs2, local_arr)                 # ‖vr‖² (local part)
        VecRestoreArray(vr, local_ref)             # restore before MatMult uses vr

        MatMult(B, vr, Bvr)                         # distributed B·vr
        bl, bref = VecGetArray(Bvr)
        nb2 = sum(abs2, bl)                         # ‖B vr‖² (local part)
        VecRestoreArray(Bvr, bref)

        nv2g = MPI.Allreduce(nv2, +, comm)         # collective (every rank)
        nb2g = MPI.Allreduce(nb2, +, comm)
        masses[ie + 1] = sqrt(nb2g) / sqrt(max(nv2g, Base.eps(Float64)))

        if rank == 0
            recvbuf = Vector{ComplexF64}(undef, N)
            MPI.Gatherv!(sendbuf, MPI.VBuffer(recvbuf, counts), 0, comm)
            λ[ie + 1] = ComplexF64(vpr)
            Χ[:, ie + 1] = recvbuf
        else
            MPI.Gatherv!(sendbuf, nothing, 0, comm)
        end
    end
    VecDestroy(Bvr)
    VecDestroy(vr)
    VecDestroy(vi)
    return λ, Χ, masses
end

# ------------------------------------------------------------------------------
# Adaptive-σ solve of one pencil (collective across all ranks)
# ------------------------------------------------------------------------------

"""
    _run_sigma_schedule(eps, A, B, k, N, comm; sigma_0, n_tries, Δσ₀, incre, ϵ, verbose)
        -> SolverResults

Sweep the adaptive-σ schedule on a CALLER-OWNED, already-configured `eps` (operators set,
`EPSSetFromOptions` done) and assembled `A`,`B`. Reuses `eps` across σ attempts via
`EPSSetTarget`. Does NOT create/destroy `eps`/`A`/`B`. The stop flag is broadcast so every
rank's control flow is identical (else collective `EPSSolve` desyncs/deadlocks).
"""
function _run_sigma_schedule(eps, A, B, k, N::Integer, comm::MPI.Comm;
                             sigma_0, n_tries, Δσ₀, incre, ϵ, verbose)
    rank = MPI.Comm_rank(comm)
    t0 = MPI.Wtime()
    schedule = _sigma_schedule(sigma_0, n_tries, Δσ₀, incre)
    hist = ConvergenceHistory()
    λ_prev = nothing
    best_λ = ComplexF64[]
    best_Χ = rank == 0 ? zeros(ComplexF64, N, 0) : zeros(ComplexF64, 0, 0)
    final_shift = Float64(sigma_0)

    for σ in schedule
        EPSSetTarget(eps, PetscScalar(σ))
        EPSSolve(eps)
        nconv = EPSGetConverged(eps)
        λ, Χ, masses = _gather_eigenpairs(eps, A, B, nconv, N, comm)

        stop = false
        if rank == 0
            push!(hist.attempts, σ)
            if nconv ≥ 1
                keep = _keep_by_mass(masses)
                λf, Χf = sort_eigenvalues!(λ[keep], Χ[:, keep], :nearest; σ=σ)
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
        stop = MPI.bcast(stop, 0, comm)
        stop && break
    end

    solve_time = MPI.Wtime() - t0
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

"""Non-reuse path (default): build per-k `A`,`B`, create a fresh `eps`, run the σ schedule,
tear everything down. Behavior matches the prior per-k solve."""
function _solve_one_adaptive(cache, k, N::Integer, comm::MPI.Comm;
                             sigma_0, n_tries, Δσ₀, incre, ϵ, verbose)
    A, B = _build_petsc_mats_local(cache, k, N, comm)
    eps = EPSCreate(comm)
    EPSSetOperators(eps, A, B)
    EPSSetFromOptions(eps)
    r = _run_sigma_schedule(eps, A, B, k, N, comm;
                            sigma_0=sigma_0, n_tries=n_tries, Δσ₀=Δσ₀, incre=incre, ϵ=ϵ,
                            verbose=verbose)
    EPSDestroy(eps)
    MatDestroy(A)
    MatDestroy(B)
    return r
end

"""Reuse path: preallocate `A`,`B` ONCE for the union pattern and create one `eps`, then per
wavenumber in `ks` refill values in place and run the σ schedule on the same objects. With
`-st_pc_factor_reuse_ordering` MUMPS reuses the LU ordering across wavenumbers. Returns one
`SolverResults` per entry of `ks` (in order). Requires `isempty(cache.derived_caches)` —
enforced by the `reuse_factorization` guard in `solve`."""
function _solve_group_reused(cache, ks, N::Integer, comm::MPI.Comm;
                             sigma_0, n_tries, Δσ₀, incre, ϵ, verbose)
    A, rstart, rend = _create_union_mat(cache.A_components, N, comm)
    B, rstartB, rendB = _create_union_mat(cache.B_components, N, comm)
    (rstartB, rendB) == (rstart, rend) ||
        error("PETSc returned different row ownership for A $((rstart, rend)) and " *
              "B $((rstartB, rendB)) — cannot assemble the pencil.")
    # Zero-valued union templates (built once): added to each wavenumber's slice so the
    # INSERT_VALUES refill overwrites every preallocated slot with explicit zeros where this
    # wavenumber is absent — no stale values, no MatZeroEntries (unwrapped in PetscWrap 0.1.5).
    tmplA = _union_template(cache.A_components, rstart, rend, N)
    tmplB = _union_template(cache.B_components, rstart, rend, N)
    eps = EPSCreate(comm)
    EPSSetOperators(eps, A, B)
    EPSSetFromOptions(eps)

    out = SolverResults[]
    for k in ks
        A_rows, B_rows = assemble_rows(cache, Float64(k), rstart, rend)
        _refill_mat!(A, A_rows + tmplA, rstart, rend)
        _refill_mat!(B, B_rows + tmplB, rstart, rend)
        EPSSetOperators(eps, A, B)            # re-flag updated values (numeric refactor; ordering reused)
        push!(out, _run_sigma_schedule(eps, A, B, Float64(k), N, comm;
                                       sigma_0=sigma_0, n_tries=n_tries, Δσ₀=Δσ₀,
                                       incre=incre, ϵ=ϵ, verbose=verbose))
    end

    EPSDestroy(eps)
    MatDestroy(A)
    MatDestroy(B)
    return out
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
                        ngroups::Integer=1, manage_init::Bool=true,
                        reuse_factorization::Bool=false, verbose::Bool=false)
    opts = _eps_options(; nev=Int(nev), which=which, tol=Float64(tol),
                        maxiter=Int(maxiter), ncv=Int(ncv),
                        mat_solver=String(mat_solver), eps_type=String(eps_type),
                        reuse_factorization=reuse_factorization)

    # MPI.jl populates COMM_WORLD only through its own MPI.Init(); the C-level
    # MPI_Init PETSc runs inside SlepcInitialize does NOT. Init MPI.jl first, then
    # let PETSc reuse the already-initialized MPI.
    MPI.Initialized() || MPI.Init()
    if manage_init
        if !_SLEPC_INITED[]
            # SlepcInitialize is *documented* to call PetscInitialize, but with some
            # SLEPc/SlepcWrap builds PETSc is left uninitialized (PetscInitialized()
            # == false), so the first MatCreate fails with PETSc error 98 (wrong state).
            # Initialize PETSc explicitly first (idempotent — SlepcInitialize skips
            # re-init when PETSc is already up), then SLEPc.
            PetscInitialize(opts)
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

    if reuse_factorization && !isempty(cache.derived_caches)
        error("solve: reuse_factorization=true is not supported for caches with derived-variable " *
              "terms (non-empty derived_caches, i.e. discretize(...; augment_derived=false) with " *
              "@derive vars) — their dense H(k) block is not in the union preallocation. " *
              "Use augment_derived=true (the default) or reuse_factorization=false.")
    end

    world = MPI.COMM_WORLD
    P = MPI.Comm_size(world)
    wrank = MPI.Comm_rank(world)
    # Validate identically on every rank (pre-split, so a bad ngroups errors
    # collectively — no deadlock).
    (1 ≤ ngroups ≤ P) ||
        error("solve: ngroups=$(ngroups) must be in 1:$(P) (number of MPI ranks)")
    (P % ngroups == 0) ||
        error("solve: nprocs=$(P) is not divisible by ngroups=$(ngroups)")

    nk = length(k_values)
    # Full-length result vector; non-(global-root) ranks return empty markers.
    results = SolverResults[
        SolverResults(ComplexF64[], zeros(ComplexF64, 0, 0), false, :Slepc,
                      Float64(sigma_0), 0, 0.0, ConvergenceHistory()) for _ in 1:nk]

    # Form groups. ngroups==1 keeps COMM_WORLD verbatim (the v4.0.0 path).
    if ngroups == 1
        group_comm = world; group_id = 0; psize = P
    else
        psize = P ÷ ngroups
        group_id = wrank ÷ psize
        group_comm = MPI.Comm_split(world, group_id, wrank)
    end
    grank = MPI.Comm_rank(group_comm)

    # Each group solves its round-robin subset, sequentially, distributed on group_comm.
    local_pairs = Tuple{Int,SolverResults}[]
    idxs = _group_indices(nk, ngroups, group_id)
    N = cache.N_total                              # replicated; no assemble-on-root, no bcast
    if reuse_factorization
        verbose && grank == 0 &&
            println("solve: group $(group_id)  reuse_factorization  $(length(idxs)) k-values  N=$(N)")
        rs = _solve_group_reused(cache, [Float64(k_values[i]) for i in idxs], N, group_comm;
                        sigma_0=Float64(sigma_0), n_tries=Int(n_tries),
                        Δσ₀=Float64(Δσ₀), incre=Float64(incre), ϵ=Float64(ϵ),
                        verbose=verbose)
        if grank == 0
            for (j, i) in enumerate(idxs)
                push!(local_pairs, (i, rs[j]))
            end
        end
    else
        for i in idxs
            verbose && grank == 0 &&
                println("solve: group $(group_id)  k=$(k_values[i])  N=$(N)  σ₀=$(sigma_0)  nev=$(nev)")
            r = _solve_one_adaptive(cache, Float64(k_values[i]), N, group_comm;
                            sigma_0=Float64(sigma_0), n_tries=Int(n_tries),
                            Δσ₀=Float64(Δσ₀), incre=Float64(incre), ϵ=Float64(ϵ),
                            verbose=verbose)
            grank == 0 && push!(local_pairs, (i, r))
        end
    end

    # Collect group-root results to global rank 0 (point-to-point over WORLD).
    if ngroups == 1
        wrank == 0 && for (i, r) in local_pairs
            results[i] = r
        end
    else
        TAG = 7777
        if wrank == 0
            for (i, r) in local_pairs                 # own group's results
                results[i] = r
            end
            for g in 1:(ngroups - 1)                   # the other groups' roots
                pairs, _ = MPI.recv(g * psize, TAG, world)
                for (i, r) in pairs
                    results[i] = r
                end
            end
        elseif grank == 0
            MPI.send(local_pairs, 0, TAG, world)       # group root → global root
        end
    end

    return results
end

"""Single-problem overload (no wavenumber sweep)."""
function BiGSTARS.solve(cache::BiGSTARS.DiscretizationCache; sigma_0::Real, kwargs...)
    return BiGSTARS.solve(cache, [0.0]; sigma_0=sigma_0, kwargs...)
end

function BiGSTARS.discretize_distributed(prob; ngroups::Integer=1,
                                         augment_derived::Bool=true, kwargs...)
    MPI.Initialized() || MPI.Init()
    world = MPI.COMM_WORLD
    P = MPI.Comm_size(world); wrank = MPI.Comm_rank(world)
    (1 ≤ ngroups ≤ P) || error("discretize_distributed: ngroups=$(ngroups) must be in 1:$(P)")
    (P % ngroups == 0) || error("discretize_distributed: nprocs=$(P) not divisible by ngroups=$(ngroups)")

    psize = P ÷ ngroups
    grank = wrank % psize                                 # rank within its group (deterministic)

    # Owned rows via PETSc's deterministic PETSC_DECIDE split (pure formula). N_total is the
    # post-augmentation size, computed WITHOUT a build so the split is known before discretize
    # runs. solve's per-group build Mat uses the same split; assemble_rows asserts the resulting
    # range matches the cache's row_range.
    N_total = _discretize_n_total(prob; augment_derived=augment_derived)
    rstart, rend = _petsc_ownership(N_total, psize, grank)

    # Distributed assembly: each rank builds ONLY its owned rows — no full operator is ever
    # materialized (peak build memory and build time drop ~1/psize). The returned cache already
    # has row_range set, so it plugs straight into assemble_rows.
    return BiGSTARS.discretize(prob; augment_derived=augment_derived,
                               row_range=(rstart, rend), kwargs...)
end

end # module
