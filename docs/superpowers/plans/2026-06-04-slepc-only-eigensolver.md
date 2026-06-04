# SLEPc-only eigensolver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `KrylovKit`/`ArnoldiMethod`/`Arpack` and make SLEPc-over-PETSc the only eigensolver, exposed through a single `solve` entrypoint with the adaptive-σ retry ported onto it.

**Architecture:** The old distributed `solve_mpi` (PETSc `MatMPIAIJ` + SLEPc EPS, shift-and-invert) becomes the package's only `solve`. Rank 0 assembles each wavenumber's pencil serially, scatters CSR row-blocks, and an adaptive-σ loop retargets SLEPc via `EPSSetTarget` per attempt with the stop decision computed on rank 0 and broadcast (collective-safe). `MPI`/`PetscWrap`/`SlepcWrap` become hard deps; the serial machinery is deleted.

**Tech Stack:** Julia 1.10+, MPI.jl, PetscWrap, SlepcWrap, complex-scalar PETSc/SLEPc (system build for solves; PETSc_jll real-scalar suffices to load + run non-solver tests cross-platform).

---

## Verification reality (read first)

- **Non-solver code** (helpers, package load, DSL/assembly tests) is verifiable locally and in the cross-platform `CI.yml` matrix on the real-scalar PETSc_jll.
- **The actual SLEPc solve** needs a complex-scalar PETSc/SLEPc + bound MPI. It is **only** verifiable in the `mpi.yml` Linux job (or on a cluster). Steps that touch the solve body mark expected verification as **CI (mpi.yml)**. Getting that job green is the real deliverable — see Task 8.
- Run Julia for this repo through the project's known invocation (juliaup binary + `JULIA_DEPOT_PATH` overlay, unsandboxed); CI is the source of truth for solver behavior.

## Critical gotcha: `PetscWrap` exports `solve`

Our entrypoint is now literally `solve`. `PetscWrap` also exports `solve`, so a blanket `using PetscWrap` collides with our definition. **Always use selective imports** (`using PetscWrap: <names>` excluding `solve`) in the package, and `import PetscWrap, SlepcWrap` (not `using`) in scripts/tests. This is baked into the tasks below.

## File structure (after this plan)

```
src/
  BiGSTARS.jl          modified: drop 3 serial libs; new includes/exports; trimmed workload
  results.jl           NEW (was eig_solver.jl, gutted): SolverResults, ConvergenceHistory,
                       sort_eigenvalues!, _filter_physical_modes, print_summary(::SolverResults)
  slepc_solve.jl       NEW (was ext/BiGSTARSMPIExt.jl): solve(...) with adaptive-σ
  mpi_prep.jl          modified: + _sigma_schedule; _eps_options drops the numeric target
  discretize.jl        modified: + _assembled_density (moved from solve.jl)
  eig_solver.jl        DELETED
  solve.jl             DELETED
  construct_linear_map.jl  DELETED
ext/BiGSTARSMPIExt.jl  DELETED
test/
  runtests.jl          modified: drop nothing structurally; trimmed files run
  test_eig_solver.jl   modified: keep only kept-utility tests
  test_integration.jl  modified: drop serial-solve testsets; rest stays (uses eigvals)
  test_coverage_gaps.jl modified: drop EigenSolver/compare_methods!/serial-solve testsets
  test_mpi_prep.jl     modified: + _sigma_schedule + _eps_options tests
  mpi/test_slepc.jl    modified: analytic reference (no serial), nev, adaptive, spurious
Project.toml           modified: deps flip; drop [weakdeps]/[extensions]
.github/workflows/mpi.yml  modified: remove continue-on-error (blocking)
examples/*.jl          modified: solve_mpi/serial → solve + MPI boilerplate
docs/src/*.md          modified: solve_mpi→solve; drop experimental + multi-backend docs
```

---

## Task 1: Pure helpers — σ schedule + drop numeric target from options

**Files:**
- Modify: `src/mpi_prep.jl`
- Test: `test/test_mpi_prep.jl`

- [ ] **Step 1: Write failing tests**

Append inside the existing top-level `@testset` in `test/test_mpi_prep.jl`:

```julia
@testset "_sigma_schedule" begin
    s = BiGSTARS._sigma_schedule(1.0, 3, 0.2, 1.2)
    @test s[1] == 1.0
    @test length(s) == 1 + 2 * 3
    @test s[2] ≈ 1.0 + 0.2 * 1.0       # first up step
    @test s[3] ≈ 1.0 + 0.2 * 1.2       # second up step (×incre)
    @test s[5] ≈ 1.0 - 0.2 * 1.0       # first down step
    # schedule scales with |σ₀|
    s2 = BiGSTARS._sigma_schedule(-2.0, 1, 0.5, 1.0)
    @test s2 == [-2.0, -2.0 + 0.5 * 2.0, -2.0 - 0.5 * 2.0]
end

@testset "_eps_options drops the numeric target" begin
    o = BiGSTARS._eps_options(; nev=4, which=:LM, tol=1e-9, maxiter=200,
                              ncv=0, mat_solver="mumps", eps_type="krylovschur")
    @test occursin("-eps_nev 4", o)
    @test occursin("-st_type sinvert", o)
    @test occursin("-eps_target_magnitude", o)   # which flag stays
    @test !occursin("-eps_target ", o)           # numeric target removed (set via EPSSetTarget)
    @test occursin("-st_pc_factor_mat_solver_type mumps", o)
end
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'` (or run just `test/test_mpi_prep.jl` through the harness)
Expected: FAIL — `_sigma_schedule` undefined; `_eps_options` still emits `-eps_target ` and still requires `sigma_0`.

- [ ] **Step 3: Add `_sigma_schedule` to `src/mpi_prep.jl`**

Add after the `_csr_block_nnz_split` function:

```julia
"""
    _sigma_schedule(σ₀, n_tries, Δσ₀, incre) -> Vector{Float64}

Adaptive shift schedule: `σ₀` first, then `n_tries` geometrically growing
increments above and below it (`Δσ₀ * incre^(i-1) * |σ₀|`). Pure-Julia, so the
adaptive-σ logic is unit-tested without PETSc. Mirrors the schedule the old
serial solver used.
"""
function _sigma_schedule(σ₀::Real, n_tries::Integer, Δσ₀::Real, incre::Real)
    up = Float64[Δσ₀ * incre^(i - 1) * abs(σ₀) for i in 1:n_tries]
    return Float64[Float64(σ₀); (σ₀ .+ up)...; (σ₀ .- up)...]
end
```

- [ ] **Step 4: Drop the numeric target from `_eps_options` in `src/mpi_prep.jl`**

Replace the existing `_eps_options` signature line and body. Change the signature
to drop `sigma_0`, and delete the `-eps_target $(sigma_0) ` segment:

```julia
function _eps_options(; nev, which, tol, maxiter, ncv, mat_solver, eps_type)
    haskey(_WHICH_OPT, which) || throw(ArgumentError("unsupported which=$which"))
    opts = "-eps_type $(eps_type) " *
           "-eps_gen_non_hermitian " *
           "-eps_nev $(nev) " *
           "-eps_tol $(tol) " *
           "-eps_max_it $(maxiter) " *
           "-eps_$(_WHICH_OPT[which]) " *
           "-st_type sinvert " *
           "-st_pc_type lu " *
           "-st_pc_factor_mat_solver_type $(mat_solver) "
    ncv > 0 && (opts *= "-eps_ncv $(ncv) ")
    return opts
end
```

Also update the docstring's first line to drop the `sigma_0` mention (the target
is now set per attempt via `EPSSetTarget`).

- [ ] **Step 5: Run tests, verify pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS for the two new testsets (the rest of `test_mpi_prep.jl` still passes).

- [ ] **Step 6: Commit**

```bash
git add src/mpi_prep.jl test/test_mpi_prep.jl
git commit -m "feat: add adaptive-σ schedule helper; drop numeric target from SLEPc options"
```

Note: the still-present `ext/BiGSTARSMPIExt.jl` calls `_eps_options(; sigma_0=…)`
with the old signature — that extension only compiles in the `test/mpi`
environment and is removed in Task 2 of this same PR. The cross-platform suite
(which never loads the extension) stays green.

---

## Task 2: Structural overhaul — deps flip, move code into core, delete serial machinery

This task is atomic: removing the serial libraries breaks `eig_solver.jl`, so the
deletions, the dependency flip, and the new SLEPc core must land together. Verify
at the end that the package loads.

**Files:**
- Modify: `Project.toml`, `src/BiGSTARS.jl`, `src/discretize.jl`
- Create: `src/results.jl`, `src/slepc_solve.jl`
- Delete: `src/eig_solver.jl`, `src/solve.jl`, `src/construct_linear_map.jl`, `ext/BiGSTARSMPIExt.jl`

- [ ] **Step 1: Flip `Project.toml` dependencies**

Set `[deps]` to exactly (UUIDs preserved from the current file/weakdeps):

```toml
[deps]
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
PetscWrap = "5be22e1c-01b5-4697-96eb-ef9ccdc854b8"
PrecompileTools = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
SlepcWrap = "c3679e3b-785e-4ccc-b734-b7685cbb935e"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
```

Delete the entire `[weakdeps]` and `[extensions]` blocks. In `[compat]`, delete
the `ArnoldiMethod`, `Arpack`, `KrylovKit`, `LinearMaps`, `VectorInterface` lines
(keep `MPI`, `PetscWrap`, `SlepcWrap`, `FFTW`, `LinearAlgebra`, `PrecompileTools`,
`Printf`, `SparseArrays`, `julia`). Leave `[extras]` and `[targets]` unchanged.

- [ ] **Step 2: Create `src/results.jl` (kept result types/helpers)**

```julia
# ==============================================================================
# results.jl — eigenproblem result types + spectral post-processing shared by the
# SLEPc solve path and by reconstruction. (Was eig_solver.jl; serial machinery
# removed when SLEPc became the sole backend.)
# ==============================================================================

"""
    ConvergenceHistory

Tracks adaptive-shift convergence information during an eigenvalue solve.
"""
mutable struct ConvergenceHistory
    attempts::Vector{Float64}
    converged::Vector{Bool}
    eigenvalues::Vector{ComplexF64}
    errors::Vector{String}
    final_shift::Float64
    total_iterations::Int

    ConvergenceHistory() = new(Float64[], Bool[], ComplexF64[], String[], 0.0, 0)
end

"""
    SolverResults

Computed eigenpairs plus metadata for one eigenproblem.
"""
struct SolverResults
    eigenvalues::Vector{ComplexF64}
    eigenvectors::Matrix{ComplexF64}
    converged::Bool
    method_used::Symbol
    final_shift::Float64
    iterations::Int
    solve_time::Float64
    history::ConvergenceHistory
end

"""
    sort_eigenvalues!(λ, Χ, by; rev=true, σ=nothing) -> (λ, Χ)

Return reordered copies. `:nearest` sorts by ascending `|λ - σ|` (requires `σ`),
the mode shift-and-invert targets; `:R`/`:I`/`:M` sort by real/imag/magnitude.
"""
function sort_eigenvalues!(λ::Vector, Χ::Matrix, by::Symbol; rev::Bool=true,
                           σ::Union{Real,Nothing}=nothing)
    if by === :nearest
        σ === nothing && throw(ArgumentError("sortby=:nearest requires the shift σ"))
        idx = sortperm(λ, by = x -> abs(x - σ))
    else
        sortfun = by == :R ? real : by == :I ? imag : abs
        idx = sortperm(λ, by=sortfun, rev=rev)
    end
    return λ[idx], Χ[:, idx]
end

"""
    _filter_physical_modes(λ, Χ, B; rtol=1e-6) -> (λ, Χ)

Drop infinite/spurious modes of `A x = λ B x` with a singular `B` (descriptor /
augmented systems): physical modes have `‖Bχ‖` of `O(1)`, spurious ones ≈ 0.
No-op for non-singular `B`; never returns an empty set (falls back to inputs).
"""
function _filter_physical_modes(λ::AbstractVector, Χ::AbstractMatrix, B; rtol::Float64=1e-6)
    (isempty(λ) || size(Χ, 2) == 0) && return λ, Χ
    masses = Vector{Float64}(undef, size(Χ, 2))
    @inbounds for i in 1:size(Χ, 2)
        χ = view(Χ, :, i)
        masses[i] = norm(B * χ) / max(norm(χ), eps())
    end
    scale = maximum(masses)
    scale == 0.0 && return λ, Χ
    keep = findall(>(rtol * scale), masses)
    isempty(keep) && return λ, Χ
    return λ[keep], Χ[:, keep]
end

"""
    print_summary(r::SolverResults)

Print a compact summary of one solve result.
"""
function print_summary(r::SolverResults)
    println("EigenSolver Results Summary")
    println("   Method: $(r.method_used)")
    println("   Converged: $(r.converged ? "✅ Yes" : "❌ No")")
    println("   Final shift: $(r.final_shift)")
    @printf("   Total time: %.3fs\n", r.solve_time)
    println("   Attempts: $(length(r.history.attempts))")
    if r.converged && !isempty(r.eigenvalues)
        for (i, λ) in enumerate(r.eigenvalues)
            @printf("   [%d] % .6f %+.6fi\n", i, real(λ), imag(λ))
        end
    end
    return nothing
end
```

- [ ] **Step 3: Move `_assembled_density` into `src/discretize.jl`**

Append to `src/discretize.jl` (it operates on a `DiscretizationCache`, defined
there). It is retained as an assembly-banding diagnostic used by tests; the SLEPc
path does not need it.

```julia
"""
    _assembled_density(cache) -> Float64

Estimate the fill fraction (nnz / N²) of the assembled operator from the cached
k-components — an upper bound on per-wavenumber fill. Any derived cache forces the
dense estimate (`1.0`). Retained as a banding diagnostic for tests.
"""
function _assembled_density(cache::DiscretizationCache)
    isempty(cache.derived_caches) || return 1.0
    N = cache.N_total
    N == 0 && return 1.0
    pat = spzeros(ComplexF64, N, N)
    for (_, M) in cache.A_kcomponents
        pat += M
    end
    return nnz(pat) / (N * N)
end
```

- [ ] **Step 4: Create `src/slepc_solve.jl` (the new `solve` with adaptive-σ)**

```julia
# ==============================================================================
# slepc_solve.jl — distributed eigensolver over SLEPc/PETSc, the package's sole
# eigensolve backend. Rank 0 assembles each wavenumber's pencil serially, scatters
# 0-based CSR row-blocks, and an adaptive-σ loop retargets SLEPc per attempt.
#
# PetscWrap exports `solve` (a KSP solve) which would shadow BiGSTARS.solve — so
# import PetscWrap/SlepcWrap names SELECTIVELY (never blanket `using`).
# ==============================================================================

using MPI
import PetscWrap
import SlepcWrap
using PetscWrap: MatCreate, MatSetSizes, MatSetFromOptions, MatGetOwnershipRange,
                 MatSetValues, MatAssemblyBegin, MatAssemblyEnd, MatCreateVecs,
                 MatDestroy, MatSetUp, VecGetArray, VecRestoreArray, VecDestroy,
                 PetscInt, PetscScalar, PETSC_DECIDE, MAT_FINAL_ASSEMBLY, INSERT_VALUES
using SlepcWrap: SlepcInitialize, EPSCreate, EPSSetOperators, EPSSetFromOptions,
                 EPSSolve, EPSGetConverged, EPSGetEigenpair, EPSDestroy, EPSSetTarget
using SparseArrays
using LinearAlgebra: norm

# SLEPc/PETSc may only be initialized once per process. Track init + the options
# string (the static options enter the database at init; the per-attempt σ target
# is set programmatically via EPSSetTarget, so the string never changes per solve).
const _SLEPC_INITED = Ref(false)
const _SLEPC_OPTS = Ref{String}("")

"""Throw if PETSc was built with real scalars (results would be wrong)."""
function _assert_complex_scalars()
    if !(PetscScalar <: Complex)
        error("BiGSTARS.solve requires a complex-scalar PETSc/SLEPc build " *
              "(configure with --with-scalar-type=complex); got PetscScalar=$(PetscScalar).")
    end
end

# ---- distributed matrix build (unchanged from the old extension) -------------

function _build_petsc_mat(A_csr, N::Integer, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    M = MatCreate()
    MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, PetscInt(N), PetscInt(N))
    MatSetFromOptions(M)
    rstart, rend = MatGetOwnershipRange(M)

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
        k1 < k0 && continue
        row  = PetscInt(rstart + r - 1)
        cols = PetscInt.(local_colind[k0:k1])
        vs   = PetscScalar.(local_vals[k0:k1])
        MatSetValues(M, PetscInt(1), [row], PetscInt(length(cols)), cols, vs, INSERT_VALUES)
    end
    return M
end

# ---- gather eigenpairs to rank 0 (unchanged from the old extension) ----------

function _gather_eigenpairs(eps, A, nconv::Integer, N::Integer, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    rstart, rend = MatGetOwnershipRange(A)
    nlocal = rend - rstart
    counts = Cint.(MPI.Allgather(Int(nlocal), comm))

    λ = rank == 0 ? Vector{ComplexF64}(undef, nconv) : ComplexF64[]
    Χ = rank == 0 ? Matrix{ComplexF64}(undef, N, nconv) : zeros(ComplexF64, 0, 0)

    vr, vi = MatCreateVecs(A)
    for ie in 0:(nconv - 1)
        vpr, vpi, _, _ = EPSGetEigenpair(eps, ie, vr, vi)
        local_arr, local_ref = VecGetArray(vr)
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

# ---- adaptive-σ solve of one pencil (collective across all ranks) ------------

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

# ---- public entrypoint -------------------------------------------------------

"""
    solve(cache, k_values; sigma_0, nev=1, which=:LM, tol=1e-10, maxiter=300,
          ncv=0, mat_solver=:mumps, eps_type=:krylovschur,
          n_tries=8, Δσ₀=0.2, incre=1.2, ϵ=1e-5,
          manage_init=true, verbose=false) -> Vector{SolverResults}

Distributed eigensolve (SLEPc over PETSc), one pencil per wavenumber spread
across all MPI ranks, with adaptive-σ shift-and-invert. Collective: every rank
must call it. Rank 0 returns fully populated results; other ranks return empty
markers. Requires a complex-scalar PETSc/SLEPc build. Run under
`mpiexec -n P julia --project=. script.jl`.
"""
function solve(cache::DiscretizationCache, k_values::AbstractVector;
               sigma_0::Real, nev::Integer=1, which::Symbol=:LM,
               tol::Real=1e-10, maxiter::Integer=300, ncv::Integer=0,
               mat_solver::Symbol=:mumps, eps_type::Symbol=:krylovschur,
               n_tries::Integer=8, Δσ₀::Real=0.2, incre::Real=1.2, ϵ::Real=1e-5,
               manage_init::Bool=true, verbose::Bool=false)
    opts = _eps_options(; nev=Int(nev), which=which, tol=Float64(tol),
                        maxiter=Int(maxiter), ncv=Int(ncv),
                        mat_solver=String(mat_solver), eps_type=String(eps_type))

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
        A_csr = nothing; B_csr = nothing; N = 0
        if rank == 0
            A, B = assemble(cache, Float64(k))
            N = size(A, 1)
            A_csr = _to_csr(A)
            B_csr = _to_csr(B)
        end
        N = MPI.bcast(N, 0, comm)
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
solve(cache::DiscretizationCache; sigma_0::Real, kwargs...) =
    solve(cache, [0.0]; sigma_0=sigma_0, kwargs...)
```

- [ ] **Step 5: Rewrite `src/BiGSTARS.jl` usings/includes/exports/workload**

Replace the `using` block (lines ~3-12) with:

```julia
    using Printf
    using SparseArrays
    using LinearAlgebra
    using FFTW
    using PrecompileTools: @setup_workload, @compile_workload
```

Replace the `export` block so the eigensolver section reads (drop `EigenSolver`,
`solve!`, `get_results`, `SolverConfig`, `compare_methods!`, `solve_mpi`; keep
`solve`, `SolverResults`, `ConvergenceHistory`, `print_summary`):

```julia
        # Eigenvalue solver
        SolverResults,
        ConvergenceHistory,
        print_summary,
```

(Keep `solve` where it already is in the "Discretization and solving" export group.)

Replace the include section (lines ~70-100) with:

```julia
    # Core spectral operators (coefficient-space)
    include("ultraspherical.jl")
    include("fourier_coeff.jl")

    # Domain and problem types
    include("domain.jl")

    # Transforms (after domain.jl — differentiate needs Domain types)
    include("transforms.jl")
    include("expr.jl")
    include("evp.jl")

    # DSL
    include("macros.jl")
    include("substitutions.jl")
    include("lowering.jl")
    include("k_separation.jl")
    include("boundary.jl")

    # Eigenproblem result types (shared by the solver and reconstruction)
    include("results.jl")

    # Pure-Julia matrix prep for the distributed backend
    include("mpi_prep.jl")

    # Discretization (defines DiscretizationCache + assemble, used by the solver)
    include("discretize.jl")

    # Distributed eigensolver (SLEPc/PETSc) — sole solve backend
    include("slepc_solve.jl")

    include("reconstruct.jl")

    # Utilities
    include("utils.jl")
```

Replace the `@compile_workload` body so it stops before the eigensolve (SLEPc
cannot init at precompile time):

```julia
    @setup_workload begin
        @compile_workload begin
            dom = Domain(x = FourierTransformed(), z = Chebyshev(N=8, lower=0.0, upper=1.0))
            prob = EVP(dom, variables=[:u], eigenvalue=:sigma)
            @equation prob sigma * u == -dx(dx(u)) - dz(dz(u))
            @bc prob left(u) == 0
            @bc prob right(u) == 0
            cache = discretize(prob)
            assemble(cache, 1.0)   # exercise discretize→assemble; no eigensolve (needs SLEPc)
        end
    end
```

- [ ] **Step 6: Delete the obsolete files**

```bash
git rm src/eig_solver.jl src/solve.jl src/construct_linear_map.jl ext/BiGSTARSMPIExt.jl
rmdir ext 2>/dev/null || true
```

- [ ] **Step 7: Verify the package loads**

Run: `julia --project=. -e 'using Pkg; Pkg.instantiate(); using BiGSTARS; println("loaded; solve methods=", length(methods(BiGSTARS.solve)))'`
Expected: instantiation pulls `MPI`/`PetscWrap`/`SlepcWrap` via JLLs; prints `loaded; solve methods=2` with no `solve` name-collision error and no `KrylovKit`/`ArnoldiMethod`/`Arpack` in the manifest.

Then a non-solver test file to confirm core still works:
Run: `julia --project=. -e 'using Pkg; Pkg.test()'` — expect failures ONLY in test files still referencing deleted symbols (fixed in Task 3); `test_transforms`, `test_domain`, `test_mpi_prep`, etc. pass.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor!: SLEPc/PETSc as sole eigensolver; rename solve_mpi→solve; drop serial backends"
```

---

## Task 3: Trim the cross-platform test suite

**Files:**
- Modify: `test/test_eig_solver.jl`, `test/test_integration.jl`, `test/test_coverage_gaps.jl`

- [ ] **Step 1: Gut `test/test_eig_solver.jl` to kept utilities only**

Replace the whole file with (keeps only tests for retained symbols
`sort_evals`/`remove_evals`/`print_evals`/`sort_eigenvalues!`):

```julia
using Test
using LinearAlgebra

@testset "Eigen utilities (retained)" begin
    @testset "sort_eigenvalues! :nearest requires σ" begin
        @test_throws ArgumentError BiGSTARS.sort_eigenvalues!(
            ComplexF64[1.0, 2.0], Matrix{ComplexF64}(I, 2, 2), :nearest)
    end

    @testset "sort_eigenvalues! non-nearest criteria" begin
        λ = ComplexF64[3.0 + 0im, 1.0 + 2im, 2.0 - 1im]
        Χ = Matrix{ComplexF64}(I, 3, 3)
        ls, _ = BiGSTARS.sort_eigenvalues!(copy(λ), copy(Χ), :R; rev=true)
        @test real(ls[1]) == 3.0
        lm, _ = BiGSTARS.sort_eigenvalues!(copy(λ), copy(Χ), :M; rev=true)
        @test abs(lm[1]) ≈ maximum(abs.(λ))
    end

    @testset "eigenvalue utilities (sort/remove/print)" begin
        λ = ComplexF64[3.0 + 0im, 1.0 + 2im, 2.0 - 1im]
        Χ = Matrix{ComplexF64}(I, 3, 3)
        ls, _ = sort_evals(λ, Χ, :R; rev=true)
        @test real(ls[1]) == 3.0
        ls2, _ = sort_evals(λ, Χ, "M"; sorting="lm")
        @test abs(ls2[1]) ≈ maximum(abs.(λ))
        lr, χr = remove_evals(λ, Χ, 1.5, 3.5, "R")
        @test length(lr) == 2 && all(e -> 1.5 ≤ real(e) ≤ 3.5, lr)
        @test size(χr, 2) == 2
        @test length(remove_evals(λ, Χ, 1.5, 3.5, :R)[1]) == 2
        @test (print_evals(λ); true)
    end
end
```

- [ ] **Step 2: Trim `test/test_integration.jl`**

Delete two testsets entirely:
- `@testset "Sparse solve path matches dense path" begin … end` (uses
  `solve(...; method=:Arnoldi, sparse=…)` — concept removed).
- `@testset "Spurious modes filtered from descriptor (augmented) solve" begin … end`
  (uses `solve(...; method=…)`; its SLEPc equivalent is added to the MPI suite in
  Task 4).

Leave every other testset unchanged — they verify via `eigvals(Matrix(A),
Matrix(B))` / `eigen(...)` (LinearAlgebra, cross-platform) and
`BiGSTARS._assembled_density(...)` (now defined in `discretize.jl`).

- [ ] **Step 3: Trim `test/test_coverage_gaps.jl`**

Delete these testsets (titles are exact anchors):
- `@testset "eig_solver: get_results / print_summary on un-run solver"` (~line 43)
- `@testset "eig_solver: all attempts fail throws"` (~line 50)
- `@testset "eig_solver: compare_methods! verbose summary"` (~line 61)
- `@testset "solve: parallel paths and failure → failed_result"` (~line 277)

Keep `@testset "eig_solver: sort_eigenvalues! non-nearest criteria"` (~line 31) —
`sort_eigenvalues!` is retained.

In `@testset "discretize: legacy-derived problem solved via in-place assembly"`
(~line 423), replace the two solver lines:

```julia
        res = solve(cache, [1.0]; sigma_0=0.1, method=:Arnoldi, sparse=false)  # dense in-place
        @test res[1] isa SolverResults
```

with an assemble + dense-eigvals check (keeps the discretize coverage,
cross-platform):

```julia
        A, B = assemble(cache, 1.0)
        ev = eigvals(Matrix(A), Matrix(B))
        @test any(e -> isfinite(e), ev)
```

Ensure `using LinearAlgebra` is present at the top of `test_coverage_gaps.jl`
(add it if missing, since `eigvals` is now used here).

- [ ] **Step 4: Run the cross-platform suite, verify green**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS for the full `runtests.jl` set (no references to deleted symbols remain).

- [ ] **Step 5: Commit**

```bash
git add test/test_eig_solver.jl test/test_integration.jl test/test_coverage_gaps.jl
git commit -m "test: trim serial-solver tests; keep DSL/assembly + retained-utility coverage"
```

---

## Task 4: MPI solver test suite — analytic correctness pin

**Files:**
- Modify: `test/mpi/test_slepc.jl`

This replaces the deleted serial reference with an **analytic** one. Verification
is **CI (mpi.yml)** — it needs a complex PETSc/SLEPc.

- [ ] **Step 1: Rewrite `test/mpi/test_slepc.jl`**

```julia
# Run with: mpiexec -n {1,2} julia --project=test/mpi test/mpi/test_slepc.jl
# Requires a complex-scalar system PETSc/SLEPc (PETSC_DIR/PETSC_ARCH/SLEPC_DIR)
# and MPI.jl bound to the same MPI.
#
# `solve` manages SlepcInitialize itself, so this script does not call it.
# PetscWrap/SlepcWrap are loaded only to provide the backend; use `import`, NOT
# `using`: PetscWrap exports `solve`, which would shadow BiGSTARS.solve.
using BiGSTARS
using MPI
import PetscWrap, SlepcWrap
using Test
using LinearAlgebra

rank = MPI.Comm_rank(MPI.COMM_WORLD)

# σ u = -dx²u - dz²u, u(0)=u(1)=0 on z∈[0,1], at wavenumber k.
# dx → ×k² (FourierTransformed), -dz² eigenvalues are (nπ)² ⇒ σ_n = k² + (nπ)².
dom = Domain(x = FourierTransformed(), z = Chebyshev(N=24, lower=0.0, upper=1.0))
prob = EVP(dom, variables=[:u], eigenvalue=:sigma)
@equation prob sigma * u == -dx(dx(u)) - dz(dz(u))
@bc prob left(u) == 0
@bc prob right(u) == 0
cache = discretize(prob)

k = 1.0
analytic(n) = k^2 + (n * π)^2          # σ_1 ≈ 10.8696, σ_2 ≈ 40.48, σ_3 ≈ 89.83

# nev=1 nearest the smallest mode
res1 = solve(cache, [k]; sigma_0=10.0, nev=1, which=:LM, tol=1e-10)
# nev=3 nearest the smallest three modes
res3 = solve(cache, [k]; sigma_0=10.0, nev=3, which=:LM, tol=1e-10)

if rank == 0
    A, B = BiGSTARS.assemble(cache, k)   # serial matrices for the residual check
    ts = @testset "SLEPc analytic reference" begin
        @test res1[1].converged
        λ1 = res1[1].eigenvalues[1]
        @test isapprox(real(λ1), analytic(1); rtol=1e-4)   # matches analytic σ_1
        @test abs(imag(λ1)) < 1e-6

        χ = res1[1].eigenvectors[:, 1]
        @test norm(A * χ - λ1 * (B * χ)) / norm(χ) < 1e-6  # residual (also checks the gather)

        @test res3[1].converged
        @test length(res3[1].eigenvalues) ≥ 3
        got = sort(real.(res3[1].eigenvalues[1:3]))
        for n in 1:3
            @test minimum(abs.(got .- analytic(n))) < 1e-3  # σ_1,σ_2,σ_3 all present
        end

        @test !isempty(res1[1].history.attempts)            # adaptive history populated
        println("σ_1 SLEPc=$(real(λ1))  analytic=$(analytic(1))")
    end
    ts.anynonpass && exit(1)
end

# Spurious-mode filter: augmented descriptor system has a singular B (infinite
# modes) — the filter must drop them. Physical σ = -1/(nπ)²; nearest -0.1 is -1/π².
dom2 = Domain(z = Chebyshev(N=48, lower=0.0, upper=1.0))
prob2 = EVP(dom2, variables=[:psi], eigenvalue=:sigma)
@derive prob2 v dz(dz(v)) = psi
@derive_bc prob2 v left(v) == 0
@derive_bc prob2 v right(v) == 0
@equation prob2 sigma * psi == v
@bc prob2 left(psi) == 0
@bc prob2 right(psi) == 0
cache2 = discretize(prob2; augment_derived=true)

res_sp = solve(cache2; sigma_0=-0.1, nev=4, n_tries=2)
if rank == 0
    ts2 = @testset "SLEPc spurious-mode filter (singular B)" begin
        @test res_sp[1].converged
        @test all(e -> abs(e) < 0.5, res_sp[1].eigenvalues)              # no infinite modes survive
        @test minimum(abs.(res_sp[1].eigenvalues .- (-1 / π^2))) < 1e-3  # physical n=1 present
    end
    ts2.anynonpass && exit(1)
end
```

- [ ] **Step 2: Commit**

```bash
git add test/mpi/test_slepc.jl
git commit -m "test(mpi): analytic SLEPc correctness pin (k²+(nπ)²), nev, residual, spurious filter"
```

Verification: **CI (mpi.yml)** at `-n 1` and `-n 2`. Cannot pass locally without
a complex PETSc/SLEPc.

---

## Task 5: CI — make the solver job blocking

**Files:**
- Modify: `.github/workflows/mpi.yml`

- [ ] **Step 1: Remove `continue-on-error` from the `mpi-slepc` job**

In `.github/workflows/mpi.yml`, delete the line:

```yaml
    continue-on-error: true        # non-blocking until a green run confirms the wrapper API
```

so the job (and its `-n 1` / `-n 2` integration steps) gates the PR.

- [ ] **Step 2: Confirm `CI.yml` needs no change**

`CI.yml`'s `julia-runtest` runs the trimmed `runtests.jl`; `julia-buildpkg`
installs `MPI`/`PetscWrap`/`SlepcWrap` via JLLs. No edit required. (If buildpkg
fails to precompile on a runner, confirm the manifest resolved PETSc_jll — see
Task 8.)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/mpi.yml
git commit -m "ci: make the MPI/SLEPc job blocking (now the sole solver gate)"
```

---

## Task 6: Convert examples to `solve` + MPI

**Files:**
- Modify: `examples/Eady.jl`, `examples/Stone1971.jl`, `examples/rRBC.jl`, `examples/Project.toml`
- Reference template: `examples/eady_mpi.jl` (already uses the distributed pattern)

- [ ] **Step 1: Update `examples/Project.toml`**

Add to `[deps]` (and remove `ArnoldiMethod`/`Arpack`/`KrylovKit` lines if present):

```toml
MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
PetscWrap = "5be22e1c-01b5-4697-96eb-ef9ccdc854b8"
SlepcWrap = "c3679e3b-785e-4ccc-b734-b7685cbb935e"
```

- [ ] **Step 2: Convert each of `Eady.jl`, `Stone1971.jl`, `rRBC.jl`**

For each file apply this transformation (mirroring `examples/eady_mpi.jl`):

1. After `using BiGSTARS`, add the backend imports (NOTE `import`, not `using`,
   for PetscWrap/SlepcWrap to avoid shadowing `solve`):

```julia
using MPI
import PetscWrap, SlepcWrap
```

2. Replace each serial solve call, dropping the removed kwargs
   (`method`, `parallel`, `sparse`). For example:

```julia
# before:
# results = solve(cache, k_values; sigma_0=0.02, method=:Krylov, nev=5)
# after:
results = solve(cache, k_values; sigma_0=0.02, nev=5)
```

3. Guard any printing/plotting of results so only rank 0 acts:

```julia
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    # existing post-processing / plotting of `results`
end
```

- [ ] **Step 3: Commit**

```bash
git add examples/
git commit -m "examples: run on the SLEPc solve entrypoint (mpiexec -n P)"
```

Verification: scripts require a complex PETSc/SLEPc to run; they are not part of
automated CI. Confirm they parse: `julia --project=examples -e 'include("examples/Eady.jl")'`
will error at the solve step without complex PETSc, which is expected — the load
+ discretize portion must run without syntax/method errors.

---

## Task 7: Documentation

**Files:**
- Modify: `docs/src/mpi.md`, `docs/src/index.md`, `docs/src/method.md`, `docs/src/performance.md`, `docs/src/equation_dsl.md`, `docs/src/installation_instructions.md`, `docs/src/literated/Eady.md`, `docs/src/literated/Stone1971.md`, `docs/src/literated/rRBC.md`

- [ ] **Step 1: `docs/src/mpi.md`**

Remove the `!!! warning "Experimental"` admonition block. Replace every
`solve_mpi` with `solve`. Change "only the solve call changes (`solve_mpi`
instead of `solve`/`EigenSolver`)" to state that `solve` IS the distributed
entrypoint. Update the "Notes" so it no longer says "matching the serial
backends" (there is no serial backend) and drop the "v1 does a single solve …"
line (adaptive-σ now applies).

- [ ] **Step 2: `installation_instructions.md`**

Add a section: `]add BiGSTARS` installs everywhere (PETSc_jll), but **solving**
requires a complex-scalar PETSc/SLEPc build (`./configure
--with-scalar-type=complex`) with `PETSC_DIR`/`PETSC_ARCH`/`SLEPC_DIR` exported,
and MPI.jl bound to that MPI. Reference the `mpi.md` environment-setup block.

- [ ] **Step 3: `index.md`, `method.md`, `performance.md`, `equation_dsl.md`, literated pages**

Replace serial-solver references: any `method=:Krylov/:Arnoldi/:Arpack`,
`EigenSolver`, `compare_methods!`, `solve!`, `get_results`, `parallel=`,
`sparse=` → the `solve(cache, k_values; sigma_0, nev, …)` form. Remove
multi-backend comparison prose. In `performance.md`, replace the
threaded-wavenumber-sweep performance story with the MPI-distributed model (one
pencil per wavenumber across ranks).

Find offenders with:

```bash
grep -rnE "solve_mpi|method\s*=\s*:(Krylov|Arnoldi|Arpack)|EigenSolver|compare_methods|solve!|get_results|parallel\s*=|sparse\s*=" docs/src
```

- [ ] **Step 4: Commit**

```bash
git add docs/src
git commit -m "docs: SLEPc solve as the only path; drop serial/multi-backend docs"
```

---

## Task 8: Final audit + get the solver gate green

**Files:** none new — verification + cleanup.

- [ ] **Step 1: Repo-wide grep for stragglers**

```bash
grep -rnE "KrylovKit|ArnoldiMethod|Arpack|solve_mpi|EigenSolver|SolverConfig|compare_methods|construct_linear_map|ShiftAndInvert|_factorize_shifted|solve_eigenvalue_problem|solve_krylov|solve_arnoldi|solve_arpack" src test docs examples
```
Expected: no hits in `src/`; only historical mentions in `docs/superpowers/`
(specs/plans) are acceptable. Fix anything else.

- [ ] **Step 2: Full cross-platform suite locally**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS.

- [ ] **Step 3: Confirm the manifest dropped the serial libs**

Run: `julia --project=. -e 'using Pkg; println(haskey(Pkg.project().dependencies, "KrylovKit"))'`
Expected: `false`.

- [ ] **Step 4: Push and drive `mpi.yml` green (the real deliverable)**

Push the branch. The now-blocking `mpi.yml` builds complex PETSc 3.22.2 / SLEPc
3.22.1 and runs `test/mpi/test_slepc.jl` at `-n 1` and `-n 2`. Debug failures
against the pinned PetscWrap 0.1.5 / SlepcWrap 0.1.x call surface (the wrapper
names in `slepc_solve.jl` were checked against the maintainer's Helmholtz example,
but this job has never gone green before). Iterate until both rank counts pass.

- [ ] **Step 5 (optional, note only): complex JLL probe**

Investigate whether `PetscWrap` can select a complex-scalar library from
`PETSc_jll` (would let laptop users + the cross-platform matrix solve without a
system build). Record findings in `docs/src/mpi.md`. Do not block this PR on it.

- [ ] **Step 6: Open the PR**

```bash
gh pr create --title "SLEPc/PETSc as the sole eigensolver" --body "<summary + link to spec/plan>"
```

(Only after the user authorizes commits/PR — see standing no-commit rule.)

---

## Self-review

**Spec coverage:** §A deps/module → T2; §B delete/keep → T2 (+results.jl, _assembled_density move); §C adaptive-σ → T1 (helper) + T2 (loop); §D workload → T2 step 5; §E tests → T3 (cross-platform) + T4 (analytic pin); §F CI → T5; §G examples/docs → T6/T7; §H risks/sequencing → T8. All covered.

**Placeholder scan:** No TBD/TODO; every code step has full code; deletions are anchored by exact `@testset` titles / file paths.

**Type consistency:** `solve` signature identical in `slepc_solve.jl` and docs/examples/tests; `SolverResults`/`ConvergenceHistory` fields match `results.jl`; `_sigma_schedule(σ₀, n_tries, Δσ₀, incre)` call in `_solve_one_adaptive` matches its definition; `_eps_options` new keyword set (no `sigma_0`) matches its single caller in `solve`; selective-import list covers every PetscWrap/SlepcWrap name used in `slepc_solve.jl` (`EPSSetTarget` added vs the old extension).
