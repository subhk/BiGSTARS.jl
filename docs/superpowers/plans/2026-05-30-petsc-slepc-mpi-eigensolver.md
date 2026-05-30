# PETSc/SLEPc MPI Eigensolver Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a distributed-memory (MPI) eigensolver backend to BiGSTARS built on SLEPc's `EPS` over PETSc `MatMPIAIJ`, reached through a separate `solve_mpi` entrypoint, with all PETSc/SLEPc/MPI code isolated in a package extension.

**Architecture:** Rank 0 assembles `A,B` serially with the existing pipeline. Pure-Julia helpers in `src/` convert to CSR and extract per-rank row-blocks (unit-tested without PETSc). The extension scatters the blocks, builds distributed `MatMPIAIJ`, runs a SLEPc Krylov-Schur shift-and-invert solve configured through the PETSc options database, gathers eigenvectors to rank 0, and returns `SolverResults` reusing the existing sort/filter helpers.

**Tech Stack:** Julia 1.10+ package extensions; `MPI.jl`, `PetscWrap.jl`, `SlepcWrap.jl` (weakdeps, system-built complex-scalar PETSc/SLEPc); `SparseArrays`.

---

## Running tests in this repo (executor note)

The `juliaup` launcher and parts of `~/.julia` are root-owned. Use the real binary plus a writable depot overlay, and run unsandboxed:

```bash
JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia
export JULIA_DEPOT_PATH="/tmp/jdepot:/Users/subha/.julia"
```

Run one new test file standalone:

```bash
$JL --project=. -e 'using BiGSTARS, Test, SparseArrays, LinearAlgebra; include("test/test_mpi_prep.jl")'
```

Run the full suite:

```bash
$JL --project=. -e 'using Pkg; Pkg.test()'
```

Tasks 1–4 are pure-Julia and run in this environment / normal CI. Tasks 5–9 are the extension (PETSc/SLEPc/MPI glue) and **cannot run here** — they are verified by the opt-in MPI integration test in Task 10, which requires a complex-scalar system PETSc/SLEPc and `mpiexec`. For extension tasks, "verify" means: code compiles when the weakdeps load, and the Task 10 integration test passes in a PETSc-equipped environment.

---

## File structure

| File | Responsibility |
|---|---|
| `src/mpi_prep.jl` (create) | Pure-Julia: CSR conversion + row-block extraction. No PETSc. |
| `src/eig_solver.jl` (modify) | Add `solve_mpi` generic + not-loaded fallback that throws an install hint. |
| `src/BiGSTARS.jl` (modify) | `include("mpi_prep.jl")`; export `solve_mpi`. |
| `ext/BiGSTARSMPIExt.jl` (create) | All MPI/PetscWrap/SlepcWrap glue: scatter, build Mat, EPS solve, gather, build `SolverResults`. |
| `Project.toml` (modify) | `[weakdeps]`, `[extensions]`, `[compat]` for MPI/PetscWrap/SlepcWrap. |
| `test/test_mpi_prep.jl` (create) | Unit tests for the pure-Julia helpers. |
| `test/runtests.jl` (modify) | Include the new unit test file. |
| `test/mpi/test_slepc.jl` (create) | Opt-in MPI integration test (run under `mpiexec`). |
| `.github/workflows/mpi.yml` (create) | Opt-in CI job building PETSc/SLEPc and running the MPI test. |
| `docs/src/mpi.md` (create) | User docs: install, env vars, usage. |

---

## Task 1: CSR conversion helper

**Files:**
- Create: `src/mpi_prep.jl`
- Modify: `src/BiGSTARS.jl` (add include)
- Test: `test/test_mpi_prep.jl`

- [ ] **Step 1: Add the include and export**

In `src/BiGSTARS.jl`, add to the export list (after `compare_methods!,`):

```julia
        solve_mpi,
```

And add an include after the eigenvalue solver includes (after `include("eig_solver.jl")`):

```julia
    include("mpi_prep.jl")
```

- [ ] **Step 2: Write the failing test**

Create `test/test_mpi_prep.jl`:

```julia
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
end
```

- [ ] **Step 3: Run the test, expect failure**

```bash
$JL --project=. -e 'using BiGSTARS, Test, SparseArrays, LinearAlgebra; include("test/test_mpi_prep.jl")'
```

Expected: FAIL — `_to_csr` not defined (`UndefVarError`).

- [ ] **Step 4: Implement `_to_csr`**

Create `src/mpi_prep.jl`:

```julia
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
```

- [ ] **Step 5: Run the test, expect pass**

```bash
$JL --project=. -e 'using BiGSTARS, Test, SparseArrays, LinearAlgebra; include("test/test_mpi_prep.jl")'
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mpi_prep.jl src/BiGSTARS.jl test/test_mpi_prep.jl
git commit -m "feat: add _to_csr CSR conversion helper for MPI backend"
```

---

## Task 2: Row-block extraction helper

**Files:**
- Modify: `src/mpi_prep.jl`
- Test: `test/test_mpi_prep.jl`

- [ ] **Step 1: Write the failing test**

Append inside the `@testset "MPI prep helpers"` block in `test/test_mpi_prep.jl`:

```julia
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
```

- [ ] **Step 2: Run, expect failure**

```bash
$JL --project=. -e 'using BiGSTARS, Test, SparseArrays, LinearAlgebra; include("test/test_mpi_prep.jl")'
```

Expected: FAIL — `_csr_row_block` not defined.

- [ ] **Step 3: Implement `_csr_row_block`**

Append to `src/mpi_prep.jl`:

```julia
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
```

- [ ] **Step 4: Run, expect pass**

```bash
$JL --project=. -e 'using BiGSTARS, Test, SparseArrays, LinearAlgebra; include("test/test_mpi_prep.jl")'
```

Expected: PASS (both new testsets).

- [ ] **Step 5: Commit**

```bash
git add src/mpi_prep.jl test/test_mpi_prep.jl
git commit -m "feat: add _csr_row_block row-block extraction for MPI scatter"
```

---

## Task 3: Wire unit tests into the suite

**Files:**
- Modify: `test/runtests.jl`

- [ ] **Step 1: Add the include**

In `test/runtests.jl`, add inside the `@testset "BiGSTARS.jl"` block, after `include("test_eig_solver.jl")`:

```julia
    include("test_mpi_prep.jl")
```

- [ ] **Step 2: Run the full suite, expect pass**

```bash
$JL --project=. -e 'using Pkg; Pkg.test()'
```

Expected: PASS, including the new `MPI prep helpers` testset. No regressions.

- [ ] **Step 3: Commit**

```bash
git add test/runtests.jl
git commit -m "test: include MPI prep helper tests in the suite"
```

---

## Task 4: `solve_mpi` not-loaded fallback

**Files:**
- Modify: `src/eig_solver.jl`
- Test: `test/test_mpi_prep.jl`

- [ ] **Step 1: Write the failing test**

Append inside `@testset "MPI prep helpers"` in `test/test_mpi_prep.jl`:

```julia
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
```

- [ ] **Step 2: Run, expect failure**

```bash
$JL --project=. -e 'using BiGSTARS, Test, SparseArrays, LinearAlgebra; include("test/test_mpi_prep.jl")'
```

Expected: FAIL — `solve_mpi` not defined (`UndefVarError`).

- [ ] **Step 3: Add the generic + fallback**

In `src/eig_solver.jl`, append at the end (before the closing of the file, after `show_example_usage`):

```julia
# ==============================================================================
# Distributed (MPI) backend entrypoint
# ==============================================================================

"""
    solve_mpi(cache, k_values; sigma_0, nev=1, which=:LM, tol=1e-10, maxiter=300, kwargs...)

Distributed-memory eigensolve via SLEPc over PETSc, one eigenproblem per
wavenumber spread across all MPI ranks. Provided by the package extension
`BiGSTARSMPIExt`, which loads only when `MPI`, `PetscWrap`, and `SlepcWrap` are
all imported. Returns `Vector{SolverResults}`, fully populated on rank 0.

Run under `mpiexec -n P julia script.jl` with `SlepcInitialize()` /
`SlepcFinalize()` bracketing the work, and a complex-scalar system PETSc/SLEPc.
"""
function solve_mpi end

# Least-specific fallback: any concrete-typed method from the extension wins over
# this Vararg signature, so when the extension is loaded its real methods are
# called; otherwise this fires with an install hint.
function solve_mpi(@nospecialize(args...); @nospecialize(kwargs...))
    error("solve_mpi requires the distributed backend: install and import MPI, " *
          "PetscWrap, and SlepcWrap, plus a complex-scalar system PETSc/SLEPc " *
          "build (set SLEPC_DIR, PETSC_DIR, PETSC_ARCH). See docs/src/mpi.md.")
end
```

- [ ] **Step 4: Run, expect pass**

```bash
$JL --project=. -e 'using BiGSTARS, Test, SparseArrays, LinearAlgebra; include("test/test_mpi_prep.jl")'
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eig_solver.jl test/test_mpi_prep.jl
git commit -m "feat: add solve_mpi generic with not-loaded install-hint fallback"
```

---

## Task 5: Extension skeleton + Project.toml wiring

**Files:**
- Modify: `Project.toml`
- Create: `ext/BiGSTARSMPIExt.jl`

- [ ] **Step 1: Add weakdeps/extensions/compat to Project.toml**

In `Project.toml`, after the `[deps]` block add:

```toml
[weakdeps]
MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
PetscWrap = "5be11700-c0a3-4f3b-be46-7c9def6e4c80"
SlepcWrap = "1f867a9b-5d8a-4dd6-9c1d-9f9d5f3b3b3b"

[extensions]
BiGSTARSMPIExt = ["MPI", "PetscWrap", "SlepcWrap"]
```

And add to the existing `[compat]` block:

```toml
MPI = "0.20"
PetscWrap = "0.1, 0.2"
SlepcWrap = "0.1, 0.2"
```

> Note: confirm the `PetscWrap`/`SlepcWrap` UUIDs and current compat bounds against the General registry during execution (`] add PetscWrap SlepcWrap` in a scratch env, then read `~/.julia/registries`). Replace the placeholder `SlepcWrap` UUID above with the registered value.

- [ ] **Step 2: Create the extension skeleton**

Create `ext/BiGSTARSMPIExt.jl`:

```julia
module BiGSTARSMPIExt

using BiGSTARS
using BiGSTARS: _to_csr, _csr_row_block, SolverResults, ConvergenceHistory,
                sort_eigenvalues!, _filter_physical_modes
using MPI
using PetscWrap
using SlepcWrap
using SparseArrays
using LinearAlgebra: norm

# Implementations added in Tasks 6–9.

end # module
```

- [ ] **Step 3: Verify the package still loads (extension parses)**

```bash
$JL --project=. -e 'using BiGSTARS; println("base loads")'
```

Expected: prints `base loads` with no error (extension not triggered without the weakdeps; its syntax is still checked when precompiled if weakdeps are installed).

- [ ] **Step 4: Commit**

```bash
git add Project.toml ext/BiGSTARSMPIExt.jl
git commit -m "feat: scaffold BiGSTARSMPIExt extension and weakdep wiring"
```

---

## Task 6: Build the distributed PETSc matrix from scattered row-blocks

**Files:**
- Modify: `ext/BiGSTARSMPIExt.jl`

> Extension code — verified by the Task 10 integration test, not runnable in the base environment.

- [ ] **Step 1: Add `_build_petsc_mat`**

Append inside the module in `ext/BiGSTARSMPIExt.jl`:

```julia
"""
    _build_petsc_mat(A_csr_or_nothing, N, comm) -> PetscMat

Collectively build an N×N distributed `MatMPIAIJ`. PETSc decides the row
ownership; rank 0 (which holds the full CSR `(rowptr, colind, vals)`) ships each
rank exactly its owned rows, which the rank inserts with global column indices.
Non-root ranks pass `nothing` for the CSR.
"""
function _build_petsc_mat(A_csr, N::Integer, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    M = MatCreate(comm)
    MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, N, N)
    MatSetFromOptions(M)
    MatSetUp(M)
    rstart, rend = MatGetOwnershipRange(M)        # 0-based [rstart, rend)

    # Gather every rank's ownership range to rank 0.
    starts = MPI.Gather(Int(rstart), 0, comm)
    ends   = MPI.Gather(Int(rend),   0, comm)

    if rank == 0
        rowptr, colind, vals = A_csr
        # Send each non-root rank its CSR row-block; insert rank 0's own block directly.
        for p in 1:(nproc - 1)
            lrp, lci, lv = _csr_row_block(rowptr, colind, vals, starts[p + 1], ends[p + 1])
            MPI.send((lrp, lci, lv), comm; dest=p, tag=0)
        end
        lrp, lci, lv = _csr_row_block(rowptr, colind, vals, rstart, rend)
        _insert_rows!(M, rstart, lrp, lci, lv)
    else
        lrp, lci, lv = MPI.recv(comm; source=0, tag=0)
        _insert_rows!(M, rstart, lrp, lci, lv)
    end

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY)
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY)
    return M
end

"""
    _insert_rows!(M, rstart, local_rowptr, local_colind, local_vals)

Insert one rank's owned rows into the distributed matrix `M` via `MatSetValues`,
one row at a time, using global (0-based) row and column indices.
"""
function _insert_rows!(M, rstart::Integer,
                       local_rowptr, local_colind, local_vals)
    nrows = length(local_rowptr) - 1
    for r in 1:nrows
        k0 = local_rowptr[r] + 1
        k1 = local_rowptr[r + 1]
        k1 < k0 && continue                       # empty row
        cols = local_colind[k0:k1]                # global 0-based columns
        vs   = local_vals[k0:k1]
        MatSetValues(M, [rstart + r - 1], cols, vs, INSERT_VALUES)
    end
    return M
end
```

- [ ] **Step 2: Sanity-compile check (PETSc env only)**

In a PETSc-equipped scratch session:

```bash
mpiexec -n 1 julia --project=. -e 'using BiGSTARS, MPI, PetscWrap, SlepcWrap; println("ext compiles")'
```

Expected: prints `ext compiles`. (Full correctness is asserted in Task 10.)

- [ ] **Step 3: Commit**

```bash
git add ext/BiGSTARSMPIExt.jl
git commit -m "feat: build distributed PETSc matrix from scattered CSR row-blocks"
```

---

## Task 7: SLEPc EPS solve configured via the options database

**Files:**
- Modify: `ext/BiGSTARSMPIExt.jl`

> Extension code — verified by Task 10.

- [ ] **Step 1: Add the `which` mapping and options-string builder**

Append inside the module:

```julia
# Map BiGSTARS `which` to a SLEPc EPS option. `:LM` means "nearest the shift",
# matching the existing shift-and-invert convention (target magnitude).
const _WHICH_OPT = Dict(
    :LM => "target_magnitude",
    :LR => "largest_real",
    :SR => "smallest_real",
    :LI => "largest_imaginary",
    :SI => "smallest_imaginary",
)

"""
    _eps_options(; sigma_0, nev, which, tol, maxiter, ncv, mat_solver, eps_type) -> String

Build the PETSc/SLEPc options-database string that configures one solve:
Krylov-Schur EPS, shift-and-invert ST targeting `sigma_0`, with a parallel LU
(direct) factorization for the inner solves.
"""
function _eps_options(; sigma_0, nev, which, tol, maxiter, ncv, mat_solver, eps_type)
    haskey(_WHICH_OPT, which) || throw(ArgumentError("unsupported which=$which"))
    opts = "-eps_type $(eps_type) " *
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
```

- [ ] **Step 2: Add the single-problem solve `_solve_one`**

Append inside the module:

```julia
"""
    _solve_one(A, B, N, comm; sigma_0, nev, which, tol, maxiter, ncv,
               mat_solver, eps_type) -> SolverResults

Distributed solve of one generalized pencil `A x = λ B x`. Returns a populated
`SolverResults` on rank 0 and an empty marker result on other ranks. `A`/`B` are
the rank-0 CSR tuples (or `nothing` off-root).
"""
function _solve_one(A_csr, B_csr, N::Integer, comm::MPI.Comm;
                    sigma_0, nev, which, tol, maxiter, ncv, mat_solver, eps_type)
    rank = MPI.Comm_rank(comm)
    t0 = MPI.Wtime()

    A = _build_petsc_mat(A_csr, N, comm)
    B = _build_petsc_mat(B_csr, N, comm)

    # Push this solve's options into the database, then create/configure the EPS.
    opts = _eps_options(; sigma_0, nev, which, tol, maxiter, ncv, mat_solver, eps_type)
    PetscOptionsInsertString(opts)

    eps = EPSCreate(comm)
    EPSSetOperators(eps, A, B)
    EPSSetFromOptions(eps)
    EPSSetUp(eps)
    EPSSolve(eps)

    nconv = EPSGetConverged(eps)
    λ, Χ = _gather_eigenpairs(eps, A, nconv, N, comm)   # Task 8

    solve_time = MPI.Wtime() - t0
    if rank == 0
        return _assemble_results(λ, Χ, B_csr, sigma_0, nconv, solve_time)  # Task 8
    else
        return SolverResults(ComplexF64[], zeros(ComplexF64, 0, 0), false,
                             :Slepc, sigma_0, 0, solve_time, ConvergenceHistory())
    end
end
```

- [ ] **Step 3: Commit**

```bash
git add ext/BiGSTARSMPIExt.jl
git commit -m "feat: configure and run SLEPc EPS shift-and-invert solve"
```

---

## Task 8: Gather eigenpairs to rank 0 and assemble results

**Files:**
- Modify: `ext/BiGSTARSMPIExt.jl`

> Extension code — verified by Task 10.

- [ ] **Step 1: Add `_gather_eigenpairs`**

Append inside the module:

```julia
"""
    _gather_eigenpairs(eps, A, nconv, N, comm) -> (λ::Vector{ComplexF64}, Χ::Matrix{ComplexF64})

Pull the `nconv` converged eigenpairs onto rank 0. Eigenvalues are scalars
replicated on every rank. Each eigenvector is a distributed `Vec`; its locally
owned slice is gathered to rank 0 via `MPI.Gatherv` using the matrix row
ownership offsets. Non-root ranks get empty arrays.
"""
function _gather_eigenpairs(eps, A, nconv::Integer, N::Integer, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    rstart, rend = MatGetOwnershipRange(A)
    nlocal = rend - rstart
    counts = MPI.Allgather(Int(nlocal), comm)        # local sizes per rank, every rank

    λ = rank == 0 ? Vector{ComplexF64}(undef, nconv) : ComplexF64[]
    Χ = rank == 0 ? Matrix{ComplexF64}(undef, N, nconv) : zeros(ComplexF64, 0, 0)

    vr = VecCreate(comm); MatCreateVecs(A, vr, C_NULL)
    vi = VecCreate(comm); MatCreateVecs(A, vi, C_NULL)
    for ie in 0:(nconv - 1)
        vpr, vpi, _, _ = EPSGetEigenpair(eps, ie, vr, vi)
        local_part = VecGetArray(vr)                 # this rank's owned entries
        full = MPI.Gatherv(Vector{ComplexF64}(local_part), counts, 0, comm)
        VecRestoreArray(vr, local_part)
        if rank == 0
            λ[ie + 1] = ComplexF64(vpr, vpi)
            Χ[:, ie + 1] = full
        end
    end
    return λ, Χ
end
```

> Note: with a complex-scalar PETSc build, `vpi` is zero and the eigenvector imaginary part lives in the complex `Vec` values; `VecGetArray` returns `ComplexF64`. If the wrapper's `VecGetArray` returns reals, fall back to assembling `ComplexF64.(VecGetArray(vr), VecGetArray(vi))`. Confirm the return element type during Task 10 bring-up and keep whichever branch matches.

- [ ] **Step 2: Add `_assemble_results`**

Append inside the module:

```julia
"""
    _assemble_results(λ, Χ, B_csr, sigma_0, nconv, solve_time) -> SolverResults

Rank-0 assembly of the final result: drop spurious (singular-B) modes, sort by
distance from the shift, and wrap in `SolverResults`. `converged` is true when at
least one eigenpair was found.
"""
function _assemble_results(λ, Χ, B_csr, sigma_0, nconv, solve_time)
    B = sparse_from_csr(B_csr)                       # rank-0 SparseMatrixCSC for mass filter
    if nconv ≥ 1
        λ, Χ = _filter_physical_modes(λ, Χ, B)
        λ, Χ = sort_eigenvalues!(λ, Χ, :nearest; σ=sigma_0)
    end
    hist = ConvergenceHistory()
    return SolverResults(ComplexF64.(λ), ComplexF64.(Χ), nconv ≥ 1,
                         :Slepc, Float64(sigma_0), Int(nconv), solve_time, hist)
end

"""Rebuild a SparseMatrixCSC on rank 0 from the CSR tuple (for the mass filter)."""
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
```

- [ ] **Step 3: Commit**

```bash
git add ext/BiGSTARSMPIExt.jl
git commit -m "feat: gather eigenpairs to rank 0 and assemble SolverResults"
```

---

## Task 9: Public `solve_mpi` methods, option defaults, and validation

**Files:**
- Modify: `ext/BiGSTARSMPIExt.jl`

> Extension code — verified by Task 10.

- [ ] **Step 1: Add the complex-scalar build check**

Append inside the module:

```julia
"""Throw a clear error if PETSc was built with real scalars (results would be wrong)."""
function _assert_complex_scalars()
    # PetscScalar is the element type PETSc was compiled with.
    if !(PetscScalar <: Complex)
        error("BiGSTARS solve_mpi requires a complex-scalar PETSc/SLEPc build " *
              "(configure with --with-scalar-type=complex); got PetscScalar=$(PetscScalar).")
    end
end
```

> Note: if `PetscWrap` exposes the scalar type under a different name than `PetscScalar`, adjust this single reference during Task 10 bring-up.

- [ ] **Step 2: Add the `solve_mpi` methods**

Append inside the module:

```julia
"""
    solve_mpi(cache, k_values; sigma_0, nev=1, which=:LM, tol=1e-10, maxiter=300,
              ncv=0, mat_solver=:mumps, eps_type=:krylovschur, verbose=false)
        -> Vector{SolverResults}

Distributed eigensolve for each wavenumber. Collective: every MPI rank must call
it. Rank 0 returns fully populated results; other ranks return empty markers.
"""
function BiGSTARS.solve_mpi(cache::BiGSTARS.DiscretizationCache,
                            k_values::AbstractVector;
                            sigma_0::Real, nev::Integer=1, which::Symbol=:LM,
                            tol::Real=1e-10, maxiter::Integer=300, ncv::Integer=0,
                            mat_solver::Symbol=:mumps, eps_type::Symbol=:krylovschur,
                            verbose::Bool=false)
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
        N = MPI.bcast(N, 0, comm)                    # all ranks need the global size

        verbose && rank == 0 &&
            println("solve_mpi: k=$(k)  N=$(N)  σ₀=$(sigma_0)  nev=$(nev)")

        results[i] = _solve_one(A_csr, B_csr, N, comm;
                                sigma_0=Float64(sigma_0), nev=Int(nev), which=which,
                                tol=Float64(tol), maxiter=Int(maxiter), ncv=Int(ncv),
                                mat_solver=String(mat_solver), eps_type=String(eps_type))
    end
    return results
end

"""Single-problem overload (no wavenumber sweep)."""
function BiGSTARS.solve_mpi(cache::BiGSTARS.DiscretizationCache; sigma_0::Real, kwargs...)
    return BiGSTARS.solve_mpi(cache, [0.0]; sigma_0=sigma_0, kwargs...)
end
```

> Confirm `BiGSTARS.assemble(cache, k)` returns `(A, B)::SparseMatrixCSC` (see `src/discretize.jl` / `src/solve.jl::_solve_sparse`). It does — `_solve_sparse` calls `A, B = assemble(cache, k_val)` and relies on sparse storage.

- [ ] **Step 3: Export `DiscretizationCache`/`assemble` access**

These are already exported from `BiGSTARS` (`assemble`, `DiscretizationCache` in the export list). No change needed; verify by grepping `src/BiGSTARS.jl` for both names.

- [ ] **Step 4: Commit**

```bash
git add ext/BiGSTARSMPIExt.jl
git commit -m "feat: add solve_mpi methods, option defaults, complex-scalar check"
```

---

## Task 10: Opt-in MPI integration test + CI job

**Files:**
- Create: `test/mpi/test_slepc.jl`
- Create: `.github/workflows/mpi.yml`

- [ ] **Step 1: Write the MPI integration test**

Create `test/mpi/test_slepc.jl`:

```julia
# Run with: mpiexec -n 2 julia --project=. test/mpi/test_slepc.jl
# Requires a complex-scalar system PETSc/SLEPc (SLEPC_DIR/PETSC_DIR/PETSC_ARCH).
using BiGSTARS
using MPI, PetscWrap, SlepcWrap
using Test
using LinearAlgebra

SlepcInitialize()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Small Poisson-type EVP with a known serial answer.
dom = Domain(x = FourierTransformed(), z = Chebyshev(N=16, lower=0.0, upper=1.0))
prob = EVP(dom, variables=[:u], eigenvalue=:sigma)
@equation prob sigma * u == -dx(dx(u)) - dz(dz(u))
@bc prob left(u) == 0
@bc prob right(u) == 0
cache = discretize(prob)

# Distributed solve.
res = solve_mpi(cache, [1.0]; sigma_0=10.0, nev=1, which=:LM, tol=1e-10)

# Serial reference on rank 0.
if rank == 0
    ref = solve(cache, [1.0]; sigma_0=10.0, method=:Krylov, nev=1, verbose=false)
    @test res[1].converged
    @test ref[1].converged
    λ_mpi = res[1].eigenvalues[1]
    λ_ser = ref[1].eigenvalues[1]
    @test abs(λ_mpi - λ_ser) < 1e-6
    println("MPI λ=$(λ_mpi)  serial λ=$(λ_ser)  ✓")
end

SlepcFinalize()
```

- [ ] **Step 2: Write the opt-in CI workflow**

Create `.github/workflows/mpi.yml`:

```yaml
name: MPI/SLEPc
on:
  workflow_dispatch:        # opt-in: run manually; not on every push
jobs:
  mpi-slepc:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install PETSc/SLEPc (complex)
        run: |
          sudo apt-get update
          sudo apt-get install -y petsc-dev slepc-dev openmpi-bin libopenmpi-dev
          # Note: distro PETSc may be real-scalar. For a complex build, compile
          # PETSc/SLEPc with --with-scalar-type=complex and export the vars below.
          echo "PETSC_DIR=/usr/lib/petscdir" >> $GITHUB_ENV
          echo "SLEPC_DIR=/usr/lib/slepcdir" >> $GITHUB_ENV
      - uses: julia-actions/setup-julia@v2
        with:
          version: '1.10'
      - name: Instantiate + add weakdeps
        run: julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.add(["MPI","PetscWrap","SlepcWrap"])'
      - name: Run MPI integration test
        run: mpiexec -n 2 julia --project=. test/mpi/test_slepc.jl
```

- [ ] **Step 3: Run locally in a PETSc-equipped environment**

```bash
mpiexec -n 2 julia --project=. test/mpi/test_slepc.jl
```

Expected: prints `MPI λ=...  serial λ=...  ✓` and the test passes. (Skipped where PETSc/SLEPc is unavailable.)

- [ ] **Step 4: Commit**

```bash
git add test/mpi/test_slepc.jl .github/workflows/mpi.yml
git commit -m "test: add opt-in MPI/SLEPc integration test and CI job"
```

---

## Task 11: User documentation

**Files:**
- Create: `docs/src/mpi.md`
- Modify: `docs/make.jl` (add the page to the nav, if a pages list exists)

- [ ] **Step 1: Write the docs page**

Create `docs/src/mpi.md`:

```markdown
# Distributed (MPI) eigensolver

For problems too large for the in-process backends, BiGSTARS can run the
eigensolve across MPI ranks using SLEPc over PETSc. Assembly stays serial on
rank 0; the shift-and-invert factorization and Krylov eigensolve are distributed.

## Requirements

- A **complex-scalar** system build of PETSc and SLEPc
  (`./configure --with-scalar-type=complex`), with `PETSC_DIR`, `PETSC_ARCH`,
  and `SLEPC_DIR` exported.
- Julia packages `MPI`, `PetscWrap`, `SlepcWrap` installed. Importing all three
  activates the `BiGSTARSMPIExt` extension and the `solve_mpi` entrypoint.

## Usage

```julia
using BiGSTARS, MPI, PetscWrap, SlepcWrap

SlepcInitialize()
cache = discretize(prob)
results = solve_mpi(cache, k_values;
                    sigma_0=0.02, nev=5, which=:LM,
                    tol=1e-10, mat_solver=:mumps)
# results is fully populated on rank 0; other ranks get empty markers.
SlepcFinalize()
```

Run with `mpiexec -n P julia --project=. script.jl`.

## Notes

- `which=:LM` targets eigenvalues nearest `sigma_0` (shift-and-invert), matching
  the serial backends. `:LR`/`:SR`/`:LI`/`:SI` select by real/imaginary extremes.
- `mat_solver` picks the parallel direct solver for the inner solves
  (`:mumps`, `:superlu_dist`, or `:petsc`).
- v1 does a single solve at `sigma_0` (no adaptive-σ retry) and gathers
  eigenvectors to rank 0 for reconstruction.
```

- [ ] **Step 2: Add to docs nav (if applicable)**

If `docs/make.jl` has a `pages = [...]` vector, add:

```julia
    "Distributed (MPI)" => "mpi.md",
```

If it auto-discovers pages, no change is needed.

- [ ] **Step 3: Commit**

```bash
git add docs/src/mpi.md docs/make.jl
git commit -m "docs: document the distributed MPI/SLEPc eigensolver backend"
```

---

## Self-review notes

- **Spec coverage:** module layout (Task 5), public API (Tasks 4, 9), data flow steps 1–8 (Tasks 1–2 CSR/partition, 6 build, 7 solve, 8 gather/assemble), config mapping (Task 7), error handling (Tasks 4, 9 + nconv in 8), testing/CI (Tasks 3, 10). All spec sections map to tasks.
- **Refinement vs spec:** PETSc row ownership is queried at runtime (`MatGetOwnershipRange`) rather than predicted, so the unit-tested core is CSR-convert + row-block extraction. Documented in Task 6.
- **Type consistency:** `_to_csr`/`_csr_row_block` signatures are stable across Tasks 1–8; `_solve_one` → `_gather_eigenpairs` → `_assemble_results` chain uses matching `(λ, Χ)` and CSR-tuple types; `SolverResults` constructed with the existing positional field order from `src/eig_solver.jl`.
- **Open API confirmations (do during execution):** registered `PetscWrap`/`SlepcWrap` UUIDs + compat (Task 5); `VecGetArray` element type and `PetscScalar` name (Tasks 8, 9); `PetscOptionsInsertString` exact name in PetscWrap (Task 7). Each is flagged inline at its task.
