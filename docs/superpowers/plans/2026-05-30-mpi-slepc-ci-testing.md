# MPI/SLEPc Backend CI Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the distributed `solve_mpi` backend real GitHub CI coverage — always-on pure-Julia unit tests plus a cached real-PETSc/SLEPc integration job that runs `solve_mpi` end-to-end.

**Architecture:** Relocate the extension's pure-Julia logic (`_eps_options`, `_WHICH_OPT`, `sparse_from_csr`) into `src/mpi_prep.jl` so it runs in the normal suite without PETSc; the extension imports it. Harden the MPI integration script to fail CI on error and check the eigenpair residual. Rewrite the MPI workflow to source-build a cached complex-scalar PETSc/SLEPc, bind `MPI.jl` to that MPI, and run the test at n=1 and n=2 from a dedicated `test/mpi/Project.toml` environment.

**Tech Stack:** Julia 1.10+ package extensions, `SparseArrays`; CI: GitHub Actions, OpenMPI, source-built PETSc/SLEPc (complex), `MPIPreferences`, `MPI`/`PetscWrap`/`SlepcWrap`.

---

## Running Tier-1 tests in this repo (executor note)

The `juliaup` launcher and parts of `~/.julia` are root-owned. Use the real binary + writable depot overlay, and run unsandboxed (Bash tool: `dangerouslyDisableSandbox: true`):

```bash
JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia
export JULIA_DEPOT_PATH="/tmp/jdepot:/Users/subha/.julia"
```

Run the MPI-prep unit file standalone:

```bash
$JL --project=. -e 'using BiGSTARS, Test, SparseArrays, LinearAlgebra; include("test/test_mpi_prep.jl")'
```

Full suite:

```bash
$JL --project=. -e 'using Pkg; Pkg.test()'
```

Tasks 1–3 are pure-Julia/refactor and run here. Tasks 4–5 (integration test hardening + workflow) **cannot run here** (no PETSc/SLEPc/mpiexec) — they are exercised only by the GitHub `MPI/SLEPc` job. For those, "verify" means: the script parses, and the workflow is valid YAML.

## Starting state

This branch (`feature/mpi-slepc-ci-testing`) already contains the full MPI feature:
`src/mpi_prep.jl` (`_to_csr`, `_csr_row_block`), `ext/BiGSTARSMPIExt.jl` (full glue,
currently including `_eps_options`/`_WHICH_OPT`/`sparse_from_csr`),
`test/test_mpi_prep.jl`, `test/mpi/test_slepc.jl`, `.github/workflows/mpi.yml`.

## File structure

| File | Change |
|---|---|
| `src/mpi_prep.jl` | Add `_WHICH_OPT`, `_eps_options`, `sparse_from_csr` (pure-Julia). |
| `test/test_mpi_prep.jl` | Add Tier-1 unit tests for the three relocated helpers. |
| `ext/BiGSTARSMPIExt.jl` | Remove the three now-relocated defs; import `_eps_options`, `sparse_from_csr` from `BiGSTARS`. |
| `test/mpi/test_slepc.jl` | Harden: `@testset` + nonzero exit, residual check, guarded finalize. |
| `test/mpi/Project.toml` | New dedicated env (BiGSTARS + MPI/PetscWrap/SlepcWrap/MPIPreferences/Test/LinearAlgebra). |
| `.github/workflows/mpi.yml` | Rewrite: cached complex source build, MPIPreferences, n=1/n=2 run, path triggers, continue-on-error. |

---

## Task 1: Relocate `_WHICH_OPT` + `_eps_options` into `src/` with unit tests

**Files:**
- Modify: `src/mpi_prep.jl`
- Test: `test/test_mpi_prep.jl`

- [ ] **Step 1: Write the failing tests**

APPEND these testsets INSIDE the existing `@testset "MPI prep helpers" begin ... end` block in `test/test_mpi_prep.jl` (before its closing `end`):

```julia
    @testset "_WHICH_OPT maps BiGSTARS which-symbols to SLEPc options" begin
        @test BiGSTARS._WHICH_OPT[:LM] == "target_magnitude"
        @test BiGSTARS._WHICH_OPT[:LR] == "largest_real"
        @test BiGSTARS._WHICH_OPT[:SR] == "smallest_real"
        @test BiGSTARS._WHICH_OPT[:LI] == "largest_imaginary"
        @test BiGSTARS._WHICH_OPT[:SI] == "smallest_imaginary"
    end

    @testset "_eps_options builds the SLEPc options string" begin
        s = BiGSTARS._eps_options(; sigma_0=0.5, nev=5, which=:LM, tol=1e-10,
                                  maxiter=300, ncv=0, mat_solver="mumps",
                                  eps_type="krylovschur")
        @test occursin("-eps_type krylovschur", s)
        @test occursin("-eps_nev 5", s)
        @test occursin("-eps_target 0.5", s)
        @test occursin("-eps_target_magnitude", s)
        @test occursin("-st_type sinvert", s)
        @test occursin("-st_pc_factor_mat_solver_type mumps", s)
        @test !occursin("-eps_ncv", s)                 # ncv=0 omitted

        s2 = BiGSTARS._eps_options(; sigma_0=1.0, nev=2, which=:SR, tol=1e-8,
                                   maxiter=100, ncv=20, mat_solver="superlu_dist",
                                   eps_type="krylovschur")
        @test occursin("-eps_ncv 20", s2)              # ncv>0 included
        @test occursin("-eps_smallest_real", s2)

        # Unsupported `which` is rejected.
        @test_throws ArgumentError BiGSTARS._eps_options(;
            sigma_0=0.0, nev=1, which=:XX, tol=1e-10, maxiter=10, ncv=0,
            mat_solver="mumps", eps_type="krylovschur")
    end
```

- [ ] **Step 2: Run the tests, expect failure**

```bash
$JL --project=. -e 'using BiGSTARS, Test, SparseArrays, LinearAlgebra; include("test/test_mpi_prep.jl")'
```

Expected: FAIL — `_WHICH_OPT` / `_eps_options` not defined in `BiGSTARS` (`UndefVarError`).

- [ ] **Step 3: Add the definitions to `src/mpi_prep.jl`**

APPEND to `src/mpi_prep.jl` (after `_csr_row_block`):

```julia
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

- [ ] **Step 4: Run the tests, expect pass**

```bash
$JL --project=. -e 'using BiGSTARS, Test, SparseArrays, LinearAlgebra; include("test/test_mpi_prep.jl")'
```

Expected: PASS (all existing + the two new testsets).

- [ ] **Step 5: Commit**

```bash
git add src/mpi_prep.jl test/test_mpi_prep.jl
git commit -m "feat: relocate _eps_options/_WHICH_OPT into src with unit tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
After committing, run `git log --oneline -1`; if a hook rewrote the message to something generic (e.g. "updates"), fix with `git commit --amend -m "<message above>"`. Do not touch earlier commits or switch branches.

---

## Task 2: Relocate `sparse_from_csr` into `src/` with a round-trip test

**Files:**
- Modify: `src/mpi_prep.jl`
- Test: `test/test_mpi_prep.jl`

- [ ] **Step 1: Write the failing test**

APPEND inside the `@testset "MPI prep helpers"` block in `test/test_mpi_prep.jl`:

```julia
    @testset "sparse_from_csr is the inverse of _to_csr" begin
        A = sprand(ComplexF64, 10, 10, 0.3)
        csr = BiGSTARS._to_csr(A)
        @test BiGSTARS.sparse_from_csr(csr) ≈ A
    end
```

- [ ] **Step 2: Run, expect failure**

```bash
$JL --project=. -e 'using BiGSTARS, Test, SparseArrays, LinearAlgebra; include("test/test_mpi_prep.jl")'
```

Expected: FAIL — `sparse_from_csr` not defined in `BiGSTARS`.

- [ ] **Step 3: Add `sparse_from_csr` to `src/mpi_prep.jl`**

APPEND to `src/mpi_prep.jl` (after `_eps_options`):

```julia
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
```

- [ ] **Step 4: Run, expect pass**

```bash
$JL --project=. -e 'using BiGSTARS, Test, SparseArrays, LinearAlgebra; include("test/test_mpi_prep.jl")'
```

Expected: PASS.

- [ ] **Step 5: Run the FULL suite (no regression)**

```bash
$JL --project=. -e 'using Pkg; Pkg.test()'
```

Expected: PASS, `Test Summary: BiGSTARS.jl | <N> <N>` with zero fails (N grows by the new assertions).

- [ ] **Step 6: Commit**

```bash
git add src/mpi_prep.jl test/test_mpi_prep.jl
git commit -m "feat: relocate sparse_from_csr into src with round-trip test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Confirm the message via `git log --oneline -1`; amend if a hook changed it.

---

## Task 3: De-duplicate the extension — import the relocated helpers

**Files:**
- Modify: `ext/BiGSTARSMPIExt.jl`

> The extension only precompiles when the weakdeps are installed, so this is
> verified here by (a) syntax-parse and (b) the base suite staying green; full
> runtime verification is Task 4/5 on a PETSc machine.

- [ ] **Step 1: Update the import list**

In `ext/BiGSTARSMPIExt.jl`, replace the existing import block:

```julia
using BiGSTARS: _to_csr, _csr_row_block, SolverResults, ConvergenceHistory,
                sort_eigenvalues!, _filter_physical_modes
```

with (adds `_eps_options`, `sparse_from_csr`):

```julia
using BiGSTARS: _to_csr, _csr_row_block, SolverResults, ConvergenceHistory,
                sort_eigenvalues!, _filter_physical_modes,
                _eps_options, sparse_from_csr
```

- [ ] **Step 2: Remove the relocated `_WHICH_OPT` + `_eps_options` from the extension**

In `ext/BiGSTARSMPIExt.jl`, DELETE this whole block (the `const _WHICH_OPT = Dict(...)` through the end of `function _eps_options(...) ... end`), including its leading comment `# Map BiGSTARS \`which\` ...`. It now lives in `src/mpi_prep.jl`. Leave the surrounding `# Task 7` banner and `_solve_one` intact.

- [ ] **Step 3: Remove the relocated `sparse_from_csr` from the extension**

In `ext/BiGSTARSMPIExt.jl`, DELETE the `"""Rebuild a SparseMatrixCSC ..."""` docstring and its `function sparse_from_csr(csr) ... end` definition (in the Task-8 section). `_assemble_results` keeps calling `sparse_from_csr(B_csr)` — it now resolves to the imported `BiGSTARS.sparse_from_csr`.

- [ ] **Step 4: Syntax-parse check**

```bash
$JL -e 'src=read("ext/BiGSTARSMPIExt.jl",String); ex=Meta.parseall(src); println(ex isa Expr && !any(a->a isa Expr && a.head===:error, ex.args) ? "PARSE OK" : "SYNTAX ERROR")'
```

Expected: `PARSE OK`. Also confirm the three names no longer appear as definitions:

```bash
grep -n "const _WHICH_OPT\|^function _eps_options\|^function sparse_from_csr" ext/BiGSTARSMPIExt.jl || echo "none defined locally (good)"
```

Expected: `none defined locally (good)`.

- [ ] **Step 5: Run the FULL base suite (extension still not loaded; must stay green)**

```bash
$JL --project=. -e 'using Pkg; Pkg.test()'
```

Expected: PASS, zero fails.

- [ ] **Step 6: Commit**

```bash
git add ext/BiGSTARSMPIExt.jl
git commit -m "refactor: extension imports relocated pure helpers from BiGSTARS

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Confirm message via `git log --oneline -1`; amend if needed.

---

## Task 4: Harden the integration test + add its CI environment

**Files:**
- Modify: `test/mpi/test_slepc.jl`
- Create: `test/mpi/Project.toml`

> Cannot run here (no PETSc/SLEPc/mpiexec). Verify by syntax-parse only; real
> verification is the GitHub job in Task 5.

- [ ] **Step 1: Rewrite `test/mpi/test_slepc.jl`**

Replace the entire file with:

```julia
# Run with: mpiexec -n {1,2} julia --project=test/mpi test/mpi/test_slepc.jl
# Requires a complex-scalar system PETSc/SLEPc (PETSC_DIR/PETSC_ARCH/SLEPC_DIR)
# and MPI.jl bound to the same MPI via MPIPreferences.use_system_binary().
using BiGSTARS
using MPI, PetscWrap, SlepcWrap
using Test
using LinearAlgebra

SlepcInitialize()
try
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # Small Poisson-type EVP with a known serial answer.
    dom = Domain(x = FourierTransformed(), z = Chebyshev(N=16, lower=0.0, upper=1.0))
    prob = EVP(dom, variables=[:u], eigenvalue=:sigma)
    @equation prob sigma * u == -dx(dx(u)) - dz(dz(u))
    @bc prob left(u) == 0
    @bc prob right(u) == 0
    cache = discretize(prob)

    # Distributed solve (collective across all ranks).
    res = solve_mpi(cache, [1.0]; sigma_0=10.0, nev=1, which=:LM, tol=1e-10)

    # Verify on rank 0 against the serial :Krylov result and the eigenpair residual.
    if rank == 0
        ref = solve(cache, [1.0]; sigma_0=10.0, method=:Krylov, nev=1, verbose=false)
        A, B = BiGSTARS.assemble(cache, 1.0)        # serial matrices for the residual

        ts = @testset "solve_mpi vs serial" begin
            @test res[1].converged
            @test ref[1].converged

            λ_mpi = res[1].eigenvalues[1]
            λ_ser = ref[1].eigenvalues[1]
            @test abs(λ_mpi - λ_ser) < 1e-6         # eigenvalue matches serial

            χ = res[1].eigenvectors[:, 1]           # gathered eigenvector
            resid = norm(A * χ - λ_mpi * (B * χ)) / norm(χ)
            @test resid < 1e-6                      # phase-independent: checks the gather too

            println("MPI λ=$(λ_mpi)  serial λ=$(λ_ser)  residual=$(resid)")
        end

        # Make CI go red on any failure (a bare @test does not set the exit code).
        nfail = ts.anynonpass ? 1 : 0
        nfail == 1 && exit(1)
    end
finally
    SlepcFinalize()
end
```

> Note: `@testset` returns a result object; `.anynonpass` is `true` if any test
> failed or errored. If that field name differs on the installed `Test`, use
> `Test.get_test_counts` or check `ts.n_passed`; confirm during the first CI run.

- [ ] **Step 2: Create `test/mpi/Project.toml`**

```toml
[deps]
BiGSTARS = "55e6fa54-20b2-4a8e-ba84-e72db1c430e6"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
MPIPreferences = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
PetscWrap = "5be22e1c-01b5-4697-96eb-ef9ccdc854b8"
SlepcWrap = "c3679e3b-785e-4ccc-b734-b7685cbb935e"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
```

> Confirm these UUIDs resolve when the env first instantiates on CI; the PetscWrap
> and SlepcWrap values match the registry entries used in the backend work.

- [ ] **Step 3: Syntax-parse the test script**

```bash
JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia
$JL -e 'src=read("test/mpi/test_slepc.jl",String); ex=Meta.parseall(src); println(ex isa Expr && !any(a->a isa Expr && a.head===:error, ex.args) ? "PARSE OK" : "SYNTAX ERROR")'
```

Expected: `PARSE OK`.

- [ ] **Step 4: Commit**

```bash
git add test/mpi/test_slepc.jl test/mpi/Project.toml
git commit -m "test: harden MPI integration test (exit code + residual) and add its env

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Confirm message via `git log --oneline -1`; amend if needed.

---

## Task 5: Rewrite the MPI/SLEPc CI workflow (cached complex source build)

**Files:**
- Modify: `.github/workflows/mpi.yml`

> Verified only by GitHub running the job. No untrusted input is interpolated
> into `run:` steps (no issue/PR/commit data), so there is no injection surface.

- [ ] **Step 1: Replace `.github/workflows/mpi.yml` entirely**

```yaml
name: MPI/SLEPc
on:
  push:
    branches: [main]
    paths:
      - 'ext/**'
      - 'src/mpi_prep.jl'
      - 'test/mpi/**'
      - 'Project.toml'
      - '.github/workflows/mpi.yml'
  pull_request:
    paths:
      - 'ext/**'
      - 'src/mpi_prep.jl'
      - 'test/mpi/**'
      - 'Project.toml'
      - '.github/workflows/mpi.yml'
  workflow_dispatch:

env:
  PETSC_VERSION: "3.22.2"
  SLEPC_VERSION: "3.22.1"
  PETSC_ARCH: "arch-linux-c-complex"

jobs:
  mpi-slepc:
    runs-on: ubuntu-latest
    continue-on-error: true        # non-blocking until a green run confirms the wrapper API
    steps:
      - uses: actions/checkout@v4

      - name: Install build tools and OpenMPI
        run: |
          sudo apt-get update
          sudo apt-get install -y build-essential gfortran python3 \
            openmpi-bin libopenmpi-dev libblas-dev liblapack-dev

      - name: Cache PETSc + SLEPc
        id: cache-petsc
        uses: actions/cache@v4
        with:
          path: |
            ${{ github.workspace }}/petsc
            ${{ github.workspace }}/slepc
          key: petsc-${{ env.PETSC_VERSION }}-slepc-${{ env.SLEPC_VERSION }}-${{ env.PETSC_ARCH }}

      - name: Build PETSc (complex) and SLEPc
        if: steps.cache-petsc.outputs.cache-hit != 'true'
        run: |
          export PETSC_DIR=${{ github.workspace }}/petsc
          export SLEPC_DIR=${{ github.workspace }}/slepc
          export PETSC_ARCH=${{ env.PETSC_ARCH }}
          # PETSc
          curl -L https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-${PETSC_VERSION}.tar.gz | tar xz
          mv petsc-${PETSC_VERSION} "$PETSC_DIR"
          cd "$PETSC_DIR"
          ./configure PETSC_ARCH=$PETSC_ARCH \
            --with-scalar-type=complex \
            --with-fortran-bindings=0 \
            --with-debugging=0 \
            --download-mumps --download-scalapack \
            --download-metis --download-parmetis
          make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all
          # SLEPc
          cd "${{ github.workspace }}"
          curl -L https://slepc.upv.es/download/distrib/slepc-${SLEPC_VERSION}.tar.gz | tar xz
          mv slepc-${SLEPC_VERSION} "$SLEPC_DIR"
          cd "$SLEPC_DIR"
          ./configure
          make SLEPC_DIR=$SLEPC_DIR PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH

      - name: Export PETSc/SLEPc env
        run: |
          echo "PETSC_DIR=${{ github.workspace }}/petsc" >> $GITHUB_ENV
          echo "SLEPC_DIR=${{ github.workspace }}/slepc" >> $GITHUB_ENV
          echo "PETSC_ARCH=${{ env.PETSC_ARCH }}" >> $GITHUB_ENV

      - uses: julia-actions/setup-julia@v2
        with:
          version: '1.10'

      - name: Configure MPI.jl to use the system OpenMPI
        run: |
          julia --project=test/mpi -e 'using Pkg; Pkg.develop(path=".")'
          julia --project=test/mpi -e 'using Pkg; Pkg.add("MPIPreferences")'
          julia --project=test/mpi -e 'using MPIPreferences; MPIPreferences.use_system_binary()'

      - name: Instantiate the MPI test environment
        run: julia --project=test/mpi -e 'using Pkg; Pkg.instantiate()'

      - name: Run integration test (n=1)
        run: mpiexec -n 1 julia --project=test/mpi test/mpi/test_slepc.jl

      - name: Run integration test (n=2)
        run: mpiexec --oversubscribe -n 2 julia --project=test/mpi test/mpi/test_slepc.jl
```

> Notes for the first runs (expected to iterate):
> - The `continue-on-error: true` keeps the job non-blocking. Once it goes green,
>   remove that line in a follow-up so it becomes a required check.
> - If `MatCreateVecs(A, vr, C_NULL)` errors, that is the predicted `CONFIRM`
>   point — switch `_gather_eigenpairs` to the wrapper's returning form
>   (e.g. `vr = MatCreateVecs(A)`), per SlepcWrap's API.
> - If MUMPS configure fails, add `--download-ptscotch` or drop to
>   `--download-superlu_dist` and set `mat_solver=:superlu_dist` in the test call.
> - `--oversubscribe` lets `-n 2` run on a 2-core runner; drop it if the runner
>   has ≥2 slots.

- [ ] **Step 2: Validate the workflow YAML**

```bash
python3 -c 'import yaml,sys; yaml.safe_load(open(".github/workflows/mpi.yml")); print("YAML OK")'
```

Expected: `YAML OK`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/mpi.yml
git commit -m "ci: source-build cached complex PETSc/SLEPc and run solve_mpi at n=1,2

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Confirm message via `git log --oneline -1`; amend if needed.

---

## Self-review notes

- **Spec coverage:** Tier-1 relocation (Tasks 1–3), Tier-1 unit tests (Tasks 1–2),
  Tier-2 test hardening incl. residual + exit code (Task 4), dedicated CI env
  (Task 4), Tier-2 workflow with cached source build + MPIPreferences + n=1/n=2 +
  path triggers + continue-on-error (Task 5). All spec sections map to tasks.
- **Refinement vs spec:** the spec said "Pkg.add MPI/PetscWrap/SlepcWrap"; the plan
  realizes this as a dedicated `test/mpi/Project.toml` env to avoid clashing with
  the `[weakdeps]` entries in the base `Project.toml`. Functionally equivalent,
  cleaner.
- **Type/name consistency:** `_eps_options`/`_WHICH_OPT`/`sparse_from_csr` keep the
  exact signatures they had in the extension, so Task 3's removal + import leaves
  `_solve_one` and `_assemble_results` callers unchanged. The test env UUIDs match
  the registry values confirmed during the backend work.
- **Known confirmations during execution:** `ts.anynonpass` field name (Task 4);
  the `MatCreateVecs` form and MUMPS configure flags (Task 5) — each flagged inline.
