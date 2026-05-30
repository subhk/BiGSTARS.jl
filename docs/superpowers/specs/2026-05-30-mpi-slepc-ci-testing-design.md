# MPI/SLEPc Backend CI Testing — Design

**Date:** 2026-05-30
**Status:** Approved design, pending implementation plan
**Author:** Subhajit Kar (with Claude)
**Builds on:** [PETSc/SLEPc MPI eigensolver backend](2026-05-30-petsc-slepc-mpi-eigensolver-backend-design.md)

## Summary

Add automated GitHub CI testing for the distributed `solve_mpi` backend in two
tiers: (1) always-on pure-Julia unit tests that need no PETSc, by relocating the
extension's pure logic into `src/`; and (2) a real integration job that
source-builds a complex-scalar PETSc/SLEPc (cached), then runs `solve_mpi`
end-to-end across MPI ranks and checks it against the serial result and the
eigenpair residual.

## Motivation

The 295-line `ext/BiGSTARSMPIExt.jl` was merged to `main` without ever executing
(no system PETSc/SLEPc in the dev environment). The base suite passes but does
not exercise the distributed solver at all. We need CI that (a) always verifies
the parts that can be tested without PETSc, and (b) actually runs the full
distributed path on a real complex PETSc/SLEPc so the solver glue is verified and
the flagged `CONFIRM` API points are resolved.

## Tier 1 — always-on unit tests (no PETSc)

### Refactor: relocate pure logic into `src/`

Extension code only loads when `MPI`/`PetscWrap`/`SlepcWrap` are imported, so it
cannot be tested on a PETSc-free CI runner. Move the pure-Julia pieces out of
`ext/BiGSTARSMPIExt.jl` into `src/mpi_prep.jl`, and have the extension import them
from `BiGSTARS`:

- `_eps_options(; sigma_0, nev, which, tol, maxiter, ncv, mat_solver, eps_type)` —
  builds the SLEPc options-database string.
- `_WHICH_OPT` — the `Symbol`→SLEPc which-string mapping.
- `sparse_from_csr(csr)` — rebuilds a `SparseMatrixCSC` from a CSR tuple (inverse
  of `_to_csr`).

This shrinks the extension to only PETSc-touching glue and widens the
CI-verified surface. The extension references them as `_eps_options` etc. via its
existing `using BiGSTARS: ...` import list.

### New unit tests (`test/test_mpi_prep.jl`, runs in the normal suite)

- `_eps_options`: for `which=:LM, nev=5, ncv=0, mat_solver=:mumps` the string
  contains `-eps_nev 5`, `-eps_target`, `-eps_target_magnitude`, `-st_type sinvert`,
  `-st_pc_factor_mat_solver_type mumps`, and does NOT contain `-eps_ncv`.
- `_eps_options` with `ncv=20` contains `-eps_ncv 20`.
- `_eps_options` with an unsupported `which` (e.g. `:XX`) throws `ArgumentError`.
- `_WHICH_OPT[:LM] == "target_magnitude"`, `[:SR] == "smallest_real"`.
- `sparse_from_csr(_to_csr(A)...) ≈ A` for a random sparse `ComplexF64` `A`.

These need no PETSc and run on the existing CI matrix — always green.

## Tier 2 — real integration job (cached source build)

### Workflow (`.github/workflows/mpi.yml`, rewritten)

Steps:
1. `actions/checkout`.
2. apt: `openmpi-bin libopenmpi-dev gfortran` + build tools.
3. `actions/cache` on the PETSc and SLEPc install directories, keyed by their
   pinned versions + arch (cache hit skips the build).
4. On cache miss: download the pinned PETSc release, configure with
   `--with-scalar-type=complex --download-mumps --download-scalapack
   --download-metis --download-parmetis --with-fortran-bindings=0`, `make`; then
   download and build SLEPc against it. Export `PETSC_DIR`, `PETSC_ARCH`,
   `SLEPC_DIR` to `$GITHUB_ENV`.
5. `julia-actions/setup-julia` (1.10).
6. `MPIPreferences.use_system_binary()` to bind `MPI.jl` to the same system
   OpenMPI PETSc was built with (mismatched MPI causes silent hangs/wrong comms).
7. `Pkg.add(["MPI", "PetscWrap", "SlepcWrap"])`, `Pkg.instantiate()`.
8. Run the integration test at `-n 1` and `-n 2`:
   `mpiexec -n 1 julia --project=. test/mpi/test_slepc.jl` and `-n 2`.

### Triggers

`push` and `pull_request` filtered to MPI-relevant paths
(`ext/**`, `test/mpi/**`, `src/mpi_prep.jl`, `Project.toml`,
`.github/workflows/mpi.yml`) plus `workflow_dispatch`. The path filter avoids
burning CI on unrelated commits; the cache keeps repeat runs fast.

### Initial non-blocking status

Tier 2 has never run; the first runs will likely surface the `CONFIRM` API points
(most notably `MatCreateVecs(A, vr, C_NULL)` in `_gather_eigenpairs`, which may
need to *return* the vecs rather than fill pre-created ones; also `MatCreate`
comm form, `PetscOptionsInsertString`, `VecGetArray` element type, `PetscScalar`).
The job lands with `continue-on-error: true` (red but non-blocking) until a green
run confirms the API, then the flag is removed to make it a required check.

## Tier 2 test hardening (`test/mpi/test_slepc.jl`)

The current script cannot fail CI (a bare `@test` on rank 0 does not set the
process exit code). Harden it:

- Wrap the rank-0 assertions in an explicit `@testset`; capture the result and
  `exit(failures > 0 ? 1 : 0)` so CI turns red on failure. Other ranks exit 0.
- Add a **residual check**: with the gathered eigenvector `χ` and the serially
  assembled `A`, `B` on rank 0, assert `‖A χ − λ B χ‖ / ‖χ‖ < tol`. This is
  phase-independent and verifies the eigenvector gather, not just the eigenvalue.
- Keep the serial `:Krylov` eigenvalue cross-check (`|λ_mpi − λ_serial| < tol`).
- Guard `SlepcInitialize` / `SlepcFinalize` so finalize always runs.

## Architecture / file changes

| File | Change |
|---|---|
| `src/mpi_prep.jl` | Add `_eps_options`, `_WHICH_OPT`, `sparse_from_csr`. |
| `ext/BiGSTARSMPIExt.jl` | Remove those three definitions; import them from `BiGSTARS`. |
| `test/test_mpi_prep.jl` | Add Tier 1 unit tests. |
| `test/mpi/test_slepc.jl` | Harden: testset + nonzero exit, residual check. |
| `.github/workflows/mpi.yml` | Rewrite: cached complex source build, MPIPreferences, n=1/n=2 run, path triggers, continue-on-error. |

## Scope cuts (YAGNI)

- No conda-forge or Docker provisioning (source build chosen).
- No GPU / multi-version PETSc matrix — one pinned complex build.
- No change to the solver design itself; this is testing/CI only. Any solver-code
  fix needed to make Tier 2 pass is a bug fix surfaced by the new test, not new
  feature work.

## Open risks

- **Build time / cache:** first run is a ~20–40 min PETSc+SLEPc build; caching
  amortizes it. Cache key must include exact versions so a bump rebuilds.
- **Wrapper API:** Tier 2 is expected to need 1–2 debug iterations against real
  CI output before it goes green; that is the point of the job.
- **MPI matching:** `MPIPreferences.use_system_binary()` must pick the OpenMPI
  used for the PETSc build; otherwise ranks misbehave.
