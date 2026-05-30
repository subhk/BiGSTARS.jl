# PETSc/SLEPc MPI Eigensolver Backend — Design

**Date:** 2026-05-30
**Status:** Approved design, pending implementation plan
**Author:** Subhajit Kar (with Claude)

## Summary

Add a distributed-memory (MPI) eigensolver backend to BiGSTARS built on
SLEPc's `EPS` over PETSc distributed matrices. The existing in-process
backends (`:Arnoldi`, `:Arpack`, `:Krylov`) and the threaded `solve`
wavenumber sweep are left untouched. The new path is reached through a
separate entrypoint, `solve_mpi`, and lives entirely in a Julia package
extension so the base package gains no PETSc/SLEPc/MPI weight.

## Motivation

Some target problems are too large for a single in-process eigensolve. SLEPc
provides a battle-tested, MPI-distributed Krylov-Schur eigensolver with
shift-and-invert spectral transform and parallel direct factorization
(MUMPS / SuperLU_DIST). The goal is to distribute the expensive work — the
shift-and-invert factorization and the Krylov eigensolve — across MPI ranks,
while keeping matrix assembly serial.

## Locked decisions

| Decision | Choice |
|---|---|
| Scale | Distributed MPI; one eigenproblem per wavenumber spread across all ranks |
| Assembly | Rank-0 serial assembly via the existing `assemble` pipeline (untouched) |
| Distribution | Rank 0 → MPI scatter of CSR row-blocks → `MatMPIAIJ` |
| Eigensolve | SLEPc `EPS` Krylov-Schur + `ST`=sinvert + parallel `KSP`/`PC` (MUMPS) |
| Invocation | Separate `solve_mpi` entrypoint; threaded `solve` is not modified |
| Packaging | Julia package extension on weakdeps `MPI`, `PetscWrap`, `SlepcWrap` |
| Hard constraint | System PETSc/SLEPc built `--with-scalar-type=complex` |

### Why these

- **MPI boundary at the solve, not assembly.** The user wants serial assembly
  with a parallel eigensolver. Rewriting `discretize`/`assemble` to be MPI-aware
  is a large, separate effort and is explicitly out of scope. Peak memory remains
  bounded by rank 0's full matrix — the accepted tradeoff.
- **Separate entrypoint.** Under MPI the whole script runs once per rank and
  SLEPc calls are collective. Overloading the threaded `solve` would force
  disabling its k-sweep threading and make it collective, leaking MPI semantics
  into the serial API. A dedicated `solve_mpi` keeps the two models clean.
- **Package extension.** `SlepcWrap`/`PetscWrap`/`MPI` pull in native binaries
  and a system PETSc/SLEPc build. As weakdeps behind an extension, base BiGSTARS
  users inherit none of that weight or version constraints.
- **SlepcWrap/PetscWrap wrapper.** PETSc.jl (JuliaParallel) wraps PETSc but has
  no eigensolver; eigensolvers live in SLEPc. The only maintained Julia path to
  SLEPc's `EPS` is `SlepcWrap.jl` (+ `PetscWrap.jl`, by bmxam). There is no
  `SLEPc_jll` in Yggdrasil, so an auto-binary thin wrapper is not available
  without building a JLL ourselves (out of scope). SlepcWrap uses a system build
  via `SLEPC_DIR`/`PETSC_ARCH`; its maturity risk is accepted.

## Architecture

### Module layout & dependency wiring

- `Project.toml`:
  - `[weakdeps]`: `MPI`, `PetscWrap`, `SlepcWrap`.
  - `[compat]`: matching bounds for the three weakdeps.
  - `[extensions]`: `BiGSTARSMPIExt = ["MPI", "PetscWrap", "SlepcWrap"]`.
  - Base `[deps]` unchanged.
- `ext/BiGSTARSMPIExt.jl`: all PETSc/SLEPc/MPI code. Loads only when the user
  does `using MPI, PetscWrap, SlepcWrap`. Nothing in `src/` imports them.
- `src/eig_solver.jl`: declare a stub `function solve_mpi end` and document the
  `:Slepc` method. The name is exported from the base package and, until the
  extension loads, throws an install-hint error.
- Export `solve_mpi` from `src/BiGSTARS.jl`.

### Public API

```julia
using BiGSTARS, MPI, PetscWrap, SlepcWrap   # extension activates

SlepcInitialize()                            # or a BiGSTARS.mpi_init() helper
cache = discretize(prob)                     # runs on all ranks; rank-0 result used
results = solve_mpi(cache, k_values;
                    sigma_0=0.02, nev=5, which=:LM,
                    tol=1e-10, maxiter=300,
                    ncv=0,                    # SLEPc subspace; 0 = SLEPc default
                    mat_solver=:mumps)        # parallel direct solver for sinvert
SlepcFinalize()
```

- Returns `Vector{SolverResults}`, one per wavenumber, **fully populated only on
  rank 0**. Other ranks receive a lightweight marker result (`converged=false`,
  empty arrays) because eigenvectors are gathered to rank 0. `method_used=:Slepc`.
- `solve_mpi(cache; sigma_0, ...)` single-problem overload mirrors the existing
  `solve(cache; ...)` no-wavenumber overload.

## Data flow (per wavenumber)

1. **Assemble** — `assemble(cache, k)` on rank 0 →
   `A, B :: SparseMatrixCSC{ComplexF64}`. Other ranks skip or discard.
2. **CSR convert** — rank 0 computes `sparse(transpose(A))`, whose CSC storage is
   the CSR of `A` (contiguous row slices). Same for `B`.
3. **Partition** — compute PETSc default row ownership `[rstart_p, rend_p)` per
   rank `p` (≈ N/P).
4. **Scatter** — rank 0 `MPI.Send`s each rank its CSR row-block (`rowptr`,
   `colind`, `vals`); rank `p` receives its slice.
5. **Build** — each rank fills its local rows of a `MatMPIAIJ` for `A` and `B`
   (`MatCreateMPIAIJWithArrays` / `MatSetValues`), then `MatAssemblyBegin/End`.
6. **Solve** — `EPS`: `EPSSetOperators(A, B)`, problem type `GNHEP`
   (non-Hermitian default), `ST`=`sinvert`, `STSetShift(σ₀)`,
   `EPSSetTarget(σ₀)`, `EPS_TARGET_MAGNITUDE`, `KSP`=`preonly`, `PC`=`lu` with
   `mat_solver`. `EPSSolve()`.
7. **Gather** — for each converged pair, `EPSGetEigenpair` gives the eigenvalue
   (scalar, replicated) and a distributed eigenvector `Vec`;
   `VecScatterCreateToZero` gathers it into a full `ComplexF64` column on rank 0.
8. **Assemble result** — rank 0 builds `SolverResults`, reusing the existing
   `sort_eigenvalues!` and `_filter_physical_modes`.

## Config / option mapping

Reuse `SolverConfig` fields where they map; SLEPc-specific knobs are `solve_mpi`
kwargs, defaulted inside the extension so the base package stays clean.

| BiGSTARS | SLEPc |
|---|---|
| `σ₀` / `sigma_0` | `EPSSetTarget` + `STSetShift` |
| `which=:LM` | `EPS_TARGET_MAGNITUDE` (nearest the shift — matches the current sinvert convention) |
| `:LR` / `:SR` / `:LI` | `EPS_LARGEST_REAL` / `EPS_SMALLEST_REAL` / `EPS_LARGEST_IMAGINARY` |
| `nev` | `EPSSetDimensions(nev, ncv, mpd)` |
| `tol`, `maxiter` | `EPSSetTolerances` |
| `sortby` | post-solve `sort_eigenvalues!` on rank 0 (unchanged) |

New kwargs: `ncv` / `mpd` (subspace sizes), `mat_solver`
(`:mumps` / `:superlu_dist` / `:petsc`), `eps_type` (default `:krylovschur`).

## Error handling

- **Extension not loaded** (no SlepcWrap) — the `solve_mpi` stub throws:
  *"solve_mpi needs MPI + PetscWrap + SlepcWrap installed, plus a complex-scalar
  system PETSc/SLEPc build."*
- **Real-scalar build** — query the PETSc scalar type at init; if real, throw a
  clear error before solving (otherwise results are silently wrong).
- **Non-convergence** — if `EPSGetConverged` < `nev`, return what converged
  (`converged = nconv ≥ 1`), matching the existing lenient behavior; `nconv = 0`
  yields a failed `SolverResults`.
- **Collective safety** — every rank calls `solve_mpi` collectively; failures are
  broadcast from rank 0 so no rank hangs.

## Testing & CI

- **Pure-Julia unit tests** (normal CI, no MPI): exercise the custom risky logic
  — CSR-transpose conversion, row-ownership partition, and a row-block extraction
  round-trip (reassemble the scattered blocks and compare to the original `A`).
  No PETSc required.
- **Opt-in MPI integration job** (separate workflow / env flag): install a
  complex-scalar system PETSc+SLEPc, run `mpiexec -n 2 julia test/mpi/test_slepc.jl`
  on a small problem, and assert the eigenvalues match the serial `:Krylov`
  result within tolerance. Skipped by default so base CI stays green without PETSc.
- Document local run instructions and required env vars (`SLEPC_DIR`,
  `PETSC_DIR`, `PETSC_ARCH`).

## Scope cuts (YAGNI)

Excluded from v1:

- Distributed assembly (assembly stays serial on rank 0).
- Adaptive-σ retry sweep (the existing `n_tries` / `ϵ` logic). v1 is a single
  SLEPc solve at `σ₀`; sinvert is robust. Deferred as future work.
- GPU backends (CUDA/HIP).
- Distributed eigenvector reconstruction — eigenvectors are gathered to rank 0.
- MPI parallelism across wavenumbers — each k uses all ranks; the k-loop is
  sequential.

## Open implementation risks

- **SlepcWrap maturity.** Low recent activity; API and Julia 1.10+ compatibility
  must be verified early in implementation. If it proves unusable, fall back to a
  direct `ccall` wrapper (higher effort) — but do not block the design on this.
- **CSR row-block scatter** is the main piece of custom, error-prone code;
  covered by the round-trip unit test above.
