# Design: SLEPc/PETSc as the sole eigensolver

**Date:** 2026-06-04
**Status:** Approved (pending spec review)
**Branch target:** single PR — addition + deletion land together

## Goal

Remove the three in-process eigensolvers (`KrylovKit`, `ArnoldiMethod`, `Arpack`)
and make SLEPc-over-PETSc the only eigensolver. The distributed path becomes the
package's single solve entrypoint, renamed from `solve_mpi` to plain **`solve`**.
The adaptive shift (σ) retry loop from the old serial solver is ported onto the
SLEPc path so convergence robustness is preserved.

This is a deliberately breaking overhaul. `MPI`, `PetscWrap`, `SlepcWrap` become
hard dependencies; there is no pure-Julia fallback.

## Decisions (locked with the user)

1. **Hard-require SLEPc**, no fallback. `MPI`/`PetscWrap`/`SlepcWrap` → `[deps]`.
2. **`solve` is the only entrypoint.** Delete the serial `solve`; rename
   `solve_mpi` → `solve`. `solve_mpi` is removed (no alias).
3. **Port adaptive-σ retry** onto the SLEPc path.
4. **Split CI** — non-solver tests run cross-platform; solver tests run in a
   blocking complex-PETSc Linux job.
5. **Single PR** — deletion of serial code and the SLEPc rework land together.

## Verified feasibility (the two linchpins)

- **`EPSSetTarget(eps, target::PetscScalar)` and `EPSSetWhichEigenpairs` are
  wrapped** in SlepcWrap. The σ target is therefore set programmatically per
  attempt. The remaining static options (`eps_type`, `nev`, `tol`, `max_it`,
  `gen_non_hermitian`, the `sinvert` ST, `pc lu`, `mat_solver_type`) keep going
  through the PETSc options database at `SlepcInitialize` time, because their
  setters (`EPSSetDimensions`, `EPSSetTolerances`, `STSetType`/`STSetShift`) are
  **not** wrapped. With `-st_type sinvert`, SLEPc uses the EPS target as the
  shift automatically, so the missing `STSetShift` is not needed.
- **PetscWrap uses `PETSc_jll` by default** (system build only when `PETSC_DIR`/
  `PETSC_ARCH` are set). So `using BiGSTARS` works on macOS/Windows/laptops with
  no system build. The default JLL is **real-scalar**, so an actual *solve*
  throws `_assert_complex_scalars` unless a complex PETSc build is present. That
  split is exactly what keeps the cross-platform CI matrix viable: it loads the
  package and runs every non-solver test on the JLL; solves run only in the
  Linux complex-PETSc job.

## Architecture overview

```
discretize (pure Julia)
        │
        ▼
   solve(cache, k_values; sigma_0, …)        ← single entrypoint (was solve_mpi)
        │  rank 0 assembles A,B per wavenumber (existing serial pipeline)
        │  → CSR → scatter row-blocks to ranks → distributed MatMPIAIJ
        ▼
   adaptive-σ loop  (ported from old solve!)
        │  per attempt: EPSSetTarget(σ) → EPSSolve → gather → rank-0 decide → bcast
        ▼
   Vector{SolverResults}   (rank 0 populated; other ranks empty markers)
        │
        ▼
   reconstruct / utils  (unchanged — SolverResults shape preserved)
```

## A. Dependencies & module structure

**`Project.toml`**
- Move `MPI`, `PetscWrap`, `SlepcWrap` from `[weakdeps]` → `[deps]`.
- Delete `[weakdeps]` and `[extensions]` blocks.
- Remove `ArnoldiMethod`, `Arpack`, `KrylovKit`, `LinearMaps`, `VectorInterface`
  from `[deps]`.
- Update `[compat]` accordingly (drop removed packages; keep MPI/PetscWrap/
  SlepcWrap entries that were under weakdeps).

**Source files**
- Move `ext/BiGSTARSMPIExt.jl` → `src/slepc_solve.jl` as plain module-internal
  code: `using MPI, PetscWrap, SlepcWrap`; methods defined directly as
  `BiGSTARS.solve` (no `BiGSTARS.` qualifier needed once in-module). Delete the
  `ext/` directory.
- Rename `src/eig_solver.jl` → `src/results.jl`, keeping only the shared result
  types/helpers (see §B).
- `src/BiGSTARS.jl`:
  - Drop `using ArnoldiMethod`, `using Arpack`, `using KrylovKit`,
    `using LinearMaps`, `using VectorInterface`.
  - Add `using MPI, PetscWrap, SlepcWrap`.
  - Replace `include("eig_solver.jl")` with `include("results.jl")`;
    `include("slepc_solve.jl")`; remove `include("solve.jl")` and
    `include("construct_linear_map.jl")`.
  - Delete the `solve_mpi` stub + `solve_mpi(@nospecialize …)` error fallback.
  - Update `@setup_workload` (see §D).
  - Fix `export` list (see §B).

## B. Deleted vs kept

**Delete**
- `src/solve.jl` (serial wavenumber driver: `solve`, `_solve_inplace`,
  `_solve_sparse`, `_assembled_density`).
- `src/construct_linear_map.jl` (in-process shift-and-invert `ShiftAndInvert`,
  `construct_linear_map`).
- From the old `eig_solver.jl`: `EigenSolver`, `SolverConfig`, `solve!`,
  `solve_arnoldi_single`, `solve_arpack_single`, `solve_krylov_single`,
  `compare_methods!`, `get_method_info`, `show_example_usage`,
  `solve_eigenvalue_problem`, `solve_arnoldi`, `solve_arpack`, `solve_krylov`,
  `get_results`, `wrapvec`, `unwrapvec`, `_factorize_shifted`.

**Keep** (moved into `src/results.jl`)
- `SolverResults`, `ConvergenceHistory` — consumed by the SLEPc path and by
  `reconstruct.jl`/`utils.jl`.
- `sort_eigenvalues!`, `_filter_physical_modes` — used by the SLEPc assembly.
- `print_summary` — **adapt** to take a `SolverResults` instead of an
  `EigenSolver`.

**Export list after the change**
- Remove: `EigenSolver`, `solve!`, `SolverConfig`, `compare_methods!`,
  `get_results`, `solve_mpi`.
- Keep/add: `solve`, `SolverResults`, `ConvergenceHistory`, `print_summary`.

**Audit task:** grep `reconstruct.jl`, `utils.jl`, and any docs/examples for the
deleted symbols (`EigenSolver`, `get_results`, `solve!`, `method=`,
`compare_methods!`) and update. They consume `SolverResults` (kept), so the
expected change is small, but it must be verified.

## C. The new `solve` and adaptive-σ on SLEPc

**Signature**

```julia
solve(cache::DiscretizationCache, k_values::AbstractVector;
      sigma_0::Real,
      nev::Integer=1, which::Symbol=:LM,
      tol::Real=1e-10, maxiter::Integer=300, ncv::Integer=0,
      mat_solver::Symbol=:mumps, eps_type::Symbol=:krylovschur,
      n_tries::Integer=8, Δσ₀::Real=0.2, incre::Real=1.2, ϵ::Real=1e-5,
      manage_init::Bool=true, verbose::Bool=false) -> Vector{SolverResults}

solve(cache::DiscretizationCache; sigma_0::Real, kwargs...) =
    solve(cache, [0.0]; sigma_0=sigma_0, kwargs...)
```

Dropped kwargs vs the old serial `solve`: `method`, `parallel`, `sparse` (all
serial-only concepts — one backend now, parallelism is intrinsic, PETSc storage
is always sparse AIJ).

**Static options string** (`_eps_options`, in `mpi_prep.jl`)
- Keep everything except the fixed `-eps_target $(sigma_0)` — the target is now
  set per attempt via `EPSSetTarget`. The `which` flag (`-eps_target_magnitude`,
  etc.) stays in the string.
- Consequence: with the numeric target out of the options string, the static
  options are invariant across σ attempts *and* across wavenumbers. The existing
  "options enter the database once per process, error on mismatch" guard
  (`_SLEPC_OPTS`) is therefore always satisfied within a run — the per-σ mismatch
  problem the old single-shot code worried about disappears.

**Adaptive-σ loop** (ported from the old `solve!`, executed collectively)
- σ schedule identical to the serial code:
  `Δσs_up = [Δσ₀ * incre^(i-1) * |σ₀| for i in 1:n_tries]`, then
  `σ_attempts = [σ₀; σ₀ .+ Δσs_up; σ₀ .- Δσs_up]`.
- Per wavenumber `k`:
  - Build the distributed PETSc `A`, `B` **once** (reuse across σ attempts).
  - For each σ in the schedule:
    1. Create a fresh `EPS` (`EPSCreate` + `EPSSetOperators(eps, A, B)`), then
       `EPSSetTarget(eps, PetscScalar(σ))`. Recreating the EPS per attempt is
       cheap relative to the factorization.
    2. `EPSSetFromOptions(eps)`; `EPSSolve(eps)`.
    3. `nconv = EPSGetConverged(eps)` (replicated on every rank).
    4. Gather eigenpairs to rank 0 (existing `_gather_eigenpairs`).
    5. **Rank 0** applies `_filter_physical_modes`, sorts `:nearest` to σ,
       computes the stop decision (a converged attempt whose `|λ₁ - λ_prev| < ϵ`,
       matching serial semantics, with a last-success fallback).
    6. **`MPI.bcast`** the stop flag (and accepted σ) from rank 0 to all ranks;
       every rank branches on the broadcast value.
  - Destroy `A`, `B` after the wavenumber's attempts finish.

**Collective-safety rule (critical):** loop control must be identical on all
ranks or `EPSSolve`/`MatDestroy` calls desync and MPI deadlocks. The σ schedule
is deterministic from kwargs; the stop decision is computed on rank 0 and
**broadcast**. Nothing branches on rank-local data.

**Records:** populate `ConvergenceHistory` (attempts, converged flags, per-attempt
λ₁, final shift) so `SolverResults` carries the same diagnostic detail the serial
path did.

**Fallback:** if a future PetscWrap build lacks `EPSSetTarget`, degrade to the
current single-σ behavior. `EPSSetTarget` is confirmed wrapped, so this is a
defensive note only.

## D. Precompile workload

`@setup_workload`/`@compile_workload` in `src/BiGSTARS.jl` currently runs a full
`solve(...)`. SLEPc cannot be initialized inside precompilation. Reduce the
workload to the pure-Julia pipeline only: `Domain` → `EVP` → `@equation`/`@bc` →
`discretize` → `assemble` (no eigensolve). TTFX for the solve itself is no longer
precompiled; that is the accepted cost.

## E. Tests & correctness cross-check

Removing the serial solvers removes the reference that validated SLEPc. The
solver test suite must therefore pin correctness against **analytic** results.

**Cross-platform `test/runtests.jl`** (runs on the JLL, no solves)
- Keep: `test_ultraspherical`, `test_fourier_coeff`, `test_mpi_prep`,
  `test_transforms`, `test_domain`, `test_expr`, `test_evp`, `test_macros`,
  `test_substitutions`, `test_lowering`, `test_k_separation`, `test_boundary`,
  `test_discretize`.
- Remove from this list: `test_eig_solver` (solver), and the solver-dependent
  parts of `test_integration` and `test_coverage_gaps`. Audit both files: keep
  any pure-Julia (discretize/DSL/assembly) assertions cross-platform; move the
  parts that call `solve`/eigensolve into the MPI suite.

**Linux complex-PETSc suite `test/mpi/`** (under `mpiexec -n 1` and `-n 2`)
- Expand `test/mpi/test_slepc.jl`:
  - **Analytic cross-check**: a problem with known spectrum (e.g. 1D Laplacian
    with Dirichlet BCs, eigenvalues n²π²; or a separable Poisson eigenproblem).
    Assert the computed λ match to tolerance. This is the standalone correctness
    pin that replaces the serial reference.
  - `nev > 1` returns the expected number of modes, sorted nearest σ.
  - `which` variants (`:LM`/`:LR`/`:SR`) select the documented eigenvalues.
  - Singular-`B` spurious-mode filter drops the infinite modes.
  - **Adaptive-σ**: a case that fails to converge at `σ₀` but converges after
    retry; assert the history shows multiple attempts and the final result is
    correct.
  - **Rank consistency**: `-n 1` and `-n 2` produce the same eigenvalues.
- Port the surviving assertions from the old `test_eig_solver.jl` here, rewritten
  against `solve`.

## F. CI

- **`CI.yml`** (cross-platform matrix, unchanged shape): `buildpkg` installs
  `MPI`/`PetscWrap`/`SlepcWrap` via JLLs; `runtests.jl` is the reduced
  non-solver suite. Stays green on macOS/Windows/x64/aarch64/arm64.
- **`mpi.yml`**: remove `continue-on-error: true` so the job is **blocking**.
  It already builds complex PETSc (3.22.2) + SLEPc (3.22.1) with MUMPS and runs
  the `test/mpi/` suite at `-n 1` and `-n 2`. It is now the solver gate.

## G. Examples & docs

- Convert `examples/Eady.jl`, `examples/Stone1971.jl`, `examples/rRBC.jl` to the
  `solve` (SLEPc) + MPI boilerplate pattern shown in `examples/eady_mpi.jl`
  (which itself just changes `solve_mpi` → `solve`). Add `MPI`, `PetscWrap`,
  `SlepcWrap` to `examples/Project.toml`.
- Docs:
  - Fold `docs/src/mpi.md` content into the main solve docs; drop the
    "Experimental" warning (this is now THE path). Rename references `solve_mpi`
    → `solve`.
  - Rewrite solver references in `index.md`, `method.md`, `performance.md`,
    `equation_dsl.md`, and the literated `Eady.md`/`Stone1971.md`/`rRBC.md`.
  - Remove `compare_methods!` / `method=` / multi-backend documentation.
  - Add the complex-PETSc build requirement to `installation_instructions.md`,
    and state clearly that `]add BiGSTARS` works everywhere but solving needs a
    complex-scalar PETSc/SLEPc.

## H. Risks & open items

- **`mpi.yml` has never gone green.** Making it blocking in the same PR means the
  PR is not done until that job passes. This is the real work item, not the
  deletions. Expect to debug the PetscWrap/SlepcWrap call surface against the
  pinned versions during implementation.
- **Complex JLL (nice-to-have):** probe whether `PETSc_jll` exposes a
  complex-scalar library that PetscWrap can select, which would let laptop users
  (and possibly CI) solve without a system build. Not required for this PR; note
  findings.
- **`reconstruct.jl`/`utils.jl` coupling:** confirmed low-risk (they use the
  retained `SolverResults`), but must be audited.

## Implementation note — pivot from hard-dep to extension (2026-06-04)

During implementation the "hard-require PETSc" decision proved unworkable and was
reversed (with the user) to **keep SLEPc as a loadable extension**:

- **Why:** the verified facts about PetscWrap turned out to be wrong. PetscWrap
  0.1.5 has **no `PETSc_jll` dependency** and throws *at module load* if a system
  `libpetsc.so` is absent (`get_petsc_location` reads `PETSC_DIR`/`PETSC_ARCH`,
  Linux-`.so`-only). Making it a hard dep would make `using BiGSTARS` fail on
  macOS/Windows/any machine without a system PETSc — breaking cross-platform load
  *and* the split-CI plan (the cross-platform jobs could not even load the package
  to run non-solver tests). The earlier web-research claim that PetscWrap falls
  back to a JLL was incorrect (that is PETSc.jl, a different package).

- **What changed vs the design above:**
  - `MPI`/`PetscWrap`/`SlepcWrap` stay **weakdeps**; `[extensions]` is retained.
    They are NOT promoted to `[deps]`.
  - The real `solve` lives in `ext/BiGSTARSMPIExt.jl` (not `src/slepc_solve.jl`).
    `src/solve.jl` holds the `solve` generic + a fallback that errors with an
    install hint when the backend is not imported.
  - Still satisfied: all three serial libs (`KrylovKit`/`ArnoldiMethod`/`Arpack`)
    are removed, SLEPc/PETSc is the only eigensolver, `solve_mpi` is renamed to
    `solve`, adaptive-σ is ported. The only deviation is "hard dep / no fallback":
    there is no in-process fallback, but `solve` is an extension method rather than
    a hard-required symbol, so the package still loads everywhere.
  - Split CI is preserved (it works again because the package loads cross-platform).
  - Examples/tests import PetscWrap/SlepcWrap with `import` (not `using`) because
    PetscWrap exports `solve`, which would otherwise shadow `BiGSTARS.solve`.

## Out of scope

- Distributing the *assembly* across ranks (still serial on rank 0).
- Adding new spectral-transform options beyond `sinvert`.
- Any unrelated refactor of the DSL / discretization pipeline.
