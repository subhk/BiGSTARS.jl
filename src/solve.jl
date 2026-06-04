# ==============================================================================
# solve.jl — public eigensolve entrypoint. The real implementation lives in the
# package extension BiGSTARSMPIExt (loaded when MPI, PetscWrap, and SlepcWrap are
# all imported). Without that backend, `solve` errors with an install hint. There
# is no in-process fallback — SLEPc/PETSc is the only eigensolver.
# ==============================================================================

"""
    solve(cache, k_values; sigma_0, nev=1, which=:LM, tol=1e-10, maxiter=300,
          ncv=0, mat_solver=:mumps, eps_type=:krylovschur,
          n_tries=8, Δσ₀=0.2, incre=1.2, ϵ=1e-5,
          manage_init=true, verbose=false) -> Vector{SolverResults}

Distributed eigensolve via SLEPc over PETSc, one generalized pencil per wavenumber
spread across all MPI ranks, with adaptive-σ shift-and-invert. Provided by the
package extension `BiGSTARSMPIExt`, which loads only when `MPI`, `PetscWrap`, and
`SlepcWrap` are all imported. Returns `Vector{SolverResults}`, fully populated on
rank 0; other ranks return empty markers.

Requires a complex-scalar PETSc/SLEPc build (`--with-scalar-type=complex`, with
`PETSC_DIR`/`PETSC_ARCH`/`SLEPC_DIR` exported). Run under
`mpiexec -n P julia --project=. script.jl`. See `docs/src/mpi.md`.
"""
function solve end

"""
    solve(cache; sigma_0, kwargs...) -> Vector{SolverResults}

Single-problem overload (no wavenumber sweep). Also provided by the extension.
"""

# Least-specific fallback: the extension's concrete-typed methods win when the
# backend is loaded; otherwise this fires with an install hint.
function solve(@nospecialize(args...); kwargs...)
    error("BiGSTARS.solve requires the SLEPc/PETSc backend: install and import " *
          "MPI, PetscWrap, and SlepcWrap, plus a complex-scalar system PETSc/SLEPc " *
          "build (set PETSC_DIR, PETSC_ARCH, SLEPC_DIR). See docs/src/mpi.md.")
end
