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
