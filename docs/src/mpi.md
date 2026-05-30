# Distributed (MPI) eigensolver

!!! warning "Experimental"
    The distributed backend is new and not yet covered by a green integration
    run. Verify your results against an in-process method on a smaller problem
    before relying on it. The API may change.

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

cache = discretize(prob)
# solve_mpi initializes SLEPc itself — the solver options must be in the PETSc
# options database at init time, so do NOT call SlepcInitialize yourself.
results = solve_mpi(cache, k_values;
                    sigma_0=0.02, nev=5, which=:LM,
                    tol=1e-10, mat_solver=:mumps)
# results is fully populated on rank 0; other ranks get empty markers.
```

Run with `mpiexec -n P julia --project=. script.jl`.

## Worked example: Eady baroclinic instability

The MPI counterpart of [`examples/Eady.jl`](https://github.com/subhk/BiGSTARS.jl/blob/main/examples/Eady.jl)
— the same problem, but the eigensolve is spread across ranks. The full runnable
script is [`examples/eady_mpi.jl`](https://github.com/subhk/BiGSTARS.jl/blob/main/examples/eady_mpi.jl).

You build the problem with the **same DSL** as the serial path; only the solve
call changes (`solve_mpi` instead of `solve`/`EigenSolver`).

```julia
using BiGSTARS
using MPI, PetscWrap, SlepcWrap
using Printf

# Build the EVP (runs on every rank; only rank-0's matrices get used)
domain = Domain(x = FourierTransformed(),
                y = Fourier(60, [0, 1]),
                z = Chebyshev(40, [0, 1]))
prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)

Y, Z = gridpoints(domain, :y, :z)
Ri = 1.0
prob[:U]    = Z .- 0.5
prob[:Ri]   = Ri
prob[:E]    = 1e-12
prob[:dBdy] = -ones(length(Z))
prob[:dQdy] = zeros(length(Z))

@substitution Lap(A) = dx(dx(A)) + dy(dy(A)) + Ri * dz(dz(A))
@equation sigma * Lap(psi) = U * dx(Lap(psi)) + dQdy * dx(psi) - E * Lap(Lap(psi))
@bc left(sigma * dz(psi)  + U * dx(dz(psi)) + dBdy * dx(psi)) = 0
@bc right(sigma * dz(psi) + U * dx(dz(psi)) + dBdy * dx(psi)) = 0

cache = discretize(prob)

# Distributed solve: one eigenproblem at k = 1.0, across all ranks.
# sigma_0 is the shift-and-invert target; the nev modes nearest it come back.
# solve_mpi initializes SLEPc itself, so don't call SlepcInitialize.
results = solve_mpi(cache, [1.0];
                    sigma_0=0.2, nev=6, which=:LM,
                    tol=1e-10, mat_solver=:mumps)

# Only rank 0 has populated results; other ranks get empty markers.
rank = MPI.Comm_rank(MPI.COMM_WORLD)
if rank == 0
    r = results[1]
    if r.converged
        i = argmax(real.(r.eigenvalues))   # most unstable = largest growth
        σ = r.eigenvalues[i]
        @printf("most unstable σ = %.6f %+.6fi\n", real(σ), imag(σ))
    else
        println("did not converge")
    end
end
```

### One-time environment setup

```bash
# 1. Build complex-scalar PETSc + SLEPc; export PETSC_DIR / PETSC_ARCH / SLEPC_DIR
#    ./configure --with-scalar-type=complex --download-mumps --download-scalapack

# 2. Bind MPI.jl to the SAME MPI used for PETSc, in an env with the weakdeps
#    (the repo's test/mpi environment works):
julia --project=test/mpi -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=test/mpi -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
```

Then launch across, e.g., four ranks:

```bash
mpiexec -n 4 julia --project=test/mpi examples/eady_mpi.jl
```

## Notes

- `which=:LM` targets eigenvalues nearest `sigma_0` (shift-and-invert), matching
  the serial backends. `:LR`/`:SR`/`:LI`/`:SI` select by real/imaginary extremes.
- `mat_solver` picks the parallel direct solver for the inner solves
  (`:mumps`, `:superlu_dist`, or `:petsc`).
- v1 does a single solve at `sigma_0` (no adaptive-σ retry) and gathers
  eigenvectors to rank 0 for reconstruction.
