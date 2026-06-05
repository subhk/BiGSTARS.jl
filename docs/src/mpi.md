# Distributed (MPI) eigensolver

BiGSTARS solves eigenproblems across MPI ranks using SLEPc over PETSc — this is
the package's only eigensolver. Assembly stays serial on rank 0; the
shift-and-invert factorization and the Krylov eigensolve are distributed. The
entrypoint is [`solve`](@ref), provided by the package extension `BiGSTARSMPIExt`,
which activates when `MPI`, `PetscWrap`, and `SlepcWrap` are all imported.

## Requirements

- A **complex-scalar** system build of PETSc and SLEPc
  (`./configure --with-scalar-type=complex`), with `PETSC_DIR`, `PETSC_ARCH`,
  and `SLEPC_DIR` exported.
- Julia packages `MPI`, `PetscWrap`, `SlepcWrap` installed. Importing all three
  activates the `BiGSTARSMPIExt` extension and the real `solve`. `]add BiGSTARS`
  works everywhere, but **solving** requires the complex PETSc/SLEPc build above;
  without it `solve` raises an install hint.

## Usage

```julia
using BiGSTARS
using MPI
import PetscWrap, SlepcWrap   # import (not using): PetscWrap exports `solve`, which would shadow BiGSTARS.solve

cache = discretize(prob)
# solve initializes SLEPc itself — the static solver options must be in the PETSc
# options database at init time, so do NOT call SlepcInitialize yourself.
results = solve(cache, k_values;
                sigma_0=0.02, nev=5, which=:LM,
                tol=1e-10, mat_solver=:mumps)
# results is fully populated on rank 0; other ranks get empty markers.
```

Run with `mpiexec -n P julia --project=. script.jl`.

### Parallel across wavenumbers (`ngroups`)

By default (`ngroups=1`) all ranks collaborate on one wavenumber's pencil at a
time. Set `ngroups=G` to split the `P` ranks into `G` equal groups (requires
`P % G == 0`): the wavenumbers are distributed round-robin across groups, each
group solving its subset on its own sub-communicator while the matrix of each
pencil is still distributed across that group's ranks. Results are gathered to
global rank 0.

```julia
# 8 ranks, 4 groups of 2: 4 wavenumbers solved concurrently, each pencil on 2 ranks
mpiexec -n 8 julia --project=. -e '... solve(cache, k_values; sigma_0=0.02, nev=5, ngroups=4) ...'
```

Pick `ngroups` to balance across-wavenumber concurrency (more groups) against
per-solve distribution (larger groups). Assembly of each pencil is still serial on
its group's rank 0.

### Lower per-rank memory: `discretize_distributed`

For very large problems, build the cache with `discretize_distributed(prob; ngroups=G)`
instead of `discretize(prob)`. It returns a cache holding only that rank's owned rows of
the operator components on every rank. Every rank (including group roots) holds only its
owned rows: the singular-B spurious-mode filter is computed distributedly (`MatMult` +
`MPI.Allreduce`), so no rank needs the full mass matrix. Pass the **same** `ngroups` to
`solve`:

```julia
cache = discretize_distributed(prob; ngroups=4)
results = solve(cache, k_values; sigma_0=0.02, nev=5, ngroups=4)
```

Notes: the full cache is still built transiently on every rank (peak memory is
unchanged), legacy `augment_derived=false` terms are not reduced, and the returned
cache is tied to this rank + `ngroups` (not portable). A plain `discretize` cache
works with `solve` exactly as before.

## Worked example: Eady baroclinic instability

The full runnable script is
[`examples/eady_mpi.jl`](https://github.com/subhk/BiGSTARS.jl/blob/main/examples/eady_mpi.jl).
You build the problem with the **same DSL** as always; `solve` runs it across ranks.

```julia
using BiGSTARS
using MPI
import PetscWrap, SlepcWrap
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
results = solve(cache, [1.0];
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

- `which=:LM` targets eigenvalues nearest `sigma_0` (shift-and-invert).
  `:LR`/`:SR`/`:LI`/`:SI` select by real/imaginary extremes.
- `mat_solver` picks the parallel direct solver for the inner solves
  (`:mumps`, `:superlu_dist`, or `:petsc`).
- `solve` runs an adaptive-σ retry loop (`n_tries`, `Δσ₀`, `incre`, `ϵ`) around
  `sigma_0`, and gathers eigenvectors to rank 0 for reconstruction.
```
