# Eigenvalue Solver

BiGSTARS solves the generalized eigenvalue problem

```math
A x = \lambda B x
```

with a single distributed eigensolver built on **SLEPc over PETSc**. The
shift-and-invert spectral transform targets eigenvalues near a shift `σ`, the
pencil is distributed across MPI ranks, and an adaptive-σ loop retries nearby
shifts until the eigenvalue at the shift is stable.

There is one entrypoint, [`solve`](@ref). It is provided by the package extension
`BiGSTARSMPIExt`, which activates when `MPI`, `PetscWrap`, and `SlepcWrap` are all
imported. Without that backend (and a complex-scalar PETSc/SLEPc build) `solve`
raises an install hint. See [Distributed (MPI)](mpi.md) for environment setup.

!!! note "Requirements"
    `solve` requires a **complex-scalar** system PETSc/SLEPc build
    (`./configure --with-scalar-type=complex`) with `PETSC_DIR`, `PETSC_ARCH`,
    `SLEPC_DIR` exported, and MPI.jl bound to the same MPI. `]add BiGSTARS` works
    everywhere, but solving needs that build. Run scripts with
    `mpiexec -n P julia --project=. script.jl`.

## Usage

```julia
using BiGSTARS
using MPI
import PetscWrap, SlepcWrap   # import (not using): PetscWrap exports `solve`, which would shadow BiGSTARS.solve

cache = discretize(prob)

# One pencil per wavenumber, distributed across ranks:
results = solve(cache, range(0.1, 5.0, length=50); sigma_0=0.02, nev=1)

# No wavenumber sweep (single problem):
results = solve(cache; sigma_0=0.02, nev=5)
```

`solve` returns a `Vector{SolverResults}`, fully populated on rank 0; other ranks
return empty markers. Guard post-processing with `MPI.Comm_rank(MPI.COMM_WORLD) == 0`.

## Keyword arguments

| Parameter     | Default        | Description                                              |
|---------------|----------------|---------------------------------------------------------|
| `sigma_0`     | required       | Shift-and-invert target (initial shift)                 |
| `nev`         | `1`            | Number of eigenvalues to compute                        |
| `which`       | `:LM`          | Selection: `:LM` (nearest σ), `:LR`, `:SR`, `:LI`, `:SI`|
| `tol`         | `1e-10`        | Convergence tolerance                                   |
| `maxiter`     | `300`          | Maximum SLEPc iterations                                |
| `ncv`         | `0`            | EPS subspace size (`0` lets SLEPc choose)               |
| `mat_solver`  | `:mumps`       | Parallel direct solver (`:mumps`, `:superlu_dist`, `:petsc`) |
| `eps_type`    | `:krylovschur` | SLEPc EPS type                                          |
| `n_tries`     | `8`            | Adaptive-σ retry attempts each side of σ₀               |
| `Δσ₀`         | `0.2`          | Initial shift increment                                 |
| `incre`       | `1.2`          | Increment growth factor                                 |
| `ϵ`           | `1e-5`         | Successive-eigenvalue convergence tolerance             |
| `manage_init` | `true`         | Let `solve` call `SlepcInitialize` on first use         |
| `verbose`     | `false`        | Print per-attempt progress on rank 0                    |

## Results

```julia
struct SolverResults
    eigenvalues::Vector{ComplexF64}   # computed eigenvalues
    eigenvectors::Matrix{ComplexF64}  # corresponding eigenvectors (gathered on rank 0)
    converged::Bool
    method_used::Symbol               # :Slepc
    final_shift::Float64              # accepted shift
    iterations::Int                   # number of σ attempts
    solve_time::Float64
    history::ConvergenceHistory       # per-attempt shifts / convergence / λ₁
end
```

`print_summary(r::SolverResults)` prints a compact summary.

## Shift-and-invert and adaptive-σ

The shift-and-invert transform targets eigenvalues near `σ`:

```math
(A - \sigma B)^{-1} B x = \mu x, \quad \mu = (\lambda - \sigma)^{-1},
```

so eigenvalues `λ` closest to `σ` become the dominant `μ`. SLEPc factorizes
`(A - σB)` with the chosen parallel direct solver (`mat_solver`) and runs
Krylov–Schur on the transformed operator.

If the eigenvalue at the shift is not yet stable, `solve` retries a geometric
schedule of nearby shifts (the same schedule the package has always used):

```julia
up   = [Δσ₀ * incre^(i-1) * abs(σ₀) for i in 1:n_tries]
schedule = vcat(σ₀, σ₀ .+ up, σ₀ .- up)
```

Each attempt re-targets SLEPc via `EPSSetTarget`; rank 0 filters spurious
(singular-`B`) modes, sorts the survivors by distance to the shift, and decides
whether successive `λ₁` agree to within `ϵ`. The stop decision is broadcast so
every rank stays in lockstep (the solve is collective).

## Spurious-mode filtering

Descriptor / augmented-derived systems have a singular `B` (zero rows for
constraint and boundary equations), producing infinite eigenvalues. `solve` drops
modes with `‖Bχ‖ ≈ 0` and keeps the physical `O(1)`-mass modes. For a non-singular
`B` this is a no-op.

## Utility functions

```julia
sort_eigenvalues!(λ, Χ, by; rev=true, σ=nothing)  # :nearest (needs σ), :R, :I, :M
print_evals(λ)                                    # pretty-print an eigenvalue list
sort_evals(λ, Χ, by; …)                           # sort an (λ, Χ) pair
remove_evals(λ, Χ, lo, hi, by)                     # keep eigenvalues in a band
```
