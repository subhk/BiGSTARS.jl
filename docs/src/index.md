# BiGSTARS.jl Documentation

## Overview

`BiGSTARS.jl` — a Julia toolkit for bi-global linear-stability analysis of geophysical flows, featuring a symbolic equation DSL and fully sparse coefficient-space spectral methods.

Bi-global analysis bridges the gap between 1D stability tools (often too idealized) and fully 3D tri-global solvers (often too costly). BiGSTARS.jl offers a practical middle ground for rotating, stratified flows with:

- **Symbolic equation DSL** — Write governing equations in physical-space notation (`dx`, `dy`, `dz`) using Julia macros. The package automatically discretizes them into generalized eigenvalue problem (GEVP) matrices.
- **Ultraspherical spectral method** — Chebyshev directions use the ultraspherical method (Olver & Townsend, 2013) for fully sparse operators. Fourier directions operate in coefficient space with diagonal derivatives.
- **Wavenumber-separated caching** — Discretize once, then assemble cheaply for each scalar wavenumber or for multiple transformed directions such as `(k_x, k_y)`.
- **Generalized boundary conditions** — Dirichlet, Neumann, Robin, higher-order, coupled, inhomogeneous, and eigenvalue-dependent (dynamic) BCs.
- **Derived variables** — Auxiliary variables like cross-front velocity `v` can be defined implicitly via `@derive` and are automatically eliminated using the inverse operator.
- **Multiple eigenvalue solvers** — Arnoldi, ARPACK, and KrylovKit methods with adaptive shift-and-invert.
- **2D background fields** — Full support for fields varying in both Fourier and Chebyshev directions.
- **Parallel wavenumber sweeps** — Thread-parallel solves with in-place assembly for zero-allocation loops.

## What To Read First

If you are new to the package, use this path:

1. [Equation DSL](@ref) — define domains, variables, equations, boundary conditions, and derived variables.
2. [Eigenvalue Solver](@ref) — choose a solver, shift, tolerance, and number of eigenvalues.
3. [Post-Processing and Visualization](@ref) — reconstruct fields from eigenvectors and convert them to physical space.
4. [Examples](@ref) — compare your workflow with complete Eady, Stone, and rotating RBC examples.

## Quick Start

```julia
using BiGSTARS

# Define the domain
domain = Domain(
    x = FourierTransformed(),       # wavenumber direction
    y = Fourier(60, [0, 1]),        # periodic, N=60, domain [0, 1)
    z = Chebyshev(30, [0, 1])       # bounded, N=30, domain [0, 1]
)

# Set up eigenvalue problem
prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)

# Background state and parameters
Z = gridpoints(domain, :z)
prob[:U] = Z .- 0.5
prob[:E] = 1e-12

# Define operators and equations
@substitution Lap(A) = dx(dx(A)) + dy(dy(A)) + dz(dz(A))
@equation sigma * Lap(psi) = U * dx(Lap(psi)) - E * Lap(Lap(psi))

# Boundary conditions
@bc left(psi) = 0
@bc right(psi) = 0

# Discretize once, solve over wavenumbers
cache = discretize(prob)
k_values = range(0.1, 5.0, length=50)
results = solve(cache, k_values; sigma_0=0.02, method=:Krylov)

# For problems without wavenumber direction:
results = solve(cache; sigma_0=0.02)
```

For domains with more than one `FourierTransformed()` direction, assemble with named wavenumbers:

```julia
domain = Domain(
    x = FourierTransformed(),
    y = FourierTransformed(),
    z = Chebyshev(40, [0, 1])
)

cache = discretize(prob)
A, B = assemble(cache; k_x=1.0, k_y=0.5)
```

Keyword names are checked. Use either the coordinate name (`x=1.0`) or the explicit wavenumber name (`k_x=1.0`), but do not provide conflicting aliases.

## Examples

* [Eady example](literated/Eady.md) — QG PV baroclinic instability with dynamic BCs
* [Stone example](literated/Stone1971.md) — Non-hydrostatic Boussinesq with derived variables
* [rRBC example](literated/rRBC.md) — Rotating Rayleigh-Benard with constraint equations
