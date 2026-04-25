# BiGSTARS.jl Documentation

```@raw html
<div class="bigstars-hero">
  <div class="bigstars-eyebrow">Bi-global stability analysis in Julia</div>
  <h1>Sparse spectral eigenvalue problems from readable equations.</h1>
  <p>
    BiGSTARS.jl turns physical-space notation such as <code>dx</code>, <code>dy</code>, and <code>dz</code>
    into sparse generalized eigenvalue matrices for rotating, stratified, and geophysical flows.
  </p>
  <div class="bigstars-actions">
    <a class="bigstars-button primary" href="equation_dsl.html">Start with the DSL</a>
    <a class="bigstars-button secondary" href="literated/Eady.html">Open an example</a>
  </div>
</div>
```

## Overview

Bi-global analysis bridges the gap between 1D stability tools (often too idealized) and fully 3D tri-global solvers (often too costly). BiGSTARS.jl offers a practical middle ground for rotating, stratified flows.

```@raw html
<div class="bigstars-card-grid">
  <div class="bigstars-card">
    <strong>Equation DSL</strong>
    <p>Write governing equations directly with <code>dx</code>, <code>dy</code>, <code>dz</code>, substitutions, and boundary-condition macros.</p>
  </div>
  <div class="bigstars-card">
    <strong>Sparse spectral operators</strong>
    <p>Chebyshev directions use ultraspherical operators; Fourier directions stay diagonal in coefficient space.</p>
  </div>
  <div class="bigstars-card">
    <strong>Wavenumber caching</strong>
    <p>Discretize once, then assemble quickly for scalar wavenumbers or named directions such as <code>k_x</code> and <code>k_y</code>.</p>
  </div>
  <div class="bigstars-card">
    <strong>Solver workflow</strong>
    <p>Use Arnoldi, ARPACK, or KrylovKit with adaptive shift-and-invert and thread-parallel wavenumber sweeps.</p>
  </div>
</div>
```

## What To Read First

```@raw html
<div class="bigstars-path">
  <div class="bigstars-step">
    <a href="equation_dsl.html">Equation DSL</a>
    <p>Define domains, variables, equations, boundary conditions, and derived variables.</p>
  </div>
  <div class="bigstars-step">
    <a href="method.html">Eigenvalue Solver</a>
    <p>Choose a solver, shift, tolerance, and number of eigenvalues.</p>
  </div>
  <div class="bigstars-step">
    <a href="visualization.html">Post-processing</a>
    <p>Reconstruct fields from eigenvectors and convert them to physical space.</p>
  </div>
  <div class="bigstars-step">
    <a href="literated/Eady.html">Examples</a>
    <p>Compare your workflow with complete Eady, Stone, and rotating RBC examples.</p>
  </div>
</div>
```

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
