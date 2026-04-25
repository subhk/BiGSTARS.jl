# Equation DSL

BiGSTARS.jl provides a symbolic DSL for defining eigenvalue problems. Users write governing equations in physical-space notation, and the package automatically discretizes them into sparse GEVP matrices.

## Domain Setup

Declare coordinates with named directions and discretization types:

```julia
domain = Domain(
    x = FourierTransformed(),        # wavenumber direction
    y = Fourier(60, [0, 1]),         # periodic, N=60, domain [0, 1)
    z = Chebyshev(30, [0, 1])        # bounded, N=30, domain [0, 1]
)
```

### Coordinate Types

| Type | Constructor forms | Produces |
|------|------------------|----------|
| `FourierTransformed()` | `FourierTransformed()` | named wavenumber multiplication, e.g. `dx(A) -> im*k_x*A` |
| Fourier (periodic) | `Fourier(N, [a, b])` or `Fourier(N=..., L=...)` | Diagonal Fourier operator, wavenumbers scaled by `2pi/(b-a)` |
| Chebyshev (bounded) | `Chebyshev(N, [a, b])` or `Chebyshev(N=..., lower=..., upper=...)` | Sparse ultraspherical operator, derivatives scaled by `2/(b-a)` |

All derivatives are automatically scaled to the physical domain. For example, `dz(psi)` on `Chebyshev(30, [0, 1])` applies the chain rule factor `2/(1-0) = 2` to the reference-domain derivative.

Multiple `FourierTransformed()` directions are allowed. A derivative in each transformed coordinate gets its own wavenumber symbol:

```julia
domain = Domain(
    x = FourierTransformed(),
    y = FourierTransformed(),
    z = Chebyshev(30, [0, 1])
)

# Internally:
# dx(A) -> im*k_x*A
# dy(A) -> im*k_y*A
```

This matters for operators such as `dx(dx(A)) + dy(dy(A))`: BiGSTARS keeps the `k_x^2` and `k_y^2` terms separate instead of collapsing them into one total power of `k`.

### Meshgrid for 2D Fields

For problems with spatially varying background fields:

```julia
Y, Z = meshgrid(domain, :y, :z)   # N_z x N_y matrices
U = @. Z - 0.5 + 0.1 * sin(2*pi*Y)  # 2D field
prob[:U] = vec(U)  # flatten with z-fastest ordering
```

## Problem Definition

Create an eigenvalue problem and declare variables and parameters:

```julia
prob = EVP(domain, variables=[:psi, :b], eigenvalue=:sigma)

# Background fields
Z = gridpoints(domain, :z)
prob[:U] = Z .- 0.5       # 1D field (z-only)
prob[:E] = 1e-12           # scalar parameter

# 2D fields
Y, Z = meshgrid(domain, :y, :z)
prob[:B] = vec(@. Ri * Z - Y)  # 2D field, flattened
```

Parameters can be: scalars (`Number`), 1D fields (`Vector` of length N_z), 2D fields (`Vector` of length N_y*N_z), or pre-computed operator matrices (`AbstractMatrix` of size N_per_var x N_per_var).

## Substitutions

Define reusable operator shorthand:

```julia
@substitution Lap(A) = dx(dx(A)) + dy(dy(A)) + dz(dz(A))
@substitution D4(A)  = Lap(Lap(A))
```

## Equations

Write equations using `dx()`, `dy()`, `dz()` for derivatives:

```julia
# Standard eigenvalue equation:
@equation sigma * Lap(psi) = U * dx(Lap(psi)) - E * Lap(Lap(psi))

# Coupled system:
@equation sigma * Lap(psi) = U * dx(Lap(psi)) + dz(b)
@equation sigma * b        = U * dx(b) - E * Lap(b)

# Constraint equation (no eigenvalue — B matrix rows are zero):
@equation 0 = dz(w) + E * D2(zeta)
```

## Derived Variables

Define auxiliary variables implicitly. The DSL computes the inverse operator and substitutes automatically. Two forms are supported:

**Form 1: Named operator** (requires a `@substitution`):
```julia
@substitution Dh2(A) = dx(dx(A)) + dy(dy(A))
@derive Dh2_v = -dy(dz(w)) + dx(zeta)
```

**Form 2: Inline operator** (no separate `@substitution` needed):
```julia
@derive v dx(dx(v)) + dy(dy(v)) = -dy(dz(w)) + dx(zeta)
```

Both define `v = Op^{-1} * rhs`. The variable `v` can then be used freely in equations:

```julia
@equation sigma * b = dBdz * w + dBdy * v + U * dx(b) - E * D2(b)
```

Form 2 works with any linear operator expression — derivatives, parameter coefficients, combinations:
```julia
@derive v 3*dz(dz(v)) + alpha*v = dz(w)   # general operator
```

### Boundary Conditions for Derived Variables

When the operator has a null space (e.g., `dz(dz(v)) = rhs` is undetermined up to `a + bz`), add BCs with `@derive_bc`:

```julia
@derive v dz(dz(v)) = source_field
@derive_bc v left(v) = 0
@derive_bc v right(v) = 0
```

For operators like `Dh2 = dx² + dy²` that are invertible per Fourier mode, no BCs are needed.

The inverse is computed per-wavenumber using sparse block-diagonal inversion (O(N_y * N_z^3) per k), correctly handling k-dependent operators like `Dh2 = dx^2 + dy^2 = -k^2 + dy^2`.

## Boundary Conditions

### Algebraic BCs (no eigenvalue)

```julia
@bc left(psi) = 0                    # Dirichlet
@bc right(dz(psi)) = 0              # Neumann
@bc left(3*psi + dz(psi)) = 0       # Robin
@bc right(dz(dz(psi))) = 0          # Higher-order
@bc left(psi + b) = 0               # Coupled
```

### Dynamic BCs (eigenvalue-dependent)

When a BC contains the eigenvalue symbol, the DSL automatically splits it into A-side and B-side rows:

```julia
# Eady buoyancy boundary condition:
@bc left(sigma * dz(psi) + U * dx(dz(psi)) + dBdy * dx(psi)) = 0
@bc right(sigma * dz(psi) + U * dx(dz(psi)) + dBdy * dx(psi)) = 0
```

For multiple Chebyshev directions, use coordinate-qualified form: `left(expr, :z)`.

## Discretize and Solve

```julia
# Discretize once
cache = discretize(prob)

# Solve over wavenumbers
k_values = range(0.1, 5.0, length=50)
results = solve(cache, k_values; sigma_0=0.02, method=:Krylov)

# Parallel (requires julia -t auto):
results = solve(cache, k_values; sigma_0=0.02, parallel=true)

# For problems without wavenumber direction:
results = solve(cache; sigma_0=0.02)
```

### Manual Assembly

For a single transformed direction, pass the scalar wavenumber:

```julia
cache = discretize(prob)
A, B = assemble(cache, 1.0)
```

For multiple transformed directions, use keyword assembly. Keyword names must match transformed coordinate names:

```julia
domain = Domain(
    x = FourierTransformed(),
    y = FourierTransformed(),
    z = Chebyshev(30, [0, 1])
)

cache = discretize(prob)
A, B = assemble(cache; k_x=1.0, k_y=0.5)

# Short coordinate aliases are also accepted:
A, B = assemble(cache; x=1.0, y=0.5)
```

Unknown keywords are rejected so typos fail loudly:

```julia
assemble(cache; kx=1.0)  # error: use k_x or x
```

!!! note
    `solve(cache, k_values; ...)` is the high-level sweep API for one scalar wavenumber applied to every transformed direction. For genuinely independent `k_x, k_y, ...` scans, call `assemble(cache; k_x=..., k_y=...)` and then use `EigenSolver` directly.

### Inspecting the Cache

```julia
cache = discretize(prob)
println(cache)  # shows size, k-powers, sparsity
A, B = assemble(cache, 1.0)  # single transformed direction
```

## Post-Processing with `@compute`

After solving, evaluate any expression on the eigenvector — same DSL syntax:

```julia
results = solve(cache, [0.1]; sigma_0=0.02)
@compute_setup cache results[1].eigenvectors[:, 1] 0.1

@compute v = -dy(dz(w)) + dx(zeta)
@compute u = -dx(dz(w)) - dy(zeta)
@compute KE = u * u + v * v
```

See [Post-Processing and Visualization](@ref) for details.

## Differentiating Background Profiles

Use `differentiate` to compute derivatives of known background fields spectrally:

```julia
z = gridpoints(domain, :z)
U = @. tanh(10 * (z - 0.5))
prob[:Uz]  = differentiate(U, domain, :z)
prob[:Uzz] = differentiate(U, domain, :z; order=2)
```

For Fourier directions, optional spectral filtering prevents Gibbs ringing:

```julia
prob[:Vy] = differentiate(V, domain, :y; filter=:exp)
```

## Complete Example (Eady Problem)

```julia
using BiGSTARS

domain = Domain(
    x = FourierTransformed(),
    y = Fourier(60, [0, 1]),
    z = Chebyshev(30, [0, 1])
)

prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)

Z = gridpoints(domain, :z)
prob[:U]    = Z .- 0.5
prob[:E]    = 1e-12
prob[:dBdy] = -ones(length(Z))

@substitution Lap(A) = dx(dx(A)) + dy(dy(A)) + dz(dz(A))

@equation sigma * Lap(psi) = U * dx(Lap(psi)) - E * Lap(Lap(psi))

# Dynamic BCs (eigenvalue-dependent buoyancy condition):
@bc left(sigma * dz(psi) + U * dx(dz(psi)) + dBdy * dx(psi)) = 0
@bc right(sigma * dz(psi) + U * dx(dz(psi)) + dBdy * dx(psi)) = 0

cache = discretize(prob)
results = solve(cache, [0.1]; sigma_0=0.02, method=:Krylov)
```

## Complete Example (Stone 1971 with Derived Variables)

```julia
using BiGSTARS

domain = Domain(
    x = FourierTransformed(),
    y = Fourier(24, [0, 1]),
    z = Chebyshev(20, [0, 1])
)

prob = EVP(domain, variables=[:w, :zeta, :b], eigenvalue=:sigma)

Z = gridpoints(domain, :z)
prob[:U]    = Z .- 0.5
prob[:dUdz] = ones(length(Z))
prob[:dBdy] = -ones(length(Z))
prob[:dBdz] = ones(length(Z))
prob[:E]    = 1e-8
prob[:eps2] = 0.01
prob[:eps2inv] = 100.0

@substitution D2(A)  = dx(dx(A)) + dy(dy(A)) + eps2inv * dz(dz(A))
@substitution Dh2(A) = dx(dx(A)) + dy(dy(A))
@substitution D4(A)  = D2(D2(A))

# Derived variable: v = cross-front velocity
@derive Dh2_v = -dy(dz(w)) + dx(zeta)
# Or equivalently with inline operator:
#   @derive v dx(dx(v)) + dy(dy(v)) = -dy(dz(w)) + dx(zeta)

@equation sigma * eps2 * D2(w) = -eps2 * U * dx(D2(w)) + eps2 * E * D4(w) - dz(zeta) + Dh2(b)
@equation sigma * zeta = dUdz * dy(w) + dz(w) - U * dx(zeta) + E * D2(zeta)
@equation sigma * b = -dBdz * w - dBdy * v - U * dx(b) + E * D2(b)

@bc left(w) = 0;          @bc right(w) = 0
@bc left(dz(dz(w))) = 0;  @bc right(dz(dz(w))) = 0
@bc left(dz(zeta)) = 0;   @bc right(dz(zeta)) = 0
@bc left(dz(b)) = 0;      @bc right(dz(b)) = 0

cache = discretize(prob)
results = solve(cache, [0.1]; sigma_0=0.02, method=:Krylov)
```
