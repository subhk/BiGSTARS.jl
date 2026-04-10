# Utility Functions

## Spectral Differentiation

### `differentiate(f, domain, coord; order=1, filter=:none)`

Spectrally differentiate a function given as physical-space values on the grid. This is the primary tool for computing derivatives of background profiles.

**Chebyshev directions** — converts to Chebyshev T coefficients, applies the ultraspherical differentiation chain, and converts back to physical space.

**Fourier directions** — applies `(ik)^p` in Fourier space with optional spectral filtering.

```julia
domain = Domain(z = Chebyshev(64, [0, 1]))
z = gridpoints(domain, :z)

U = @. z^2
dUdz   = differentiate(U, domain, :z)            # 2z
d2Udz2 = differentiate(U, domain, :z; order=2)   # 2
```

**Fourier with filtering:**

```julia
domain = Domain(y = Fourier(64, [0, 2*pi]))
y = gridpoints(domain, :y)
f = sin.(y)
dfdy = differentiate(f, domain, :y; filter=:exp)   # exponential filter
```

| Filter | Description |
|--------|-------------|
| `:none` | No filtering (default) |
| `:exp` | Exponential filter (Vandeven 1991) |
| `Symbol("2/3")` | 2/3 dealiasing rule |
| `Function` | Custom `sigma(k, k_max)` |

## Eigenvalue Analysis

### `print_evals(lambdas)`

Pretty-print eigenvalues in a formatted table.

### `sort_evals(lambdas, chi, by; rev=true)`

Sort eigenvalues and eigenvectors. `by` can be a Symbol (`:R`, `:I`, `:M`) or String (`"R"`, `"I"`, `"M"`).

```julia
lambdas_sorted, chi_sorted = sort_evals(lambdas, chi, :R; rev=true)
```

### `remove_evals(lambdas, chi, lower, higher, which)`

Filter eigenvalues within bounds. `which` can be Symbol or String.

```julia
lambdas_filtered, chi_filtered = remove_evals(lambdas, chi, 0.1, 1e5, :R)
```

## Grid and Transform Utilities

### `gridpoints(domain, dims...)`

Return grid points for resolved dimensions.

```julia
z = gridpoints(domain, :z)              # 1D
y, z = gridpoints(domain, :y, :z)       # separate vectors
```

### `meshgrid(domain, dim1, dim2)`

Return 2D meshgrid arrays (N_dim2 x N_dim1, z-fastest ordering).

```julia
Y, Z = meshgrid(domain, :y, :z)
U = @. Z - 0.5 + 0.1 * sin(2*pi * Y)  # 2D field
prob[:U] = vec(U)
```

### `to_coefficients(f, coord_type)`

Transform physical-space values to spectral coefficients.

```julia
c = to_coefficients(f_values, :chebyshev)
c = to_coefficients(f_values, :fourier)
```

### `to_physical(c, coord_type; x=nothing)`

Transform spectral coefficients back to physical space.

### `chebyshev_points(N, a, b)`

N Chebyshev-Gauss-Lobatto points on [a, b].

### `chebyshev_coefficients(f_values)`

Chebyshev T coefficients from values at CGL points.

### `chebyshev_evaluate(c, x)`

Evaluate a Chebyshev expansion via Clenshaw's algorithm.

### `fourier_points(N, L)`

N equally-spaced points on [0, L).
