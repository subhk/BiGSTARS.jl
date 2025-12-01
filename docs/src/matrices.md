# Chebyshev and Fourier Differentiation Matrices

This document presents a detailed exposition of Chebyshev‑ and Fourier‑based differentiation matrices, which are essential to spectral discretisations of differential equations, with particular emphasis on their deployment in Chebyshev–Fourier spectral‑collocation frameworks for bi‑global linear stability analysis of geophysical flows, as implemented in BiGSTARS.jl.

## Chebyshev Differentiation Matrices

The code uses **Chebyshev-Gauss-Lobatto collocation points**, defined by:
```julia
xₖ = cos(πk/(n-1)),  k = 0, 1, ..., n-1,
```
where `n` is the number of grid points.
These are the extrema of the Chebyshev polynomial $T_{n-1}(x)$, 
including both endpoints at $x = \pm1$ [@trefethen2000spectral](@cite).

### `chebdif(n::Int, m::Int)`

Compute Chebyshev differentiation matrix of order `m` on `n` Chebyshev points.

### `ChebyshevDiffn`

A spectral differentiation operator using Chebyshev polynomials with pre-computed and cached differentiation matrices.

#### Description

This is the core function that implements the spectral differentiation using a recursive algorithm with Toeplitz matrices for efficient computation.

The `ChebyshevDiffn` struct provides a high-level interface that pre-computes and caches differentiation matrices up to 4th order for efficient repeated use, with automatic domain transformation from [-1,1] to arbitrary intervals [a,b].

#### Parameters

**For `chebdif(n::Int, m::Int)`:**
- **`n`** (`Int`): Number of grid points (n ≥ 2)
- **`m`** (`Int`): Order of differentiation (1 ≤ m < n)

**For `ChebyshevDiffn(n, domain, max_order=1)`:**
- **`n`** (`Int`): Number of grid points
- **`domain`** (`AbstractVector{T}`): Domain interval [a, b] as a 2-element vector where T<:Real
- **`max_order`** (`Int`, optional): Maximum derivative order to compute (1 ≤ max_order ≤ 4, default: 1)

#### Returns

**For `chebdif`:**
- **`x`** (`Vector{Float64}`): Chebyshev grid points in descending order
- **`D`** (`Matrix{Float64}`): m-th order differentiation matrix

**For `ChebyshevDiffn`:**
- A struct containing grid points, domain, and pre-computed differentiation matrices D₁, D₂, D₃, D₄

#### Usage Examples

```julia
# Basic usage with chebdif
x, D1 = chebdif(8, 1)     # 1st derivative on 8 points
x, D2 = chebdif(8, 2)     # 2nd derivative on 8 points
f = exp.(x)               # test function
df = D1 * f               # numerical derivative

# High-level interface with ChebyshevDiffn
cd = ChebyshevDiffn(16, [0.0, 2π], 2)  # 16 points on [0, 2π] with derivatives up to 2nd order

# Use the differentiation matrices
f = sin.(cd.x)
df_dx = cd.D₁ * f         # first derivative
d2f_dx2 = cd.D₂ * f       # second derivative

# Convenience method
df_dx = derivative(cd, f, 1)    # equivalent to cd.D₁ * f
d2f_dx2 = derivative(cd, f, 2)  # equivalent to cd.D₂ * f

# Operator overloading
df_dx = cd * f            # equivalent to cd.D₁ * f
```

#### Properties

- **Spectral accuracy**: For smooth functions, errors decrease exponentially with `N`
- **Dense matrix**: All entries are generally non-zero
- **Boundary treatment**: Properly handles Dirichlet and Neumann boundary conditions
- **Stability**: Well-conditioned for moderate values of `N` (typically N < 100-200)

## Fourier Differentiation Matrices

### `FourierDiff(n::Integer, m::Integer)` 

Compute Fourier spectral differentiation matrices for periodic functions.

#### Description

The `FourierDiff` function computes the m-th order Fourier spectral differentiation matrix on n equispaced points in the interval [0, 2π). It uses efficient Toeplitz matrix representation and handles different derivative orders through specialized computation methods.

The `FourierDiffn` struct provides an elegant high-level interface with automatic domain scaling, intelligent caching of derivative operators, and beautiful indexing syntax for accessing different derivative orders.

#### Parameters

**For `FourierDiff(n::Integer, m::Integer)`:**
- **`n`** (`Integer`): Number of grid points
- **`m`** (`Integer`): Derivative order (m ≥ 0)

**For `FourierDiffn(n::Integer; L::Real = 2π)`:**
- **`n`** (`Integer`): Number of grid points
- **`L`** (`Real`, optional): Domain length (default: 2π, creates domain [0, L))

#### Returns

**For `FourierDiff`:**
- **`x`** (`Vector`): Grid points in [0, 2π)
- **`D`** (`Toeplitz`): Toeplitz differentiation matrix

**For `FourierDiffn`:**
- A struct with fields `n`, `L`, `x` (grid points), and `cache` (cached derivative operators)

#### Usage Examples

```julia
# Basic usage with FourierDiff
x, D1 = FourierDiff(16, 1)    # 1st derivative on 16 points
x, D2 = FourierDiff(16, 2)    # 2nd derivative on 16 points

# Applying to a periodic function
f = sin.(2 .* x)              # Function values
df_dx = D1 * f                # First derivative

# High-level interface with FourierDiffn
𝒟 = FourierDiffn(64; L = 4π)  # 64 points on [0, 4π)

# Beautiful indexing syntax
u = sin.(𝒟.x)
∂u = 𝒟[1] * u                 # First derivative using indexing
∂²u = 𝒟[2] * u                # Second derivative

# Elegant property access with subscripts
∂u = 𝒟.D₁ * u                 # First derivative using property
∂²u = 𝒟.D₂ * u                # Second derivative using property
∂³u = 𝒟.D₃ * u                # Third derivative

# Check cached derivatives
println("Cached orders: ", derivative_orders(𝒟))

# Grid information
println("Domain: [0, $(𝒟.L))")
println("Grid spacing: ", grid_spacing(𝒟))
```

#### Properties

- **Spectral accuracy**: For smooth functions, errors decrease exponentially with `n`
- **Memory-efficient**: Uses Toeplitz matrix representation to reduce storage
- **Automatic scaling**: Properly handles arbitrary domains [0, L) through scaling factor (2π/L)^m
- **Caching system**: `FourierDiffn` intelligently caches computed derivative operators
- **Elegant interface**: Multiple syntax options (indexing `𝒟[m]`, properties `𝒟.Dₘ`)
- **Domain flexibility**: Easy transformation from canonical [0, 2π) to arbitrary [0, L)

#### Advanced Features

**Boundary Condition Enforcement (Chebyshev)**

The BiGSTARS.jl package includes a sophisticated boundary condition system via `setBCs.jl`:

```julia
# Apply boundary conditions
bc_handler = BoundaryConditionHandler(Nz) # Nz is the number of grid points in z-direction
bc_handler(grid, :dirichlet)  # Apply Dirichlet BCs to all relevant operators
bc_handler(grid, :neumann)    # Apply Neumann BCs to all relevant operators
```

**Multiple Domains and Mapping (Chebyshev)**

For domains other than `[-1, 1]`, the ChebyshevDiffn automatically handles scaling:
```julia
# Linear transformation: ζ ∈ [-1,1] → x ∈ [a,b]
# x = (b-a)/2 * (ζ + 1) + a
# Scaling factor α = 2/(b-a) ensures: d^n f/dx^n = α^n * (d^n f/dζ^n)
cd = ChebyshevDiffn(32, [0.0, 10.0], 2)  # Maps to [0, 10]
```

**Efficient Caching (Fourier)**

```julia
# Check what's cached
orders = derivative_orders(𝒟)

# Clear cache except identity
clear_cache!(𝒟)

# Access grid properties
spacing = grid_spacing(𝒟)  # Physical grid spacing
```

