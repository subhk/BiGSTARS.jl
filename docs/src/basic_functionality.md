# utils.jl Documentation

A Julia utility library providing functions for eigenvalue analysis, matrix operations, numerical differentiation, and sparse matrix construction.

## Core Functions

### Array Utilities

#### `myfindall(condition, x)`

Custom implementation of `findall` that returns indices where a condition is satisfied.

**Parameters:**
- `condition`: Function that returns a boolean
- `x`: Input array to search

**Returns:**
- `Vector{Int}`: Indices where condition is true

**Example:**
```julia
indices = myfindall(x -> x > 5, [1, 6, 3, 8, 2])
# Returns: [2, 4]
```

### Eigenvalue Analysis

#### `EigenvalueDisplay`

Custom struct for formatted eigenvalue output with real and imaginary parts.

**Fields:**
- `λ::Complex`: The eigenvalue
- `idx::Int`: Index/position of the eigenvalue

#### `print_evals(λs::Vector{<:Complex})`

Pretty-print eigenvalues in a formatted table showing real and imaginary parts.

**Parameters:**
- `λs`: Vector of complex eigenvalues

**Output Format:**
```
Top N eigenvalues (sorted):
Idx │ Real Part     Imag Part
────┼──────────────────────────────
  1 │  1.234567   +1.234567im
  2 │ -2.345678e  +0.548678im         
```

#### `sort_evals(λs, χ, which; sorting="lm")`

Sort eigenvalues and corresponding eigenvectors by specified criteria.

**Parameters:**
- `λs::AbstractVector`: Eigenvalues
- `χ::AbstractMatrix`: Eigenvectors (columns correspond to eigenvalues)
- `which::String`: Sorting criterion (`"M"` for magnitude, `"I"` for imaginary part, `"R"` for real part)
- `sorting::String`: Order (`"lm"` for descending, default)

**Returns:**
- `Tuple`: Sorted eigenvalues and eigenvectors

#### `sort_evals_(λ, Χ, by; rev=true)`

Alternative eigenvalue sorting function with symbol-based criteria.

**Parameters:**
- `λ::Vector`: Eigenvalues
- `Χ::Matrix`: Eigenvectors
- `by::Symbol`: Sorting criterion (`:R` for real, `:I` for imaginary, `:M` for magnitude)
- `rev::Bool`: Reverse order (descending if true)

**Returns:**
- `Tuple`: Sorted eigenvalues and eigenvectors

#### `remove_evals(λs, χ, lower, higher, which)`

Filter eigenvalues within specified bounds and remove corresponding eigenvectors.

**Parameters:**
- `λs`: Eigenvalues
- `χ`: Eigenvectors
- `lower`: Lower bound
- `higher`: Upper bound
- `which::String`: Component to filter (`"M"`, `"I"`, or `"R"`)

**Returns:**
- `Tuple`: Filtered eigenvalues and eigenvectors

#### `remove_spurious(λₛ, X)`

Remove the first (typically spurious) eigenvalue and corresponding eigenvector.

**Parameters:**
- `λₛ`: Eigenvalues
- `X`: Eigenvectors

**Returns:**
- `Tuple`: Eigenvalues and eigenvectors with first element removed

### Matrix Operations

#### `inverse_Lap_hor(∇ₕ²)`

Compute the inverse of a horizontal Laplacian matrix using QR decomposition.

**Parameters:**
- `∇ₕ²`: Horizontal Laplacian matrix

**Returns:**
- Inverse matrix `H = R⁻¹ * Qᵀ`

#### `InverseLaplace` Struct

Efficient representation of the inverse Laplacian operator using precomputed QR factorization.

**Constructor:**
```julia
H = InverseLaplace(∇ₕ²::AbstractMatrix{T}) where T
```

**Usage:**
```julia
# Create inverse operator
H = InverseLaplace(∇ₕ²)

# Apply to vector
x = rand(size(∇ₕ², 1))
u = H(x)  # equivalent to H * x
```


### Numerical Differentiation

#### `∇f(f, x)`

Compute numerical derivative using high-order finite difference schemes.

**Parameters:**
- `f::AbstractVector{T}`: Function values
- `x::AbstractVector{T}`: Uniformly spaced grid points

**Returns:**
- `Vector{T}`: Numerical derivative ∂f/∂x

**Features:**
- Requires uniformly spaced grid

**Example:**
```julia
x = 0:0.1:2π
f = sin.(x)
df_dx = ∇f(f, x)  # ≈ cos.(x)
```

### Sparse Matrix Construction

#### `field_to_spdiagm(U; k=0, order=:col, dims=nothing, scale=identity, pad=:error)`

Convert a 2D field/matrix into a sparse diagonal matrix.

**Parameters:**
- `U::AbstractMatrix`: Input 2D field
- `k::Integer`: Diagonal offset (0 for main diagonal)
- `order::Symbol`: Vectorization order (`:col` or `:row`)
- `dims::Union{Nothing,Tuple{Int,Int}}`: Output matrix dimensions
- `scale::Function`: Scaling function applied to elements
- `pad::Symbol`: Behavior when vector is too long (`:error`, `:trim`, `:zero`, `:wrap`)

**Returns:**
- `SparseMatrixCSC`: Sparse diagonal matrix

#### `spdiag_to_field(S, m, n; k=0, order=:col)`

Inverse operation of `field_to_spdiagm` - extract diagonal as a 2D field.

**Parameters:**
- `S::SparseMatrixCSC`: Sparse matrix
- `m::Int, n::Int`: Dimensions of output field
- `k::Integer`: Diagonal offset
- `order::Symbol`: Reshaping order

**Returns:**
- `Matrix`: 2D field reconstructed from diagonal

#### `DiagM(U; k=0, order=:col, sparse=true, dims=nothing, scale=identity, pad=:error)`

Flexible diagonal matrix constructor from 2D arrays.

**Parameters:**
- `U::AbstractMatrix`: Input 2D array
- `sparse::Bool`: Return sparse (`true`) or dense (`false`) matrix
- Other parameters same as `field_to_spdiagm`

**Returns:**
- `SparseMatrixCSC` or `Matrix`: Diagonal matrix (sparse or dense)

**Example:**
```julia
# Create sparse diagonal matrix from 2D field
U = rand(10, 10)
S_sparse = DiagM(U; sparse=true)
S_dense = DiagM(U; sparse=false)
```

## Usage Patterns

### Eigenvalue Analysis Workflow
```julia
# Solve eigenvalue problem
λs, χ = eigen(A, B)

# Sort by magnitude (descending)
λs_sorted, χ_sorted = sort_evals(λs, χ, "M")

# Remove spurious modes
λs_clean, χ_clean = remove_spurious(λs_sorted, χ_sorted)

# Display results
print_evals(λs_clean)
```

### Inverse Laplacian Operations
```julia
# Setup inverse operator
∇ₕ² = build_horizontal_laplacian()  # your matrix
H = InverseLaplace(∇ₕ²)

# Solve Poisson equation: ∇ₕ²u = f
f = rand(size(∇ₕ², 1))
u = H(f)  # u = ∇ₕ²⁻¹ * f
```

### Numerical Differentiation
```julia
# Setup grid and function
x = range(0, 2π, length=100)
f = sin.(x)

# Compute derivative
df_dx = ∇f(f, collect(x))
```

## Notes

- All eigenvalue functions work with complex eigenvalues
- Sparse matrix operations are optimized for memory efficiency
- Numerical differentiation requires uniformly spaced grids
- QR-based inverse operators are suitable for well-conditioned matrices