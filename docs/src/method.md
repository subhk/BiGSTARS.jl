# Eigenvalue Solver Documentation

A unified Julia interface for solving generalized eigenvalue problems using multiple methods with adaptive shift selection.

## Overview

This package provides a robust solution for the generalized eigenvalue problem:

```math
A x = \lambda B x
```

It implements three different solver methods with automatic shift selection and convergence verification:
- **Arnoldi Method** ([ArnoldiMethod.jl](https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl))
- **ARPACK Method** ([Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl)) 
- **Krylov Method** ([KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl))

## Core Components

### Data Structures

#### `SolverConfig`
Configuration parameters for eigenvalue solvers.

```julia
@kwdef struct SolverConfig
    method::Symbol = :Krylov          # Solver method (:Arnoldi, :Arpack, :Krylov)
    σ₀::Float64                       # Initial shift value (required)
    which::Symbol = :LM               # Which eigenvalues to find (:LM, :LR, :SR, etc.)
    nev::Int = 1                      # Number of eigenvalues to compute
    maxiter::Int = 300                # Maximum iterations
    tol::Float64 = 1e-12              # Convergence tolerance
    sortby::Symbol = :M               # Sort eigenvalues by (:R, :I, :M)
    n_tries::Int = 8                  # Number of retry attempts
    Δσ₀::Float64 = 0.2                # Initial shift increment
    incre::Float64 = 1.2              # Increment growth factor
    ϵ::Float64 = 1e-5                 # Successive eigenvalue tolerance
    krylovdim::Int = 200              # Krylov subspace dimension (Krylov method only)
end
```

#### `EigenSolver`
Main solver object for generalized eigenvalue problems.

```julia
mutable struct EigenSolver
    A::AbstractMatrix                 # Left-hand side matrix
    B::AbstractMatrix                 # Right-hand side matrix
    config::SolverConfig             # Solver configuration
    results::Union{SolverResults, Nothing}  # Latest results
end
```

#### `SolverResults`
Contains comprehensive results from eigenvalue computation.

```julia
struct SolverResults
    eigenvalues::Vector{ComplexF64}   # Computed eigenvalues
    eigenvectors::Matrix{ComplexF64}  # Corresponding eigenvectors
    converged::Bool                   # Whether computation converged
    method_used::Symbol               # Method that produced the result
    final_shift::Float64              # Final shift value used
    iterations::Int                   # Total iterations
    solve_time::Float64               # Wall-clock time for solution
    history::ConvergenceHistory       # Detailed convergence information
end
```

## Usage Examples

### Basic Usage (Object-Oriented Interface)

```julia
using LinearAlgebra

# Create test matrices
n = 100
A = rand(n, n); A = A + A'  # Symmetric matrix
B = Matrix(I, n, n)         # Identity matrix

# Create solver and solve
solver = EigenSolver(A, B; σ₀=1.0, method=:Arnoldi, nev=5)
solve!(solver)

# Extract results
λ, Χ = get_results(solver)
print_summary(solver)
```

### Functional Interface (Backwards Compatible)

```julia
# Direct solve without creating solver object
λ, Χ = solve_eigenvalue_problem(A, B; method=:Arpack, σ₀=1.0, nev=3)

# Legacy method-specific functions
λ, Χ = solve_arnoldi(A, B; σ₀=1.0, nev=5, tol=1e-10)
λ, Χ = solve_arpack(A, B; σ₀=2.0, which=:LM, maxiter=500)
λ, Χ = solve_krylov(A, B; σ₀=1.5, krylovdim=150)
```

### Advanced Configuration

```julia
# Custom configuration
config = SolverConfig(
    method = :Krylov,
    σ₀ = 2.5, 
    which = :LR,
    nev = 10,
    maxiter = 500,
    tol = 1e-14,
    krylovdim = 200,
    n_tries = 12,
    ϵ = 1e-6
)

solver = EigenSolver(A, B, config)
solve!(solver; verbose=true)
```

### Method Comparison

```julia
# Compare all available methods
solver = EigenSolver(A, B; σ₀=1.0, nev=3)
results = compare_methods!(solver; methods=[:Arnoldi, :Arpack, :Krylov])

# Compare specific methods
results = compare_methods!(solver; methods=[:Arnoldi, :Krylov], verbose=true)
```

## Solver Methods

### Arnoldi Method
- **Best for:** General sparse matrices, moderate problem sizes
- **Strengths:** Reliable, good theoretical foundation
- **Key parameters:** `which`, `nev`, `maxiter`, `tol`

### ARPACK Method  
- **Best for:** Large sparse matrices, production use
- **Strengths:** Battle-tested, widely used, efficient for large problems
- **Key parameters:** `which`, `nev`, `maxiter`, `tol`

### Krylov Method
- **Best for:** Modern problems, flexible interface
- **Strengths:** Clean interface, good for research
- **Key parameters:** `which`, `maxiter`, `krylovdim`

## Shift-and-Invert Transformation

The solver uses the shift-and-invert technique to target eigenvalues near a specified shift value σ:

```math
(A - \sigma B)^{-1} B x = \mu x, \quad \mu = (\lambda - \sigma)^{-1}
```

### Advantages
- **Selective targeting:** Eigenvalues λ closest to σ correspond to largest magnitudes of μ
- **Accelerated convergence:** Inverting the shifted operator compresses the spectrum
- **Numerical stability:** Better conditioning for interior eigenvalues

### Linear Map Construction

The `construct_linear_map` function creates the shift-and-invert operator:

```julia
function construct_linear_map(A, B)
    ShiftAndInvert(factorize(A), B, Vector{eltype(A)}(undef, size(A,1))) |>
    M -> LinearMap{eltype(A)}(M, size(A,1), ismutating=true)
end
```

## Adaptive Shift Selection

The solver automatically tries multiple shift values if the initial shift fails:

1. **Generate shift attempts:** Creates upward and downward shifts from σ₀
2. **Progressive increments:** Uses geometric progression with factor `incre`
3. **Convergence checking:** Verifies successive eigenvalues differ by less than `ϵ`
4. **Fallback strategy:** Continues until convergence or maximum attempts reached

```julia
# Shift generation example
Δσs_up = [Δσ₀ * incre^(i-1) * abs(σ₀) for i in 1:n_tries]
Δσs_dn = [-δ for δ in Δσs_up]
σ_attempts = [σ₀ + δ for δ in vcat(Δσs_up, Δσs_dn)]
```

## API Reference

### Core Functions

#### `EigenSolver(A, B, config::SolverConfig)`
Create a new eigenvalue solver.

#### `EigenSolver(A, B; σ₀::Float64, kwargs...)`
Convenience constructor with keyword arguments.

#### `solve!(solver::EigenSolver; verbose::Bool=true)`
Solve the eigenvalue problem with adaptive shift selection.

#### `get_results(solver::EigenSolver)`
Extract eigenvalues and eigenvectors from solved system.
- **Returns:** `(λ, Χ)` tuple of eigenvalues and eigenvectors
- **Throws:** `ArgumentError` if solver hasn't converged

#### `print_summary(solver::EigenSolver)`
Print detailed summary of solver results.

#### `compare_methods!(solver::EigenSolver; methods, verbose=true)`
Compare multiple solver methods on the same problem.
- **Returns:** `Dict{Symbol, SolverResults}` with results for each method

### Utility Functions

#### `sort_eigenvalues!(λ, Χ, by::Symbol; rev::Bool=true)`
Sort eigenvalues and eigenvectors in-place.
- `:R` → Sort by real part
- `:I` → Sort by imaginary part  
- `:M` → Sort by magnitude

#### `get_method_info(method::Symbol)`
Get detailed information about a specific solver method.

#### `show_example_usage()`
Display comprehensive usage examples.

### Legacy Functions

```julia
solve_eigenvalue_problem(A, B; method=:Arnoldi, σ₀, kwargs...)
solve_arnoldi(A, B; σ₀, kwargs...)
solve_arpack(A, B; σ₀, kwargs...)  
solve_krylov(A, B; σ₀, kwargs...)
```

## Configuration Parameters

| Parameter    | Default   | Description                     |
|--------------|-----------|---------------------------------|
| `method`     | `:Krylov` | Solver method                   |
| `σ₀`         | Required  | Initial shift value             |
| `which`      | `:LM`     | Target eigenvalues              |
| `nev`        | `1`       | Number of eigenvalues           |
| `maxiter`    | `300`     | Maximum iterations              |
| `tol`        | `1e-12`   | Convergence tolerance           |
| `sortby`     | `:M`      | Sorting criterion               |
| `n_tries`    | `8`       | Retry attempts                  |
| `Δσ₀`        | `0.2`     | Initial shift increment         |
| `incre`      | `1.2`     | Increment growth factor         |
| `ϵ`          | `1e-5`    | Successive eigenvalue tolerance |
| `krylovdim`  | `200`     | Krylov subspace dimension       |

## Error Handling

The solver includes comprehensive error handling:

- **Input validation:** Checks matrix dimensions and properties
- **Convergence monitoring:** Tracks failed attempts and reasons
- **Graceful degradation:** Continues with alternative shifts if initial attempts fail
- **Detailed diagnostics:** Provides convergence history and error messages

## Performance Considerations

- **Matrix factorization:** The shift-and-invert method requires factorizing `(A - σB)`
- **Memory usage:** Eigenvectors are stored as dense matrices
- **Convergence rate:** Depends on spectral properties and shift selection
- **Method selection:** ARPACK typically fastest for large sparse problems
