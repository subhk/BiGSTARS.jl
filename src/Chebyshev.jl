using LinearAlgebra, Printf

"""
    ChebyshevDiff(n::Int, domain=2.0, orders::Vector{Int}=Int[])

Chebyshev differentiation on specified domain.

# Domain Specifications
- `L::Float64`: Domain [0, L]
- `[a, b]`: General domain [a, b]
- `[-L, L]`: Symmetric domain [-L, L]

# Usage
```julia
# Basic usage with domain length
cd = ChebyshevDiff(16, 4.0)  # 16 points on domain [0, 4.0]

# Domain as interval
cd = ChebyshevDiff(16, [0, 4])    # Domain [0, 4]
cd = ChebyshevDiff(16, [-2, 2])   # Domain [-2, 2]
cd = ChebyshevDiff(16, [1, 5])    # Domain [1, 5]

# Pre-compute specific derivative orders
cd = ChebyshevDiff(16, [0, 4], [1, 2, 4])  # Pre-compute 1st, 2nd, and 4th derivatives
df = cd(f, 1)      # First derivative (already computed)
d2f = cd(f, 2)     # Second derivative (already computed)
d4f = cd(f, 4)     # Fourth derivative (already computed)
```
"""
struct ChebyshevDiff
    n::Int
    domain::Vector{Float64}
    L::Float64  # Domain length for backward compatibility
    grid::Vector{Float64}
    dx::Float64
    matrices::Dict{Int, Matrix{Float64}}
    
    function ChebyshevDiff(n::Int, domain=2.0, orders::Vector{Int}=Int[])
        n < 3 && throw(ArgumentError("Grid size must be ≥ 3"))
        any(o -> o < 0, orders) && throw(ArgumentError("All derivative orders must be ≥ 0"))
        
        # Parse domain specification
        if isa(domain, Real)
            # Single number means [0, domain]
            domain_vec = [0.0, Float64(domain)]
            domain <= 0 && throw(ArgumentError("Domain length must be positive"))
        elseif isa(domain, AbstractVector) && length(domain) == 2
            # Domain specified as [a, b]
            domain_vec = [Float64(domain[1]), Float64(domain[2])]
            domain_vec[1] >= domain_vec[2] && throw(ArgumentError("Domain bounds must satisfy a < b"))
        else
            throw(ArgumentError("Domain must be a positive number or [a, b] interval"))
        end
        
        a, b = domain_vec[1], domain_vec[2]
        L = b - a  # Domain length
        
        # Create Chebyshev-Gauss-Lobatto points on [a, b]
        k = 0:1:n-1
        x_cheb = sin.(π * (n-1 .- 2 * reverse(k)) / (2 * (n-1)))  # Points on [-1, 1]
        
        # Transform to [a, b]: x = (b-a)/2 * ζ + (b+a)/2
        grid = @. (b - a) / 2.0 * x_cheb + (b + a) / 2.0
        dx = L / (n - 1)  # Approximate spacing
        
        matrices = Dict{Int, Matrix{Float64}}()
        
        # Pre-compute requested derivative matrices
        for order in orders
            matrices[order] = _compute_matrix(n, order, domain_vec)
        end
        
        new(n, domain_vec, L, grid, dx, matrices)
    end
end

# Callable interface
(cd::ChebyshevDiff)(f::AbstractVector, order::Int=1) = differentiate(cd, f, order)

function derivative_matrix(cd::ChebyshevDiff, order::Int)
    order < 0 && throw(ArgumentError("Derivative order must be ≥ 0"))
    haskey(cd.matrices, order) && return cd.matrices[order]
    
    cd.matrices[order] = _compute_matrix(cd.n, order, cd.domain)
    return cd.matrices[order]
end

function differentiate(cd::ChebyshevDiff, f::AbstractVector, order::Int=1)
    length(f) != cd.n && throw(ArgumentError("Function length must match grid size"))
    return derivative_matrix(cd, order) * f
end

function _compute_matrix(n::Int, order::Int, domain::Vector{Float64})
    order == 0 && return Matrix{Float64}(I, n, n)
    
    a, b = domain[1], domain[2]
    
    # Compute base Chebyshev differentiation matrices on [-1, 1]
    x_cheb, D_base = _compute_base_matrices(n, order)
    
    # Transform from [-1, 1] to [a, b]
    # Transformation: x = (b-a)/2 * ζ + (b+a)/2, where ζ ∈ [-1, 1], x ∈ [a, b]
    # Chain rule: d/dx = (dζ/dx) * d/dζ = (2/(b-a)) * d/dζ
    transform_factor = (2.0 / (b - a))^order
    
    # Apply transformation
    D_transformed = transform_factor * D_base
    
    return D_transformed
end

function _compute_base_matrices(n::Int, max_order::Int)
    # Compute Chebyshev-Gauss-Lobatto points on [-1, 1]
    k = 0:1:n-1
    θ = k * π / (n - 1)
    x = sin.(π * (n - 1 .- 2 * reverse(k)) / (2 * (n - 1)))
    
    # First derivative matrix using standard Chebyshev method
    D1 = _compute_first_derivative_matrix(x, n)
    
    if max_order == 1
        return x, D1
    end
    
    # For higher orders, compute iteratively
    # This is a simplified approach - in practice, you'd want the full
    # recursive formulation from the original code
    D_current = D1
    for order in 2:max_order
        if order == 2
            D_current = D1 * D1  # Second derivative
        else
            D_current = D1 * D_current  # Higher derivatives (approximate)
        end
    end
    
    return x, D_current
end

function _compute_first_derivative_matrix(x::Vector{Float64}, n::Int)
    D = zeros(n, n)
    
    # Compute weights
    c = ones(n)
    c[1] = 2.0
    c[end] = 2.0
    for i in 2:n-1
        c[i] = (-1.0)^(i-1)
    end
    
    # Fill differentiation matrix
    for i in 1:n
        for j in 1:n
            if i != j
                D[i, j] = (c[i] / c[j]) * (-1.0)^(i + j) / (x[i] - x[j])
            end
        end
    end
    
    # Diagonal entries (negative sum of off-diagonal elements)
    for i in 1:n
        D[i, i] = -sum(D[i, j] for j in 1:n if j != i)
    end
    
    return D
end

# More accurate implementation using the original algorithm structure
function _compute_matrix_accurate(n::Int, order::Int, domain::Vector{Float64})
    order == 0 && return Matrix{Float64}(I, n, n)
    
    a, b = domain[1], domain[2]
    
    # Create base Chebyshev points and matrices on [-1, 1]
    k = 0:1:n-1
    θ = k * π / (n - 1)
    x_cheb = sin.(π * (n - 1 .- 2 * reverse(k)) / (2 * (n - 1)))
    
    # Compute first derivative matrix on [-1, 1]
    D1_base = _compute_accurate_derivative_matrix(x_cheb, n, 1)
    
    if order == 1
        # Transform first derivative: [a, b] domain
        # x = (b-a)/2 * ζ + (b+a)/2, so dx/dζ = (b-a)/2, and d/dx = 2/(b-a) * d/dζ
        return (2.0 / (b - a)) * D1_base
    elseif order == 2
        # Second derivative
        D2_base = _compute_accurate_derivative_matrix(x_cheb, n, 2)
        return (2.0 / (b - a))^2 * D2_base
    else
        # Higher order derivatives
        D_base = _compute_accurate_derivative_matrix(x_cheb, n, order)
        return (2.0 / (b - a))^order * D_base
    end
end

function _compute_accurate_derivative_matrix(x::Vector{Float64}, n::Int, order::Int)
    if order == 1
        return _compute_first_derivative_matrix(x, n)
    end
    
    # For higher orders, use the iterative method from the original code
    # This is simplified - the full implementation would follow the original algorithm
    D1 = _compute_first_derivative_matrix(x, n)
    
    # Compute coefficient vectors
    c = (-1.0) .^ (0:n-1)
    c[1] *= 2
    c[end] *= 2
    C = c ./ c'
    
    # Compute distance matrix
    Z = zeros(n, n)
    for i in 1:n
        for j in 1:n
            if i != j
                Z[i, j] = 1.0 / (x[i] - x[j])
            end
        end
    end
    
    # Iteratively compute higher derivatives
    D = copy(D1)
    for ell in 2:order
        diag_D = diag(D)
        D = ell * Z .* (C .* repeat(diag_D, 1, n) .- D)
        
        # Diagonal correction
        for i in 1:n
            D[i, i] = -sum(D[i, j] for j in 1:n if j != i)
        end
    end
    
    return D
end

# Update the main computation function to use accurate method
function _compute_matrix(n::Int, order::Int, domain::Vector{Float64})
    return _compute_matrix_accurate(n, order, domain)
end

# Convenience function
chebyshev_diff(n::Int, domain=2.0, orders::Vector{Int}=Int[], order::Int=1) = 
    (cd = ChebyshevDiff(n, domain, orders); (cd.grid, derivative_matrix(cd, order)))

# Utility functions
Base.show(io::IO, cd::ChebyshevDiff) = 
    print(io, "ChebyshevDiff(n=$(cd.n), domain=$(cd.domain), cached=$(sort(collect(keys(cd.matrices)))))")

clear_cache!(cd::ChebyshevDiff) = (empty!(cd.matrices); cd)

memory_usage(cd::ChebyshevDiff) = 
    sum(length(m) for m in values(cd.matrices)) * sizeof(Float64) / 1024^2

# Testing
function test_chebyshev_diff(n::Int=16, domain=[0.0, 2.0], orders::Vector{Int}=[1,2])
    cd = ChebyshevDiff(n, domain, orders)
    
    # Test with polynomial function (should be exact for low-degree polynomials)
    # Use x^2 which has known derivatives
    f = cd.grid.^2
    
    df_num = cd(f, 1)
    df_exact = 2 * cd.grid
    d2f_num = cd(f, 2)
    d2f_exact = 2 * ones(length(cd.grid))
    
    err1 = maximum(abs.(df_num - df_exact))
    err2 = maximum(abs.(d2f_num - d2f_exact))
    
    println("n=$n, domain=$domain, orders=$orders: 1st deriv error = $(err1), 2nd deriv error = $(err2)")
    return err1 < 1e-10 && err2 < 1e-10
end

# Demo
if abspath(PROGRAM_FILE) == @__FILE__
    # Test on standard domain [0, 2] using scalar input
    println("Testing on domain [0, 2] using scalar input:")
    cd1 = ChebyshevDiff(16, 2.0, [1, 2])
    x1 = cd1.grid
    f1 = x1.^3  # Cubic polynomial
    
    println("Derivatives of x³:")
    df_exact = 3 * x1.^2
    d2f_exact = 6 * x1
    
    println("1st: max error = $(maximum(abs.(cd1(f1,1) - df_exact)))")
    println("2nd: max error = $(maximum(abs.(cd1(f1,2) - d2f_exact)))")
    
    # Test on domain [0, 4] using vector input
    println("\nTesting on domain [0, 4] using vector input:")
    cd2 = ChebyshevDiff(16, [0, 4], [1, 2, 3])
    x2 = cd2.grid
    L = cd2.L
    f2 = sin.(π * x2 / L)  # Sine function
    
    println("Derivatives of sin(πx/L):")
    df_exact = (π / L) * cos.(π * x2 / L)
    d2f_exact = -(π / L)^2 * sin.(π * x2 / L)
    d3f_exact = -(π / L)^3 * cos.(π * x2 / L)
    
    println("1st: max error = $(maximum(abs.(cd2(f2,1) - df_exact)))")
    println("2nd: max error = $(maximum(abs.(cd2(f2,2) - d2f_exact)))")
    println("3rd: max error = $(maximum(abs.(cd2(f2,3) - d3f_exact)))")
    
    # Test on symmetric domain [-2, 2]
    println("\nTesting on symmetric domain [-2, 2]:")
    cd3 = ChebyshevDiff(16, [-2, 2], [1, 2])
    x3 = cd3.grid
    f3 = x3.^4  # Even polynomial
    
    println("Derivatives of x⁴:")
    df_exact = 4 * x3.^3
    d2f_exact = 12 * x3.^2
    
    println("1st: max error = $(maximum(abs.(cd3(f3,1) - df_exact)))")
    println("2nd: max error = $(maximum(abs.(cd3(f3,2) - d2f_exact)))")
    
    # Test on general domain [1, 5]
    println("\nTesting on general domain [1, 5]:")
    cd4 = ChebyshevDiff(16, [1, 5], [1, 2])
    x4 = cd4.grid
    f4 = (x4 .- 3).^2  # Shifted parabola
    
    println("Derivatives of (x-3)²:")
    df_exact = 2 * (x4 .- 3)
    d2f_exact = 2 * ones(length(x4))
    
    println("1st: max error = $(maximum(abs.(cd4(f4,1) - df_exact)))")
    println("2nd: max error = $(maximum(abs.(cd4(f4,2) - d2f_exact)))")
    
    println("\nCached matrices: $(cd2)")
    test_chebyshev_diff(16, [0.0, 2.0], [1,2])
    test_chebyshev_diff(16, [-1.0, 1.0], [1,2])
    test_chebyshev_diff(16, [1.0, 5.0], [1,2])
end