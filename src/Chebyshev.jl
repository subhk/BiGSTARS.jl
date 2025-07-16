using LinearAlgebra
using Printf

# Coordinate transformation functions
"""
`cheb_coord_transform`: transform the derivative operator from a domain of ζ ∈ [-1, 1] → x ∈ [0, L] via
x = (1.0 + ζ) / 2.0 * L

Input:
 D¹: First-order Chebyshev derivative in ζ
 D²: Second-order Chebyshev derivative in ζ
 d¹: Transformed coefficient
 d²: Transformed coefficient

Output:
 Dₓ : First-order Chebyshev derivative in x
 Dₓₓ: Second-order Chebyshev derivative in x
"""
function cheb_coord_transform(D¹, D², d¹, d²)
    Dₓ = zeros(size(D¹, 1), size(D¹, 2))
    mul!(Dₓ, diagm(d¹), D¹)
    
    Dₓₓ = zeros(size(D², 1), size(D², 2))
    tmp₁ = zeros(size(D², 1), size(D², 2))
    mul!(Dₓₓ, diagm(d²), D¹)
    mul!(tmp₁, diagm(d¹ .^ 2), D²)
    Dₓₓ = Dₓₓ + tmp₁
    
    return Dₓ, Dₓₓ
end

"""
`cheb_coord_transform_ho`: transform the derivative operator from a domain of ζ ∈ [-1, 1] → x ∈ [0, L] via
x = (1.0 + ζ) / 2.0 * L for higher-order derivatives

Input:
 D¹: First-order Chebyshev derivative in ζ
 D²: Second-order Chebyshev derivative in ζ
 D³: Third-order Chebyshev derivative in ζ
 D⁴: Fourth-order Chebyshev derivative in ζ
 d¹: Transformed coefficient
 d²: Transformed coefficient
 d³: Transformed coefficient
 d⁴: Transformed coefficient

Output:
 Dₓₓₓ : Third-order Chebyshev derivative in x
 Dₓₓₓₓ: Fourth-order Chebyshev derivative in x
"""
function cheb_coord_transform_ho(D¹, D², D³, D⁴, d¹, d², d³, d⁴)
    Dₓₓₓ = zeros(size(D², 1), size(D², 2))
    tmp₁ = zeros(size(D², 1), size(D², 2))
    tmp₂ = zeros(size(D², 1), size(D², 2))
    
    mul!(Dₓₓₓ, diagm(d³), D¹)
    mul!(tmp₁, diagm(d¹), diagm(d²))
    mul!(tmp₂, diagm(d¹ .^ 3), D³)
    Dₓₓₓ = Dₓₓₓ + 3tmp₁ * D² + tmp₂
    
    Dₓₓₓₓ = zeros(size(D², 1), size(D², 2))
    mul!(Dₓₓₓₓ, diagm(d⁴), D¹)
    mul!(tmp₁, diagm(d¹), diagm(d³))
    mul!(tmp₂, diagm(d¹ .^ 2), diagm(d²))
    tmp₁ = 4tmp₁ + 3.0 * diagm(d² .^ 2)
    tmp₂ = 6.0 * diagm(d¹ .^ 2) * diagm(d²)
    Dₓₓₓₓ = Dₓₓₓₓ + tmp₁ * D² + 6tmp₂ * D³ + diagm(d¹ .^ 4) * D⁴
    
    return Dₓₓₓ, Dₓₓₓₓ
end

# Domain transformation functions
# [-1, 1] ↦ [-L, L]
function MinusLtoPlusL_transform(x, L::Float64)
    z = @. x * L
    Δ1 = @. 1.0 / L + 0.0 * z
    Δ2 = @. 0.0 * z
    return z, Δ1, Δ2
end

# [-1, 1] ↦ [-L, 0]
function MinusLtoZero_transform(x, L::Float64)
    z = @. -1.0 * (1.0 - x) /2.0 * L
    Δ1 = @. 2.0 / L + 0.0 * z
    Δ2 = @. 0.0 * z
    return z, Δ1, Δ2
end

# [-1, 1] ↦ [0, L]
function zerotoL_transform(x, L::Float64)
    z = @. (1.0 + x) / 2.0 * L
    Δ1 = @. 2.0 / L + 0.0 * z
    Δ2 = @. 0.0 * z
    return z, Δ1, Δ2
end

# [0, 2π] → [0, L]
function transform_02π_to_0L(x, L::Float64)
    z = @. x/2π * L
    Δ1 = @. 2π / L + 0.0 * z
    Δ2 = @. 0.0 * z
    return z, Δ1, Δ2
end

function zerotoL_transform_ho(x, L::Float64)
    z = @. (1.0 + x) / 2.0 * L
    Δ1 = @. 2.0 / L + 0.0 * z
    Δ2 = @. 0.0 * z
    Δ3 = @. 0.0 * z
    Δ4 = @. 0.0 * z
    return z, Δ1, Δ2, Δ3, Δ4
end

# [-1, 1] ↦ [0, 1]
function zerotoone_transform(x)
    z = @. (1.0 + x) / 2.0
    Δ1 = @. 2.0 + 0.0 * z
    Δ2 = @. 0.0 * z
    return z, Δ1, Δ2
end

function chebder_transform(x, D¹, D², fun_transform, kwargs...)
    z, d¹, d² = fun_transform(x, kwargs...)
    Dₓ, Dₓₓ = cheb_coord_transform(D¹, D², d¹, d²)
    return z, Dₓ, Dₓₓ
end

function chebder_transform_ho(x, D¹, D², D³, D⁴, fun_transform, kwargs...)
    z, d¹, d², d³, d⁴ = fun_transform(x, kwargs...)
    Dₓₓₓ, Dₓₓₓₓ = cheb_coord_transform_ho(D¹, D², D³, D⁴, d¹, d², d³, d⁴)
    return z, Dₓₓₓ, Dₓₓₓₓ
end

"""
    Chebyshev(n::Int, domain=2.0, orders::Vector{Int}=Int[])

Optimized Chebyshev differentiation with coordinate transformations.

# Domain Specifications
- `L::Float64`: Domain [0, L]
- `[a, b]`: General domain [a, b]
- `[-L, L]`: Symmetric domain [-L, L]

# Usage
```julia
# Basic usage with domain length
cd = Chebyshev(16, 4.0)  # 16 points on domain [0, 4.0]

# Domain as interval
cd = Chebyshev(16, [0, 4])    # Domain [0, 4]
cd = Chebyshev(16, [-2, 2])   # Domain [-2, 2]
cd = Chebyshev(16, [1, 5])    # Domain [1, 5]

# Pre-compute specific derivative orders
cd = Chebyshev(16, [0, 4], [1, 2, 4])  # Pre-compute 1st, 2nd, and 4th derivatives
df = cd(f, 1)      # First derivative (already computed)
d2f = cd(f, 2)     # Second derivative (already computed)
d4f = cd(f, 4)     # Fourth derivative (already computed)
```
"""
struct Chebyshev
    n::Int
    domain::Vector{Float64}
    L::Float64  # Domain length for backward compatibility
    grid::Vector{Float64}
    dx::Float64
    matrices::Dict{Int, Matrix{Float64}}
    base_matrices::Dict{Int, Matrix{Float64}}  # Base matrices on [-1, 1]
    
    function Chebyshev(n::Int, domain=2.0, orders::Vector{Int}=Int[])
        n < 3 && throw(ArgumentError("Grid size must be ≥ 3"))
        any(o -> o < 0, orders) && throw(ArgumentError("All derivative orders must be ≥ 0"))
        
        # Parse domain specification
        domain_vec = parse_domain_opt(domain)
        a, b = domain_vec[1], domain_vec[2]
        L = b - a  # Domain length
        
        # Create Chebyshev-Gauss-Lobatto points on the specified domain
        grid, x_base = create_chebyshev_grid_opt(n, domain_vec)
        dx = L / (n - 1)  # Approximate spacing
        
        matrices = Dict{Int, Matrix{Float64}}()
        base_matrices = Dict{Int, Matrix{Float64}}()
        
        # Pre-compute requested derivative matrices using simple approach
        for order in orders
            matrices[order] = compute_derivative_matrix_simple(n, order, domain_vec)
        end
        
        new(n, domain_vec, L, grid, dx, matrices, base_matrices)
    end
end

function parse_domain_opt(domain)
    if isa(domain, Real)
        domain <= 0 && throw(ArgumentError("Domain length must be positive"))
        return [0.0, Float64(domain)]
    elseif isa(domain, AbstractVector) && length(domain) == 2
        domain_vec = [Float64(domain[1]), Float64(domain[2])]
        domain_vec[1] >= domain_vec[2] && throw(ArgumentError("Domain bounds must satisfy a < b"))
        return domain_vec
    else
        throw(ArgumentError("Domain must be a positive number or [a, b] interval"))
    end
end

function create_chebyshev_grid_opt(n::Int, domain::Vector{Float64})
    a, b = domain[1], domain[2]
    
    # Chebyshev-Gauss-Lobatto points on [-1, 1]
    k = 0:n-1
    x_base = cos.(π * k / (n - 1))
    
    # Transform to [a, b]: x = (b-a)/2 * ζ + (b+a)/2
    grid = @. (b - a) / 2.0 * x_base + (b + a) / 2.0
    
    return grid, x_base
end

function compute_derivative_matrix_optimized_opt(n::Int, order::Int, domain::Vector{Float64}, base_matrices::Dict{Int, Matrix{Float64}})
    order == 0 && return Matrix{Float64}(I, n, n)
    
    a, b = domain[1], domain[2]
    
    # For all cases, use the simple and robust transformation
    # Compute base differentiation matrix on [-1, 1]
    k = 0:n-1
    ζ = cos.(π * k / (n - 1))
    D = compute_base_derivative_matrix_opt(ζ, n, order)
    
    # Apply domain transformation scaling: x = (b-a)/2 * ζ + (b+a)/2
    # Chain rule: d/dx = (2/(b-a)) * d/dζ
    transform_factor = (2.0 / (b - a))^order
    return transform_factor * D
end

function compute_base_derivative_matrix_opt(x::Vector{Float64}, n::Int, order::Int)
    # Compute first derivative matrix
    D = zeros(n, n)
    
    # Weight vector for Chebyshev points
    c = [i == 1 || i == n ? 2.0 : 1.0 for i in 1:n]
    c .*= [(-1.0)^(i-1) for i in 1:n]
    
    # Off-diagonal entries
    for i in 1:n
        for j in 1:n
            if i != j
                D[i, j] = (c[i] / c[j]) / (x[i] - x[j])
            end
        end
    end
    
    # Diagonal entries
    for i in 1:n
        D[i, i] = -sum(D[i, j] for j in 1:n if j != i)
    end
    
    # For higher orders, use the stable iterative formula
    if order == 1
        return D
    end
    
    # Iterative computation for higher derivatives
    # This implements the stable algorithm from Weideman & Reddy
    C = c * c'  # Coefficient matrix
    Z = zeros(n, n)
    for i in 1:n
        for j in 1:n
            if i != j
                Z[i, j] = 1.0 / (x[i] - x[j])
            end
        end
    end
    
    D_curr = copy(D)
    for ell in 2:order
        diag_D = diag(D_curr)
        D_new = zeros(n, n)
        
        for i in 1:n
            for j in 1:n
                if i != j
                    D_new[i, j] = ell * Z[i, j] * (C[i, j] * diag_D[i] - D_curr[i, j])
                end
            end
        end
        
        # Diagonal correction
        for i in 1:n
            D_new[i, i] = -sum(D_new[i, j] for j in 1:n if j != i)
        end
        
        D_curr = D_new
    end
    
    return D_curr
end

# Callable interface
(cd::Chebyshev)(f::AbstractVector, order::Int=1) = differentiate_opt(cd, f, order)

function compute_derivative_matrix_simple(n::Int, order::Int, domain::Vector{Float64})
    order == 0 && return Matrix{Float64}(I, n, n)
    
    a, b = domain[1], domain[2]
    
    # Compute base differentiation matrix on [-1, 1]
    k = 0:n-1
    ζ = cos.(π * k / (n - 1))
    D = compute_base_derivative_matrix_opt(ζ, n, order)
    
    # Apply domain transformation scaling: x = (b-a)/2 * ζ + (b+a)/2
    # Chain rule: d/dx = (2/(b-a)) * d/dζ
    transform_factor = (2.0 / (b - a))^order
    return transform_factor * D
end

function derivative_matrix_opt(cd::Chebyshev, order::Int)
    order < 0 && throw(ArgumentError("Derivative order must be ≥ 0"))
    haskey(cd.matrices, order) && return cd.matrices[order]
    
    # Compute on-demand using simple method
    cd.matrices[order] = compute_derivative_matrix_simple(cd.n, order, cd.domain)
    return cd.matrices[order]
end

function differentiate_opt(cd::Chebyshev, f::AbstractVector, order::Int=1)
    length(f) != cd.n && throw(ArgumentError("Function length must match grid size"))
    return derivative_matrix_opt(cd, order) * f
end

# Convenience function
chebyshev_diff(n::Int, domain=2.0, orders::Vector{Int}=Int[], order::Int=1) = 
    (cd = Chebyshev(n, domain, orders); (cd.grid, derivative_matrix_opt(cd, order)))

# Utility functions
Base.show(io::IO, cd::Chebyshev) = 
    print(io, "Chebyshev(n=$(cd.n), domain=$(cd.domain), cached=$(sort(collect(keys(cd.matrices)))))")

clear_cache_opt!(cd::Chebyshev) = (empty!(cd.matrices); empty!(cd.base_matrices); cd)

memory_usage_opt(cd::Chebyshev) = 
    (sum(length(m) for m in values(cd.matrices)) + sum(length(m) for m in values(cd.base_matrices))) * sizeof(Float64) / 1024^2

# Demo
if abspath(PROGRAM_FILE) == @__FILE__
    println("Optimized Chebyshev Differentiation")
    println("=" ^ 35)
    
    # Test on [0, L] domain
    cd1 = Chebyshev(16, 4.0, [1, 2, 3, 4])
    x1 = cd1.grid
    f1 = sin.(π * x1 / 4.0)
    
    println("Testing sin(πx/L) on [0, 4]:")
    L = 4.0
    for order in 1:4
        df_computed = cd1(f1, order)
        if order == 1
            df_exact = (π/L) * cos.(π * x1 / L)
        elseif order == 2
            df_exact = -(π/L)^2 * sin.(π * x1 / L)
        elseif order == 3
            df_exact = -(π/L)^3 * cos.(π * x1 / L)
        else
            df_exact = (π/L)^4 * sin.(π * x1 / L)
        end
        error = maximum(abs.(df_computed - df_exact))
        println("Order $order: Error = $(error)")
    end
    
    # Test symmetric domain [-L, L]
    println("\nTesting x⁴ on [-2, 2]:")
    cd2 = Chebyshev(16, [-2.0, 2.0], [1, 2])
    x2 = cd2.grid
    f2 = x2.^4
    
    df_exact = 4 * x2.^3
    d2f_exact = 12 * x2.^2
    
    println("1st: max error = $(maximum(abs.(cd2(f2,1) - df_exact)))")
    println("2nd: max error = $(maximum(abs.(cd2(f2,2) - d2f_exact)))")
    
    println("\nMemory usage: $(memory_usage_opt(cd1)) MB")
end