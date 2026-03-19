"""
    FourierDifferentiation

Implementation of Fourier spectral differentiation matrices.

This module provides spectrally accurate differentiation operators for periodic functions,
with elegant handling of arbitrary domains and efficient caching mechanisms.

# Features
- Spectral accuracy for smooth periodic functions
- Memory-efficient Toeplitz matrix representation
- Automatic scaling for arbitrary domains [0, L)
- Intelligent caching system
- Clean, intuitive API

# Example
```julia
# Create differentiation object on [0, 4π) with 64 points
𝒟 = FourierDiffn(64; L = 4π)

# Differentiate a function - multiple beautiful syntaxes available!
u   = sin.(𝒟.x)
∂u  = 𝒟[1] * u     # indexing syntax
∂u  = 𝒟.D₁ * u     # property syntax with subscript
∂²u = 𝒟.D₂ * u    # second derivative
```
"""
#module FourierDifferentiation

using ToeplitzMatrices: Toeplitz
using FFTW: fft, ifft

#export FourierDiff, FourierDiffn

# ═══════════════════════════════════════════════════════════════════════════════════
# Core Fourier differentiation matrix computation
# ═══════════════════════════════════════════════════════════════════════════════════

"""
    FourierDiff(n::Integer, m::Integer) -> x, D

Compute the m-th order Fourier spectral differentiation matrix on n equispaced points
in the interval [0, 2π).

# Arguments
- `n::Integer`: Number of grid points
- `m::Integer`: Derivative order (m ≥ 0)

# Returns
- `x`: Grid points in [0, 2π)
- `D`: Toeplitz differentiation matrix

# Mathematical Background
For a periodic function u(x), the Fourier differentiation matrix D satisfies:
    (D^m * u) ≈ d^m u/dx^m
with spectral accuracy for smooth functions.
"""
function FourierDiff(n::Integer, m::Integer)
    @assert n > 0 "Number of points must be positive"
    @assert m ≥ 0 "Derivative order must be non-negative"
    
    # Construct equispaced grid on [0, 2π)
    x = range(0, 2π, length = n + 1)[1:n]
    Δx = 2π / n
    
    # Compute Nyquist-related indices
    ν₁ = (n - 1) ÷ 2
    ν₂ = n ÷ 2
    
    # Dispatch on derivative order for optimal performance
    col, row = _compute_toeplitz_entries(n, m, Δx, ν₁, ν₂)
    
    return collect(x), Toeplitz(col, row)
end

# ───────────────────────────────────────────────────────────────────────────────────
# Specialized computations for different derivative orders
# ───────────────────────────────────────────────────────────────────────────────────

function _compute_toeplitz_entries(n::Integer, m::Integer, Δx::Real, ν₁::Integer, ν₂::Integer)
    if m == 0
        return _identity_operator(n)
    elseif m == 1
        return _first_derivative_operator(n, Δx, ν₁, ν₂)
    elseif m == 2
        return _second_derivative_operator(n, Δx, ν₁, ν₂)
    else
        return _higher_order_operator(n, m, ν₁, ν₂)
    end
end

function _identity_operator(n::Integer)
    col = zeros(n)
    col[1] = 1
    return col, copy(col)
end

function _first_derivative_operator(n::Integer, Δx::Real, ν₁::Integer, ν₂::Integer)
    # Alternating sign pattern for first derivative
    alternating_signs = [(-1)^k for k in 1:n-1]
    
    if iseven(n)
        # Even grid: use cotangent weights
        weights = [cot(k * Δx / 2) for k in 1:ν₂]
        entries = 0.5 * alternating_signs .* vcat(weights, -reverse(weights[1:ν₁]))
    else
        # Odd grid: use cosecant weights
        weights = [csc(k * Δx / 2) for k in 1:ν₂]
        entries = 0.5 * alternating_signs .* vcat(weights, reverse(weights[1:ν₁]))
    end
    
    col = vcat(0, entries)
    row = -col
    
    return col, row
end

function _second_derivative_operator(n::Integer, Δx::Real, ν₁::Integer, ν₂::Integer)
    # Alternating sign pattern for second derivative
    alternating_signs = -0.5 * [(-1)^k for k in 1:n-1]
    
    if iseven(n)
        # Even grid: use cosecant squared weights
        weights = [csc(k * Δx / 2)^2 for k in 1:ν₂]
        entries = alternating_signs .* vcat(weights, reverse(weights[1:ν₁]))
        diagonal_entry = -π^2 / (3 * Δx^2) - 1/6
    else
        # Odd grid: use cotangent-cosecant weights
        weights = [cot(k * Δx / 2) * csc(k * Δx / 2) for k in 1:ν₂]
        entries = alternating_signs .* vcat(weights, -reverse(weights[1:ν₁]))
        diagonal_entry = -π^2 / (3 * Δx^2) + 1/12
    end
    
    col = vcat(diagonal_entry, entries)
    row = col
    
    return col, row
end

function _higher_order_operator(n::Integer, m::Integer, ν₁::Integer, ν₂::Integer)
    # Higher-order derivatives via FFT approach
    nyquist_mode = iseven(n) ? -n ÷ 2 : 0
    
    # Construct wavenumber vector correctly
    if iseven(n)
        # Even n: [0, 1, 2, ..., n/2-1, -n/2, -n/2+1, ..., -1]
        k = 1im * vcat(0:ν₁, nyquist_mode, -(ν₁:-1:1))
    else
        # Odd n: [0, 1, 2, ..., (n-1)/2, -(n-1)/2, ..., -1]  
        k = 1im * vcat(0:ν₁, -(ν₁:-1:1))
    end
    
    # Ensure k has exactly n elements
    @assert length(k) == n "Wavenumber vector length mismatch: got $(length(k)), expected $n"
    
    # Compute differentiation via spectral method
    δ = vcat(1, zeros(n - 1))
    fft_result = real(ifft(k.^m .* fft(δ)))
    
    if iseven(m)
        # Even derivatives: symmetric
        col = fft_result
        row = col
    else
        # Odd derivatives: anti-symmetric
        col = vcat(0, fft_result[2:end])
        row = -col
    end
    
    return col, row
end

# ═══════════════════════════════════════════════════════════════════════════════════
# High-level differentiation object with domain scaling and caching
# ═══════════════════════════════════════════════════════════════════════════════════

"""
    FourierDiffn{T}

An elegant container for Fourier differentiation operators on arbitrary domains.

This structure provides:
- Automatic domain scaling from [0, 2π) to [0, L)
- Intelligent caching of derivative operators
- Clean indexing syntax: `𝒟[m]` returns the m-th derivative operator
- Memory-efficient storage using Toeplitz matrices

# Fields
- `n::Integer`: Number of grid points
- `L::T`: Domain length
- `x::Vector{T}`: Physical grid points in [0, L)
- `cache::Dict{Int,Toeplitz{T}}`: Cached derivative operators

# Mathematical Details
The scaling transformation from [0, 2π) to [0, L) requires multiplication
by the factor (2π/L)^m for the m-th derivative operator.
"""
struct FourierDiffn{T <: Real}
    n     :: Int
    L     :: T
    x     :: Vector{T}
    cache :: Dict{Int, Toeplitz{T}}
    
    function FourierDiffn{T}(n::Integer, L::T) where {T <: Real}
        @assert n > 0 "Number of points must be positive"
        @assert L > 0 "Domain length must be positive"
        
        # Create base grid and identity operator
        x₀, D₀ = FourierDiff(n, 0)
        
        # Scale grid to physical domain [0, L)
        x = (L / 2π) * x₀
        
        # Initialize cache with identity operator (ensure it's Toeplitz)
        cache = Dict{Int, Toeplitz{T}}(0 => Toeplitz(T.(D₀)))
        
        new{T}(n, L, x, cache)
    end
end

"""
    FourierDiffn(n::Integer; L::Real = 2π)

Construct a Fourier differentiation object on n points over [0, L).

# Arguments
- `n::Integer`: Number of grid points
- `L::Real = 2π`: Domain length

# Example
```julia
𝒟 = FourierDiffn(64; L = 4π)
println("Grid: ", 𝒟.x[1:5])  # First 5 points
```
"""
function FourierDiffn(n::Integer; L::Real = 2π)
    return FourierDiffn{typeof(float(L))}(n, L)
end

# ───────────────────────────────────────────────────────────────────────────────────
# Derivative operator computation and caching
# ───────────────────────────────────────────────────────────────────────────────────

"""
    derivative!(𝒟::FourierDiffn, m::Integer) -> Toeplitz

Compute and cache the m-th derivative operator, automatically scaled for domain [0, L).

The scaling factor (2π/L)^m ensures correct differentiation on the physical domain.
"""
function derivative!(𝒟::FourierDiffn{T}, m::Integer) where {T}
    @assert m ≥ 0 "Derivative order must be non-negative"
    
    # Return cached operator if available
    haskey(𝒟.cache, m) && return 𝒟.cache[m]
    
    # Compute new operator on canonical domain [0, 2π)
    _, Dₘ = FourierDiff(𝒟.n, m)

    # Apply domain scaling: (2π/L)^m
    # Scale the Toeplitz column/row vectors directly to preserve O(n) storage
    s = T((2π / 𝒟.L)^m)
    𝒟.cache[m] = Toeplitz(s .* T.(Dₘ.vc), s .* T.(Dₘ.vr))

    return 𝒟.cache[m]
end

# ───────────────────────────────────────────────────────────────────────────────────
# Elegant indexing and property access interface
# ───────────────────────────────────────────────────────────────────────────────────

"""
    𝒟[m] -> Toeplitz

Beautiful syntax for accessing the m-th derivative operator.

# Example
```julia
𝒟 = FourierDiffn(64)
D₁ = 𝒟[1]  # First derivative
D₂ = 𝒟[2]  # Second derivative
```
"""
Base.getindex(𝒟::FourierDiffn, m::Integer) = derivative!(𝒟, m)

"""
    𝒟.D₀, 𝒟.D₁, 𝒟.D₂, ... -> Toeplitz

Elegant property access for derivative operators using beautiful mathematical notation.

# Example
```julia
𝒟 = FourierDiffn(64)
identity = 𝒟.D₀    # Identity operator
first_deriv = 𝒟.D₁  # First derivative  
second_deriv = 𝒟.D₂ # Second derivative
third_deriv = 𝒟.D₃  # Third derivative
```
"""
function Base.getproperty(𝒟::FourierDiffn, sym::Symbol)
    # Handle standard fields
    if sym ∈ (:n, :L, :x, :cache)
        return getfield(𝒟, sym)
    end
    
    # Handle derivative operators: D₀, D₁, D₂, D₃, ...
    str_sym = string(sym)
    if startswith(str_sym, "D") && length(str_sym) > 1
        # Extract subscript digits
        subscript_str = str_sym[2:end]
        
        # Handle both regular digits and Unicode subscripts
        order = _parse_subscript(subscript_str)
        
        if order !== nothing
            return derivative!(𝒟, order)
        end
    end
    
    # Fallback to default behavior
    throw(ArgumentError("FourierDiffn has no property $sym"))
end

"""
    _parse_subscript(s::String) -> Union{Int, Nothing}

Parse subscript notation (both Unicode and regular digits) to extract derivative order.
"""
function _parse_subscript(s::String)
    # Unicode subscript mapping
    subscript_map = Dict(
        '₀' => '0', '₁' => '1', '₂' => '2', '₃' => '3', '₄' => '4',
        '₅' => '5', '₆' => '6', '₇' => '7', '₈' => '8', '₉' => '9'
    )
    
    # Convert Unicode subscripts to regular digits
    regular_digits = map(c -> get(subscript_map, c, c), s)
    
    # Parse as integer
    try
        return parse(Int, String(regular_digits))
    catch
        return nothing
    end
end

# Make property names available for tab completion
function Base.propertynames(𝒟::FourierDiffn)
    base_props = (:n, :L, :x, :cache)
    derivative_props = [Symbol("D$i") for i in 0:9]  # D₀ through D₉
    subscript_props = [Symbol("D$(i)") for i in '₀':'₉']  # D₀ through D₉
    return (base_props..., derivative_props..., subscript_props...)
end

# ───────────────────────────────────────────────────────────────────────────────────
# Beautiful display methods
# ───────────────────────────────────────────────────────────────────────────────────

function Base.show(io::IO, 𝒟::FourierDiffn{T}) where {T}
    print(io, "FourierDiffn{$T}")
end

function Base.show(io::IO, ::MIME"text/plain", 𝒟::FourierDiffn{T}) where {T}
    println(io, "╭───────────────────────────────────────────────────────────────────────    ")
    println(io, "│                   FourierDiffn{$T}                                        ")
    println(io, "├───────────────────────────────────────────────────────────────────────    ")
    println(io, "├──── Domain: [0, $(𝒟.L))                                                   ")
    println(io, "├──── Points: $(𝒟.n)                                                        ")
    println(io, "├──── Grid spacing: $(round(𝒟.L / 𝒟.n, sigdigits=4))                        ")
    println(io, "├──── Range: $(round(𝒟.x[1], sigdigits=4)) → $(round(𝒟.x[end], sigdigits=4))")
    println(io, "├──── Cached derivatives: $(sort(collect(keys(𝒟.cache))))                   ")
    println(io, "╰───────────────────────────────────────────────────────────────────────    ")
    println(io, "")
    println(io, "Usage: 𝒟[m] or 𝒟.Dₘ returns the m-th derivative operator")
end

# ───────────────────────────────────────────────────────────────────────────────────
# Convenience methods for common operations
# ───────────────────────────────────────────────────────────────────────────────────

"""
    derivative_orders(𝒟::FourierDiffn) -> Vector{Int}

Return the currently cached derivative orders.
"""
derivative_orders(𝒟::FourierDiffn) = sort(collect(keys(𝒟.cache)))

"""
    clear_cache!(𝒟::FourierDiffn)

Clear all cached derivative operators except the identity.
"""
function clear_cache!(𝒟::FourierDiffn)
    identity_op = 𝒟.cache[0]
    empty!(𝒟.cache)
    𝒟.cache[0] = identity_op
    return 𝒟
end

"""
    grid_spacing(𝒟::FourierDiffn) -> Real

Return the grid spacing in the physical domain.
"""
grid_spacing(𝒟::FourierDiffn) = 𝒟.L / 𝒟.n

#end # module FourierDifferentiation