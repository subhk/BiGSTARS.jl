import Dierckx

"""
    GradientCalculator{T<:AbstractFloat}

A configurable derivative calculator that computes numerical derivatives of multi-dimensional 
arrays using cubic spline interpolation. Supports 1st, 2nd, 3rd, and higher-order derivatives.

# Fields
- `boundary_condition::String`: Boundary condition for spline interpolation (default: "nearest")
- `spline_degree::Int`: Degree of spline interpolation (default: 3)
- `derivative_order::Int`: Order of derivative to compute (default: 1)

# Examples
```julia
# Create calculator with default settings (1st derivative)
calc = GradientCalculator()

# Create calculator for 2nd derivative
calc = GradientCalculator(derivative_order=2)

# Create calculator for 3rd derivative with custom boundary conditions
calc = GradientCalculator("zero", 3, 3)

# Compute derivatives
x = 0:0.1:2π
f = sin.(x)
df_dx = calc(f, x)                                    # 1st derivative: cos(x)
d2f_dx2 = gradient(f, x, order=2)                     # 2nd derivative: -sin(x)
d3f_dx3 = derivative(f, x, 3)                         # 3rd derivative: -cos(x)
```
"""
struct GradientCalculator{T<:AbstractFloat}
    boundary_condition::String
    spline_degree::Int
    derivative_order::Int
    
    function GradientCalculator{T}(boundary_condition::String="nearest", 
                                  spline_degree::Int=3, 
                                  derivative_order::Int=1) where {T<:AbstractFloat}
        @assert spline_degree > 0 "Spline degree must be positive"
        @assert derivative_order > 0 "Derivative order must be positive"
        @assert derivative_order ≤ spline_degree "Derivative order ($derivative_order) cannot exceed spline degree ($spline_degree)"
        @assert boundary_condition in ["nearest", "zero", "extrapolate", "error"] "Invalid boundary condition"
        
        new{T}(boundary_condition, spline_degree, derivative_order)
    end
end

# Convenience constructor
GradientCalculator(args...) = GradientCalculator{Float64}(args...)

"""
    (calc::GradientCalculator)(f::AbstractArray{T}, x::AbstractVector{T}; dims::Int=1) where {T<:AbstractFloat}

Compute the nth-order derivative of multi-dimensional array `f` sampled at points `x` along dimension `dims`.

# Arguments
- `f`: Input array (1D, 2D, or 3D)
- `x`: Sample points corresponding to the specified dimension
- `dims`: Dimension along which to compute the derivative (default: 1)

# Returns
- Array of the same size as `f` containing the nth-order derivative values

# Note
The derivative order is determined by the `derivative_order` field of the calculator.
"""
function (calc::GradientCalculator{S})(f::AbstractArray{T}, x::AbstractVector{T}; 
                                      dims::Int=1) where {S<:AbstractFloat, T<:AbstractFloat}
    
    grad = similar(f)
    return calc(grad, f, x; dims=dims)
end

"""
    (calc::GradientCalculator)(grad, f, x; dims=1)

In-place derivative computation. Writes into the preallocated `grad` array
instead of allocating a new output.
"""
function (calc::GradientCalculator{S})(grad::AbstractArray{T},
                                       f::AbstractArray{T},
                                       x::AbstractVector{T};
                                       dims::Int=1) where {S<:AbstractFloat, T<:AbstractFloat}
    # Input validation
    nd = ndims(f)
    @assert nd == ndims(grad) "grad and f must have the same dimensionality"
    @assert size(grad) == size(f) "grad size must match f"
    @assert 1 ≤ dims ≤ nd "dims must be between 1 and $nd"
    @assert length(x) == size(f, dims) "Length of x ($(length(x))) must match size of f along dimension $dims ($(size(f, dims)))"
    @assert nd ≤ 3 "GradientCalculator currently only supports 1D, 2D, or 3D arrays"

    n = size(f)
    ws = _spline_workspace(length(x), calc.spline_degree)

    if nd == 1
        _compute_gradient_1d!(calc, grad, f, x, ws)
    elseif nd == 2
        _compute_gradient_2d!(calc, grad, f, x, dims, n, ws)
    else # nd == 3
        _compute_gradient_3d!(calc, grad, f, x, dims, n, ws)
    end

    return grad
end

# Convenience methods for different derivative orders and boundary conditions
"""
    gradient(f, x; dims=1, order=1, bc="nearest")

Compute nth-order derivative (gradient) using specified boundary conditions.

# Arguments
- `f`: Input array (1D, 2D, or 3D)
- `x`: Sample points corresponding to the specified dimension
- `dims`: Dimension along which to compute the derivative (default: 1)
- `order`: Order of derivative to compute (default: 1)
- `bc`: Boundary condition for spline interpolation (default: "nearest")
"""
gradient(f, x; dims=1, order=1, bc="nearest") = GradientCalculator(bc, max(3, order), order)(f, x; dims=dims)

"""
    gradient!(grad, f, x; dims=1, order=1, bc="nearest")

In-place version of `gradient` that writes into `grad` to avoid allocating
the output array.
"""
gradient!(grad, f, x; dims=1, order=1, bc="nearest") = GradientCalculator(bc, max(3, order), order)(grad, f, x; dims=dims)

"""
    derivative(f, x, order; dims=1, bc="nearest")

Compute nth-order derivative using specified boundary conditions.
"""
derivative(f, x, order; dims=1, bc="nearest") = GradientCalculator(bc, max(3, order), order)(f, x; dims=dims)

"""
    derivative!(grad, f, x, order; dims=1, bc="nearest")

In-place nth-order derivative; writes into `grad` to avoid output allocation.
"""
derivative!(grad, f, x, order; dims=1, bc="nearest") = GradientCalculator(bc, max(3, order), order)(grad, f, x; dims=dims)

# Convenience methods for different boundary conditions (1st derivative)
"""
    nearest_gradient(f, x; dims=1)

Compute 1st derivative using nearest boundary conditions.
"""
nearest_gradient(f, x; dims=1) = GradientCalculator("nearest", 3, 1)(f, x; dims=dims)

"""
    zero_gradient(f, x; dims=1)

Compute 1st derivative using zero boundary conditions.
"""
zero_gradient(f, x; dims=1) = GradientCalculator("zero", 3, 1)(f, x; dims=dims)

"""
    extrapolate_gradient(f, x; dims=1)

Compute 1st derivative using extrapolation boundary conditions.
"""
extrapolate_gradient(f, x; dims=1) = GradientCalculator("extrapolate", 3, 1)(f, x; dims=dims)

# ────────────────────────────────────────────────────────────────────────────────
# Lightweight spline workspace to reuse allocations across slices
# ────────────────────────────────────────────────────────────────────────────────
mutable struct Spline1DWorkspace
    x::Vector{Float64}
    y::Vector{Float64}
    w::Vector{Float64}
    t::Vector{Float64}
    c::Vector{Float64}
    wrk::Vector{Float64}
    iwrk::Vector{Int32}
    k::Int
end

function _spline_workspace(m::Int, k::Int)
    nest = max(m + k + 1, 2k + 3)
    lwrk = m * (k + 1) + nest * (7 + 3k)
    return Spline1DWorkspace(
        Vector{Float64}(undef, m),          # x buffer
        Vector{Float64}(undef, m),          # y buffer
        ones(Float64, m),                   # weights (all ones)
        Vector{Float64}(undef, nest),       # knots
        Vector{Float64}(undef, nest),       # coeffs (upper bound)
        Vector{Float64}(undef, lwrk),       # work array
        Vector{Int32}(undef, nest),         # integer workspace
        k
    )
end

@inline function _ensure_workspace!(ws::Spline1DWorkspace, m::Int, k::Int)
    ws.k = k
    if length(ws.x) != m
        resize!(ws.x, m)
        resize!(ws.y, m)
        resize!(ws.w, m)
        fill!(ws.w, 1.0)
    end

    nest = max(m + k + 1, 2k + 3)
    lwrk = m * (k + 1) + nest * (7 + 3k)

    length(ws.t) < nest && resize!(ws.t, nest)
    length(ws.c) < nest && resize!(ws.c, nest)
    length(ws.wrk) < lwrk && resize!(ws.wrk, lwrk)
    length(ws.iwrk) < nest && resize!(ws.iwrk, nest)

    return nest, lwrk
end

function _fit_spline!(ws::Spline1DWorkspace,
                      x::AbstractVector,
                      y::AbstractVector,
                      k::Int,
                      bc::AbstractString)
    m = length(x)
    nest, lwrk = _ensure_workspace!(ws, m, k)

    copyto!(ws.x, x)
    copyto!(ws.y, y)
    fill!(ws.w, 1.0)

    n  = Ref{Int32}(0)
    fp = Ref{Float64}(0.0)
    ier = Ref{Int32}(0)

    ccall((:curfit_, Dierckx.libddierckx), Nothing,
          (Ref{Int32}, Ref{Int32}, Ref{Float64}, Ref{Float64}, Ref{Float64},
           Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Float64},
           Ref{Int32}, Ref{Int32}, Ref{Float64}, Ref{Float64}, Ref{Float64},
           Ref{Float64}, Ref{Int32}, Ref{Int32}, Ref{Int32}),
          0, m, ws.x, ws.y, ws.w, ws.x[1], ws.x[end], k, 0.0,
          nest, n, ws.t, ws.c, fp, ws.wrk, lwrk, ws.iwrk, ier)

    if ier[] > 0
        error(Dierckx._fit1d_messages[ier[]])
    end

    resize!(ws.t, n[])
    resize!(ws.c, n[] - k - 1)

    bc_int = Dierckx._translate_bc(bc)
    return Dierckx.Spline1D(ws.t, ws.c, k, bc_int, fp[], ws.wrk)
end

# Helper function for 1D arrays
function _compute_gradient_1d!(calc::GradientCalculator{S}, 
                              grad::AbstractVector{T}, 
                              f::AbstractVector{T}, 
                              x::AbstractVector{T},
                              ws::Spline1DWorkspace) where {S<:AbstractFloat, T<:AbstractFloat}
    
    itp = _fit_spline!(ws, x, f, calc.spline_degree, calc.boundary_condition)
    
    @inbounds for i in eachindex(x)
        grad[i] = Dierckx.derivative(itp, x[i]; nu=calc.derivative_order)
    end
end

# Helper function for 2D arrays
function _compute_gradient_2d!(calc::GradientCalculator{S}, 
                              grad::AbstractMatrix{T}, 
                              f::AbstractMatrix{T}, 
                              x::AbstractVector{T}, 
                              dims::Int, 
                              n::Tuple{Int,Int},
                              ws::Spline1DWorkspace) where {S<:AbstractFloat, T<:AbstractFloat}
    
    if dims == 1
        # Gradient along rows (columns are independent)
        @inbounds for j in 1:n[2]
            column_slice = view(f, :, j)
            itp = _fit_spline!(ws, x, column_slice, calc.spline_degree, calc.boundary_condition)
            
            for i in 1:n[1]
                grad[i, j] = Dierckx.derivative(itp, x[i]; nu=calc.derivative_order)
            end
        end
    else # dims == 2
        # Gradient along columns (rows are independent)
        @inbounds for i in 1:n[1]
            row_slice = view(f, i, :)
            itp = _fit_spline!(ws, x, row_slice, calc.spline_degree, calc.boundary_condition)
            
            for j in 1:n[2]
                grad[i, j] = Dierckx.derivative(itp, x[j]; nu=calc.derivative_order)
            end
        end
    end
end

# Helper function for 3D arrays
function _compute_gradient_3d!(calc::GradientCalculator{S}, 
                              grad::AbstractArray{T,3}, 
                              f::AbstractArray{T,3}, 
                              x::AbstractVector{T}, 
                              dims::Int, 
                              n::Tuple{Int,Int,Int},
                              ws::Spline1DWorkspace) where {S<:AbstractFloat, T<:AbstractFloat}
    
    if dims == 1
        # Gradient along first dimension
        @inbounds for j in 1:n[2], k in 1:n[3]
            slice = view(f, :, j, k)
            itp = _fit_spline!(ws, x, slice, calc.spline_degree, calc.boundary_condition)
            
            for i in 1:n[1]
                grad[i, j, k] = Dierckx.derivative(itp, x[i]; nu=calc.derivative_order)
            end
        end
        
    elseif dims == 2
        # Gradient along second dimension
        @inbounds for i in 1:n[1], k in 1:n[3]
            slice = view(f, i, :, k)
            itp = _fit_spline!(ws, x, slice, calc.spline_degree, calc.boundary_condition)
            
            for j in 1:n[2]
                grad[i, j, k] = Dierckx.derivative(itp, x[j]; nu=calc.derivative_order)
            end
        end
        
    else # dims == 3
        # Gradient along third dimension
        @inbounds for i in 1:n[1], j in 1:n[2]
            slice = view(f, i, j, :)
            itp = _fit_spline!(ws, x, slice, calc.spline_degree, calc.boundary_condition)
            
            for k in 1:n[3]
                grad[i, j, k] = Dierckx.derivative(itp, x[k]; nu=calc.derivative_order)
            end
        end
    end
end

# Pretty printing
function Base.show(io::IO, calc::GradientCalculator{T}) where {T}
    print(io, "GradientCalculator{$T}(")
    print(io, "boundary_condition=\"$(calc.boundary_condition)\", ")
    print(io, "spline_degree=$(calc.spline_degree), ")
    print(io, "derivative_order=$(calc.derivative_order)")
    print(io, ")")
end
