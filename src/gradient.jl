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
    
    # Input validation
    @assert 1 ≤ dims ≤ ndims(f) "dims must be between 1 and $(ndims(f))"
    @assert length(x) == size(f, dims) "Length of x ($(length(x))) must match size of f along dimension $dims ($(size(f, dims)))"
    @assert ndims(f) ≤ 3 "GradientCalculator currently only supports 1D, 2D, or 3D arrays"
    
    # Initialize result array
    n = size(f)
    gradient_array = similar(f)
    
    # Compute gradient based on array dimensionality
    if ndims(f) == 1
        _compute_gradient_1d!(calc, gradient_array, f, x)
    elseif ndims(f) == 2
        _compute_gradient_2d!(calc, gradient_array, f, x, dims, n)
    else # ndims(f) == 3
        _compute_gradient_3d!(calc, gradient_array, f, x, dims, n)
    end
    
    return gradient_array
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
    derivative(f, x, order; dims=1, bc="nearest")

Compute nth-order derivative using specified boundary conditions.
"""
derivative(f, x, order; dims=1, bc="nearest") = GradientCalculator(bc, max(3, order), order)(f, x; dims=dims)

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

# Helper function for 1D arrays
function _compute_gradient_1d!(calc::GradientCalculator{S}, 
                              grad::AbstractVector{T}, 
                              f::AbstractVector{T}, 
                              x::AbstractVector{T}) where {S<:AbstractFloat, T<:AbstractFloat}
    
    itp = Spline1D(x, f, bc=calc.boundary_condition)
    
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
                              n::Tuple{Int,Int}) where {S<:AbstractFloat, T<:AbstractFloat}
    
    if dims == 1
        # Gradient along rows (columns are independent)
        @inbounds for j in 1:n[2]
            column_slice = view(f, :, j)
            itp = Spline1D(x, column_slice, bc=calc.boundary_condition)
            
            for i in 1:n[1]
                grad[i, j] = Dierckx.derivative(itp, x[i]; nu=calc.derivative_order)
            end
        end
    else # dims == 2
        # Gradient along columns (rows are independent)
        @inbounds for i in 1:n[1]
            row_slice = view(f, i, :)
            itp = Spline1D(x, row_slice, bc=calc.boundary_condition)
            
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
                              n::Tuple{Int,Int,Int}) where {S<:AbstractFloat, T<:AbstractFloat}
    
    if dims == 1
        # Gradient along first dimension
        @inbounds for j in 1:n[2], k in 1:n[3]
            slice = view(f, :, j, k)
            itp = Spline1D(x, slice, bc=calc.boundary_condition)
            
            for i in 1:n[1]
                grad[i, j, k] = Dierckx.derivative(itp, x[i]; nu=calc.derivative_order)
            end
        end
        
    elseif dims == 2
        # Gradient along second dimension
        @inbounds for i in 1:n[1], k in 1:n[3]
            slice = view(f, i, :, k)
            itp = Spline1D(x, slice, bc=calc.boundary_condition)
            
            for j in 1:n[2]
                grad[i, j, k] = Dierckx.derivative(itp, x[j]; nu=calc.derivative_order)
            end
        end
        
    else # dims == 3
        # Gradient along third dimension
        @inbounds for i in 1:n[1], j in 1:n[2]
            slice = view(f, i, j, :)
            itp = Spline1D(x, slice, bc=calc.boundary_condition)
            
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