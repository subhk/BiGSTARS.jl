# Boundary Conditions for Spectral Methods
# =====================================

"""
Physical boundary condition mappings for oceanographic/atmospheric models.

Variables:
- w: vertical velocity
- ζ: vorticity  
- b: buoyancy
"""
const BOUNDARY_CONDITIONS = Dict(
    (:w, "rigid_lid")  => :dirichlet,
    (:w, "free")       => :neumann,
    (:w, "open")       => :neumann,
    (:ζ, "free_slip")  => :neumann,
    (:ζ, "no_slip")    => :dirichlet,
    (:b, "zero_flux")  => :neumann,
    (:b, "fixed")      => :dirichlet,
)

"""
    BoundaryConditionHandler

A handler for applying boundary conditions to spectral derivative operators.

# Fields
- `grid_size::Int`: Number of grid points
- `tolerance::Float64`: Tolerance for zero pivot detection

# Example
```julia
bc_handler = BoundaryConditionHandler(64, 1e-14)
bc_handler(grid, :dirichlet)
```
"""
struct BoundaryConditionHandler
    grid_size::Int
    tolerance::Float64
    
    function BoundaryConditionHandler(grid_size::Int, tolerance::Float64 = 1e-14)
        grid_size > 0 || throw(ArgumentError("Grid size must be positive"))
        tolerance > 0 || throw(ArgumentError("Tolerance must be positive"))
        new(grid_size, tolerance)
    end
end

"""
    bc_type(variable::Symbol, condition::String) -> Symbol

Map physical boundary condition to mathematical type.

# Examples
```julia
bc_type(:w, "rigid_lid")  # :dirichlet
bc_type(:ζ, "free_slip")  # :neumann
```
"""
function bc_type(variable::Symbol, condition::String)
    return get(BOUNDARY_CONDITIONS, (variable, condition)) do
        error("Unknown boundary condition: $variable => \"$condition\"")
    end
end

# Dirichlet Boundary Conditions (Fixed Values)
# ============================================

"""Apply Dirichlet BC to first derivative operator."""
function apply_dirichlet!(bc::BoundaryConditionHandler, matrix::AbstractMatrix, ::Val{:D1})
    n = bc.grid_size
    matrix[1, 1] = 0.0
    matrix[n, n] = 0.0
    return nothing
end

"""Apply Dirichlet BC to second derivative operator."""
function apply_dirichlet!(bc::BoundaryConditionHandler, matrix::AbstractMatrix, ::Val{:D2})
    n = bc.grid_size
    matrix[1, 1] = 0.0
    matrix[n, n] = 0.0
    return nothing
end

"""Apply Dirichlet BC to third derivative operator."""
function apply_dirichlet!(bc::BoundaryConditionHandler, matrix::AbstractMatrix, ::Val{:D3})
    n = bc.grid_size
    matrix[1, 1] = 0.0
    matrix[n, n] = 0.0
    return nothing
end

"""Apply Dirichlet BC to fourth derivative operator with elimination."""
function apply_dirichlet!(bc::BoundaryConditionHandler, D₄::AbstractMatrix, D₂::AbstractMatrix, ::Val{:D4})
    n = bc.grid_size
    
    # Eliminate boundary dependencies
    for j in 2:n
        D₄[1, j] -= D₄[1, 1] * D₂[1, j]
    end
    
    for j in 1:(n-1)
        D₄[n, j] -= D₄[n, n] * D₂[n, j]
    end
    
    # Set boundary conditions
    D₄[1, 1] = 0.0
    D₄[n, n] = 0.0
    
    return nothing
end

# Neumann Boundary Conditions (Fixed Derivatives)
# ===============================================

"""Apply Neumann BC to first derivative operator."""
function apply_neumann!(bc::BoundaryConditionHandler, matrix::AbstractMatrix, ::Val{:D1})
    n = bc.grid_size
    matrix[1, :] .= 0.0
    matrix[n, :] .= 0.0
    return nothing
end

"""Apply Neumann BC to second derivative operator with elimination."""
function apply_neumann!(bc::BoundaryConditionHandler, D₂::AbstractMatrix, D₁::AbstractMatrix, ::Val{:D2})
    n = bc.grid_size
    
    # Check for zero pivots to avoid division by zero
    pivot_top = D₁[1, 1]
    pivot_bot = D₁[n, n]
    
    if abs(pivot_top) < bc.tolerance || abs(pivot_bot) < bc.tolerance
        error("Zero pivot encountered in Neumann BC elimination for D₂")
    end
    
    # Eliminate boundary dependencies
    for j in 2:n
        D₂[1, j] -= D₂[1, 1] * D₁[1, j] / pivot_top
    end
    
    for j in 1:(n-1)
        D₂[n, j] -= D₂[n, n] * D₁[n, j] / pivot_bot
    end
    
    # Set boundary conditions
    D₂[1, 1] = 0.0
    D₂[n, n] = 0.0
    
    return nothing
end

"""Apply Neumann BC to third derivative operator with elimination."""
function apply_neumann!(bc::BoundaryConditionHandler, D₃::AbstractMatrix, D₁::AbstractMatrix, ::Val{:D3})
    n = bc.grid_size
    
    # Check for zero pivots
    pivot_top = D₁[1, 1]
    pivot_bot = D₁[n, n]
    
    if abs(pivot_top) < bc.tolerance || abs(pivot_bot) < bc.tolerance
        error("Zero pivot encountered in Neumann BC elimination for D₃")
    end
    
    # Eliminate boundary dependencies
    for j in 2:n
        D₃[1, j] -= D₃[1, 1] * D₁[1, j] / pivot_top
    end
    
    for j in 1:(n-1)
        D₃[n, j] -= D₃[n, n] * D₁[n, j] / pivot_bot
    end
    
    # Set boundary conditions
    D₃[1, 1] = 0.0
    D₃[n, n] = 0.0
    
    return nothing
end

"""Apply Neumann BC to fourth derivative operator with elimination."""
function apply_neumann!(bc::BoundaryConditionHandler, D₄::AbstractMatrix, D₁::AbstractMatrix, ::Val{:D4})
    n = bc.grid_size
    
    # Check for zero pivots
    pivot_top = D₁[1, 2]
    pivot_bot = D₁[n, n-1]
    
    if abs(pivot_top) < bc.tolerance || abs(pivot_bot) < bc.tolerance
        error("Zero pivot encountered in Neumann BC elimination for D₄")
    end
    
    # Eliminate boundary dependencies
    for j in 3:n
        D₄[1, j] -= D₄[1, 2] * D₁[1, j] / pivot_top
    end
    
    for j in 1:(n-2)
        D₄[n, j] -= D₄[n, n-1] * D₁[n, j] / pivot_bot
    end
    
    # Set boundary conditions
    D₄[1, 2] = 0.0
    D₄[n, n-1] = 0.0
    
    return nothing
end

# Main Interface
# ==============

"""
    (bc::BoundaryConditionHandler)(grid, bc_type::Symbol)

Apply boundary conditions to all relevant derivative operators.

# Arguments
- `grid`: Grid structure containing derivative matrices
- `bc_type`: Either `:dirichlet` or `:neumann`

# Examples
```julia
bc_handler = BoundaryConditionHandler(64)
bc_handler(grid, :dirichlet)
bc_handler(grid, :neumann)
```
"""
function (bc::BoundaryConditionHandler)(grid, bc_type::Symbol)
    if bc_type == :dirichlet
        apply_dirichlet!(bc, grid.Dᶻᴰ,            Val(:D1))
        apply_dirichlet!(bc, grid.D²ᶻᴰ,           Val(:D2))
        apply_dirichlet!(bc, grid.D⁴ᶻᴰ, grid.D²ᶻ, Val(:D4))
        
    elseif bc_type == :neumann
        apply_neumann!(bc, grid.Dᶻᴺ, Val(:D1))
        apply_neumann!(bc, grid.D²ᶻᴺ, grid.Dᶻ, Val(:D2))
        apply_neumann!(bc, grid.D³ᶻᴺ, grid.Dᶻ, Val(:D3))
        apply_neumann!(bc, grid.D⁴ᶻᴺ, grid.Dᶻ, Val(:D4))
        
    else
        error("Unknown boundary condition type: $bc_type. Use :dirichlet or :neumann")
    end
    
    return nothing
end

# Convenience Methods with Symbol Dispatch
# ========================================

"""
    apply_dirichlet!(bc, matrix, derivative_order::Symbol)

Convenience method using symbol dispatch for derivative order.

# Examples
```julia
apply_dirichlet!(bc_handler, matrix, :D1)
apply_dirichlet!(bc_handler, D4_matrix, D2_matrix, :D4)
```
"""
apply_dirichlet!(bc::BoundaryConditionHandler, matrix::AbstractMatrix, order::Symbol) = 
    apply_dirichlet!(bc, matrix, Val(order))

apply_dirichlet!(bc::BoundaryConditionHandler, D₄::AbstractMatrix, D₂::AbstractMatrix, order::Symbol) = 
    apply_dirichlet!(bc, D₄, D₂, Val(order))

"""
    apply_neumann!(bc, matrix, derivative_order::Symbol)

Convenience method using symbol dispatch for derivative order.

# Examples
```julia
apply_neumann!(bc_handler, matrix, :D1)
apply_neumann!(bc_handler, D2_matrix, D1_matrix, :D2)
```
"""
apply_neumann!(bc::BoundaryConditionHandler, matrix::AbstractMatrix, order::Symbol) = 
    apply_neumann!(bc, matrix, Val(order))

apply_neumann!(bc::BoundaryConditionHandler, target::AbstractMatrix, reference::AbstractMatrix, order::Symbol) = 
    apply_neumann!(bc, target, reference, Val(order))

# Backward Compatibility
# ======================

"""
    apply_boundary_conditions!(grid, params, bc_type::Symbol)

Convenience method for backward compatibility.
"""
function apply_boundary_conditions!(grid, params, bc_type::Symbol)
    bc_handler = BoundaryConditionHandler(params.Nz)
    bc_handler(grid, bc_type)
    return nothing
end

"""
    setBCs!(grid, params, bc_type::Symbol)

Legacy alias for backward compatibility.
"""
const setBCs! = apply_boundary_conditions!

# Factory Functions
# ================

"""
    create_bc_handler(grid_size::Int; tolerance::Float64 = 1e-14) -> BoundaryConditionHandler

Create a boundary condition handler with specified parameters.

# Example
```julia
bc_handler = create_bc_handler(64, tolerance=1e-12)
```
"""
function create_bc_handler(grid_size::Int; tolerance::Float64 = 1e-14)
    return BoundaryConditionHandler(grid_size, tolerance)
end

"""
    create_bc_handler(params; tolerance::Float64 = 1e-14) -> BoundaryConditionHandler

Create a boundary condition handler from parameters object.

# Example
```julia
bc_handler = create_bc_handler(params)
```
"""
function create_bc_handler(params; tolerance::Float64 = 1e-14)
    return BoundaryConditionHandler(params.Nz, tolerance)
end

# Utilities
# =========

"""
    validate_derivative_order(order::Symbol)

Validate that the derivative order is supported.
"""
function validate_derivative_order(order::Symbol)
    valid_orders = (:D1, :D2, :D3, :D4)
    order ∈ valid_orders || throw(ArgumentError("Invalid derivative order: $order. Must be one of $valid_orders"))
    return true
end