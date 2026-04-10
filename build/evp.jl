#══════════════════════════════════════════════════════════════════════════════#
#  EVP: Eigenvalue Problem container                                          #
#══════════════════════════════════════════════════════════════════════════════#

"""Stored substitution: a template expression tree with named arguments."""
struct Substitution
    name::Symbol
    arg_names::Vector{Symbol}
    body::ExprNode
end

"""Stored equation: LHS == RHS expression trees."""
struct Equation
    lhs::ExprNode
    rhs::ExprNode
end

"""Stored boundary condition."""
struct BoundaryCondition
    side::Symbol          # :left or :right
    coord::Symbol         # which Chebyshev coordinate
    expr::ExprNode        # expression evaluated at boundary
    rhs::Number           # inhomogeneous value (0 for homogeneous)
end

"""
    EVP(domain; variables, eigenvalue)

Eigenvalue problem container. Accumulates equations, BCs, parameters,
and substitutions before discretization.

Usage:
```julia
prob = EVP(domain, variables=[:psi, :b], eigenvalue=:sigma)
prob[:U] = U_field
prob[:E] = 1e-12
```
"""
mutable struct EVP
    domain::Domain
    variables::Vector{Symbol}
    eigenvalue::Symbol
    parameters::Dict{Symbol, Any}
    equations::Vector{Equation}
    bcs::Vector{BoundaryCondition}
    substitutions::Dict{Symbol, Substitution}

    function EVP(domain::Domain; variables::Vector{Symbol}, eigenvalue::Symbol)
        new(domain, variables, eigenvalue,
            Dict{Symbol, Any}(),
            Equation[],
            BoundaryCondition[],
            Dict{Symbol, Substitution}())
    end
end

function Base.setindex!(prob::EVP, value, name::Symbol)
    if name in prob.variables
        throw(ArgumentError("Cannot set parameter '$name': it is a declared variable"))
    end
    if name == prob.eigenvalue
        throw(ArgumentError("Cannot set parameter '$name': it is the eigenvalue symbol"))
    end
    prob.parameters[name] = value
end

Base.getindex(prob::EVP, name::Symbol) = prob.parameters[name]

"""Add an equation LHS == RHS to the problem."""
function add_equation!(prob::EVP, lhs::ExprNode, rhs::ExprNode)
    push!(prob.equations, Equation(lhs, rhs))
end

"""Add a boundary condition to the problem."""
function add_bc!(prob::EVP, side::Symbol, coord::Symbol, expr::ExprNode, rhs::Number=0.0)
    push!(prob.bcs, BoundaryCondition(side, coord, expr, rhs))
end

"""Add a substitution template to the problem."""
function add_substitution!(prob::EVP, name::Symbol, arg_names::Vector{Symbol}, body::ExprNode)
    prob.substitutions[name] = Substitution(name, arg_names, body)
end

"""Find the first Chebyshev coordinate in the domain."""
function first_chebyshev_coord(prob::EVP)
    for dim in prob.domain.coord_order
        if prob.domain.coords[dim] isa ChebyshevBasisSpec
            return dim
        end
    end
    error("No Chebyshev coordinate found in domain")
end
