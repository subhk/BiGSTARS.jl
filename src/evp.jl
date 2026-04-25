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

"""
Stored boundary condition.

Two kinds:
- **Algebraic** (`is_dynamic == false`): `left(psi) == 0`. Only modifies A; B row is zeroed.
- **Dynamic** (`is_dynamic == true`): `left(sigma*dz(psi) + U*dx(dz(psi))) == 0`.
  Contains the eigenvalue symbol. The eigenvalue side goes to B, the rest to A —
  exactly like an equation but evaluated at the boundary.
"""
struct BoundaryCondition
    side::Symbol          # :left or :right
    coord::Symbol         # which Chebyshev coordinate
    expr::ExprNode        # full expression (may contain EigenvalueNode)
    rhs::Number           # inhomogeneous value (0 for homogeneous)
    is_dynamic::Bool      # true if expr contains the eigenvalue symbol
end

"""
Derived (auxiliary) variable: defined implicitly by `Op(v) = rhs_expr(w, zeta, ...)`.
The code computes `v = Op^{-1} * rhs_expr` during discretization and substitutes
it wherever `v` appears in equations.

BCs for the inversion can be specified to ensure uniqueness when the operator
has a null space (e.g., `dz(dz(v)) = rhs` needs 2 BCs to pin the solution).
"""
struct DerivedVariable
    name::Symbol            # :v
    operator_name::Symbol   # :Dh2  (the operator applied to v on the LHS)
    rhs::ExprNode           # dy(dz(w)) - dx(zeta)  (the RHS expression)
    bcs::Vector{BoundaryCondition}  # BCs for the inversion (can be empty)
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
    derived_vars::Dict{Symbol, DerivedVariable}

    function EVP(domain::Domain; variables::Vector{Symbol}, eigenvalue::Symbol)
        prob = new(domain, variables, eigenvalue,
            Dict{Symbol, Any}(),
            Equation[],
            BoundaryCondition[],
            Dict{Symbol, Substitution}(),
            Dict{Symbol, DerivedVariable}())
        _set_active_prob!(prob)
        return prob
    end
end

# ──────────────────────────────────────────────────────────────────────────────
#  Active problem registry
# ──────────────────────────────────────────────────────────────────────────────

const _active_prob = Ref{Union{EVP, Nothing}}(nothing)

_set_active_prob!(prob::EVP) = (_active_prob[] = prob)

"""Return the active problem, or error with a clear message."""
function _get_active_prob()
    isnothing(_active_prob[]) &&
        error("No active EVP problem. Create one with EVP(...) or pass `prob` explicitly.")
    return _active_prob[]
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
    dynamic = _contains_eigenvalue(expr, prob.eigenvalue)
    push!(prob.bcs, BoundaryCondition(side, coord, expr, rhs, dynamic))
end

"""Register a derived (auxiliary) variable: Op(v) = rhs_expr."""
function add_derived!(prob::EVP, name::Symbol, operator_name::Symbol, rhs::ExprNode)
    prob.derived_vars[name] = DerivedVariable(name, operator_name, rhs, BoundaryCondition[])
end

"""Add a boundary condition for a derived variable's inversion."""
function add_derived_bc!(prob::EVP, var_name::Symbol, side::Symbol, coord::Symbol,
                         deriv_order::Int, rhs::Float64=0.0)
    haskey(prob.derived_vars, var_name) ||
        error("No derived variable :$var_name. Call @derive before @derive_bc.")
    deriv_order >= 0 || throw(ArgumentError("Derivative order must be non-negative"))
    expr::ExprNode = VarNode(var_name)
    for _ in 1:deriv_order
        expr = DerivNode(expr, coord)
    end
    bc = BoundaryCondition(side, coord, expr, rhs, false)
    push!(prob.derived_vars[var_name].bcs, bc)
end

"""Check if an expression tree contains the eigenvalue symbol."""
function _contains_eigenvalue(expr::ExprNode, eig::Symbol)
    expr isa EigenvalueNode && expr.name == eig && return true
    expr isa BinaryOpNode && return _contains_eigenvalue(expr.left, eig) || _contains_eigenvalue(expr.right, eig)
    expr isa UnaryOpNode && return _contains_eigenvalue(expr.expr, eig)
    expr isa DerivNode && return _contains_eigenvalue(expr.expr, eig)
    expr isa SubstitutionNode && return any(a -> _contains_eigenvalue(a, eig), expr.args)
    return false
end

"""Add a substitution template to the problem."""
function add_substitution!(prob::EVP, name::Symbol, arg_names::Vector{Symbol}, body::ExprNode)
    prob.substitutions[name] = Substitution(name, arg_names, body)
end

function Base.show(io::IO, prob::EVP)
    println(io, "EVP Problem")
    println(io, "  Domain: ", prob.domain)
    println(io, "  Variables: ", prob.variables)
    println(io, "  Eigenvalue: ", prob.eigenvalue)
    println(io, "  Parameters: ", join(keys(prob.parameters), ", "))
    println(io, "  Equations: ", length(prob.equations))
    println(io, "  BCs: ", length(prob.bcs),
            " (", count(bc -> bc.is_dynamic, prob.bcs), " dynamic)")
    println(io, "  Substitutions: ", join(keys(prob.substitutions), ", "))
    isempty(prob.derived_vars) || println(io, "  Derived: ", join(keys(prob.derived_vars), ", "))
end

"""Find the first Chebyshev coordinate in the domain, or nothing if none exists."""
function first_chebyshev_coord(prob::EVP)
    for dim in prob.domain.coord_order
        if prob.domain.coords[dim] isa ChebyshevBasisSpec
            return dim
        end
    end
    return nothing
end
