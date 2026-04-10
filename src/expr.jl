#══════════════════════════════════════════════════════════════════════════════#
#  Expression tree node types for the symbolic equation DSL                   #
#══════════════════════════════════════════════════════════════════════════════#

"""Abstract type for all expression tree nodes."""
abstract type ExprNode end

"""A variable (unknown being solved for, e.g., ψ)."""
struct VarNode <: ExprNode
    name::Symbol
end

"""A parameter (scalar or field, set by the user, e.g., U, E)."""
struct ParamNode <: ExprNode
    name::Symbol
end

"""A numeric constant."""
struct ConstNode <: ExprNode
    value::Number
end

"""The eigenvalue symbol (e.g., σ in σ*ψ)."""
struct EigenvalueNode <: ExprNode
    name::Symbol
end

"""A wavenumber variable (generated from lowering dx → im*k)."""
struct WavenumberNode <: ExprNode
    name::Symbol
end

"""A derivative applied to an expression along a coordinate."""
struct DerivNode <: ExprNode
    expr::ExprNode
    coord::Symbol
end

"""A binary operation (+, -, *) on two expressions."""
struct BinaryOpNode <: ExprNode
    op::Symbol
    left::ExprNode
    right::ExprNode
end

"""A unary operation (negation) on an expression."""
struct UnaryOpNode <: ExprNode
    op::Symbol
    expr::ExprNode
end

"""A user-defined substitution call (expanded before discretization)."""
struct SubstitutionNode <: ExprNode
    name::Symbol
    args::Vector{ExprNode}
end

# ──────────────────────────────────────────────────────────────────────────────
#  Equality
# ──────────────────────────────────────────────────────────────────────────────

Base.:(==)(a::VarNode, b::VarNode) = a.name == b.name
Base.:(==)(a::ParamNode, b::ParamNode) = a.name == b.name
Base.:(==)(a::ConstNode, b::ConstNode) = a.value == b.value
Base.:(==)(a::EigenvalueNode, b::EigenvalueNode) = a.name == b.name
Base.:(==)(a::WavenumberNode, b::WavenumberNode) = a.name == b.name
Base.:(==)(a::DerivNode, b::DerivNode) = a.expr == b.expr && a.coord == b.coord
Base.:(==)(a::BinaryOpNode, b::BinaryOpNode) = a.op == b.op && a.left == b.left && a.right == b.right
Base.:(==)(a::UnaryOpNode, b::UnaryOpNode) = a.op == b.op && a.expr == b.expr
Base.:(==)(a::SubstitutionNode, b::SubstitutionNode) = a.name == b.name && a.args == b.args
Base.:(==)(::ExprNode, ::ExprNode) = false

# ──────────────────────────────────────────────────────────────────────────────
#  Display
# ──────────────────────────────────────────────────────────────────────────────

Base.show(io::IO, n::VarNode) = print(io, n.name)
Base.show(io::IO, n::ParamNode) = print(io, n.name)
Base.show(io::IO, n::ConstNode) = print(io, n.value)
Base.show(io::IO, n::EigenvalueNode) = print(io, n.name)
Base.show(io::IO, n::WavenumberNode) = print(io, n.name)
Base.show(io::IO, n::DerivNode) = print(io, "d", n.coord, "(", n.expr, ")")
Base.show(io::IO, n::BinaryOpNode) = print(io, "(", n.left, " ", n.op, " ", n.right, ")")
Base.show(io::IO, n::UnaryOpNode) = print(io, n.op, "(", n.expr, ")")
Base.show(io::IO, n::SubstitutionNode) = print(io, n.name, "(", join(n.args, ", "), ")")
