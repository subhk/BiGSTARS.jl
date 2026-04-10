#══════════════════════════════════════════════════════════════════════════════#
#  Substitution expansion: recursively replace SubstitutionNode in trees      #
#══════════════════════════════════════════════════════════════════════════════#

"""
    expand_substitutions(expr, subs; max_depth=100)

Recursively expand all SubstitutionNode instances in the expression tree.
Detects cycles via a depth limit.
"""
function expand_substitutions(expr::ExprNode, subs::Dict{Symbol, Substitution};
                              max_depth::Int=100, _depth::Int=0)
    _depth > max_depth && error("Substitution expansion exceeded max depth ($max_depth). Possible cycle.")

    if expr isa SubstitutionNode
        sub = get(subs, expr.name, nothing)
        isnothing(sub) && error("Unknown substitution: $(expr.name)")
        length(expr.args) == length(sub.arg_names) ||
            error("Substitution $(expr.name) expects $(length(sub.arg_names)) args, got $(length(expr.args))")

        # Expand arguments first
        expanded_args = [expand_substitutions(a, subs; max_depth, _depth=_depth+1) for a in expr.args]

        # Substitute argument names in the body
        body = substitute_args(sub.body, sub.arg_names, expanded_args)

        # Recursively expand in case body contains more substitutions
        return expand_substitutions(body, subs; max_depth, _depth=_depth+1)

    elseif expr isa BinaryOpNode
        left = expand_substitutions(expr.left, subs; max_depth, _depth)
        right = expand_substitutions(expr.right, subs; max_depth, _depth)
        return BinaryOpNode(expr.op, left, right)

    elseif expr isa UnaryOpNode
        inner = expand_substitutions(expr.expr, subs; max_depth, _depth)
        return UnaryOpNode(expr.op, inner)

    elseif expr isa DerivNode
        inner = expand_substitutions(expr.expr, subs; max_depth, _depth)
        return DerivNode(inner, expr.coord)

    else
        return expr
    end
end

"""
    substitute_args(body, arg_names, arg_values)

Replace VarNode placeholders in `body` that match `arg_names` with `arg_values`.
"""
function substitute_args(body::ExprNode, arg_names::Vector{Symbol}, arg_values::Vector{<:ExprNode})
    if body isa VarNode && body.name in arg_names
        idx = findfirst(==(body.name), arg_names)
        return arg_values[idx]
    elseif body isa BinaryOpNode
        left = substitute_args(body.left, arg_names, arg_values)
        right = substitute_args(body.right, arg_names, arg_values)
        return BinaryOpNode(body.op, left, right)
    elseif body isa UnaryOpNode
        inner = substitute_args(body.expr, arg_names, arg_values)
        return UnaryOpNode(body.op, inner)
    elseif body isa DerivNode
        inner = substitute_args(body.expr, arg_names, arg_values)
        return DerivNode(inner, body.coord)
    elseif body isa SubstitutionNode
        args = [substitute_args(a, arg_names, arg_values) for a in body.args]
        return SubstitutionNode(body.name, args)
    else
        return body
    end
end

"""Check if an expression tree contains any unexpanded SubstitutionNode."""
function contains_substitution(expr::ExprNode)
    expr isa SubstitutionNode && return true
    expr isa BinaryOpNode && return contains_substitution(expr.left) || contains_substitution(expr.right)
    expr isa UnaryOpNode && return contains_substitution(expr.expr)
    expr isa DerivNode && return contains_substitution(expr.expr)
    return false
end
