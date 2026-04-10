#══════════════════════════════════════════════════════════════════════════════#
#  Derivative lowering: FourierTransformed dx → im*k, tree utilities          #
#══════════════════════════════════════════════════════════════════════════════#

"""
    lower_derivatives(expr, domain)

Replace DerivNode in FourierTransformed directions with im*k multiplication.
Resolved directions (Fourier, Chebyshev) are left as DerivNode.
"""
function lower_derivatives(expr::ExprNode, domain::Domain)
    if expr isa DerivNode
        inner = lower_derivatives(expr.expr, domain)
        coord = expr.coord
        spec = get(domain.coords, coord, nothing)
        if spec isa FourierTransformed
            # dx(f) → im * k_x * f
            k_node = WavenumberNode(Symbol(:k_, coord))
            return BinaryOpNode(:*, BinaryOpNode(:*, ConstNode(im), k_node), inner)
        else
            return DerivNode(inner, coord)
        end
    elseif expr isa BinaryOpNode
        left = lower_derivatives(expr.left, domain)
        right = lower_derivatives(expr.right, domain)
        return BinaryOpNode(expr.op, left, right)
    elseif expr isa UnaryOpNode
        inner = lower_derivatives(expr.expr, domain)
        return UnaryOpNode(expr.op, inner)
    else
        return expr
    end
end

"""Check if tree contains a WavenumberNode."""
function contains_wavenumber(expr::ExprNode)
    expr isa WavenumberNode && return true
    expr isa BinaryOpNode && return contains_wavenumber(expr.left) || contains_wavenumber(expr.right)
    expr isa UnaryOpNode && return contains_wavenumber(expr.expr)
    expr isa DerivNode && return contains_wavenumber(expr.expr)
    return false
end

"""Check if tree contains a DerivNode in a specific direction."""
function any_deriv_in_direction(expr::ExprNode, coord::Symbol)
    expr isa DerivNode && expr.coord == coord && return true
    expr isa DerivNode && return any_deriv_in_direction(expr.expr, coord)
    expr isa BinaryOpNode && return any_deriv_in_direction(expr.left, coord) || any_deriv_in_direction(expr.right, coord)
    expr isa UnaryOpNode && return any_deriv_in_direction(expr.expr, coord)
    return false
end

"""Find the maximum derivative order in a given resolved direction."""
function max_deriv_order(expr::ExprNode, coord::Symbol)
    if expr isa DerivNode && expr.coord == coord
        return 1 + max_deriv_order(expr.expr, coord)
    elseif expr isa DerivNode
        return max_deriv_order(expr.expr, coord)
    elseif expr isa BinaryOpNode
        return max(max_deriv_order(expr.left, coord), max_deriv_order(expr.right, coord))
    elseif expr isa UnaryOpNode
        return max_deriv_order(expr.expr, coord)
    else
        return 0
    end
end

"""Count consecutive DerivNodes in the same direction."""
function count_chained_derivs(expr::DerivNode, coord::Symbol)
    if expr.expr isa DerivNode && expr.expr.coord == coord
        return 1 + count_chained_derivs(expr.expr, coord)
    end
    return 1
end

"""Unwrap consecutive DerivNodes, returning the inner expression."""
function unwrap_chained_derivs(expr::DerivNode, coord::Symbol)
    if expr.expr isa DerivNode && expr.expr.coord == coord
        return unwrap_chained_derivs(expr.expr, coord)
    end
    return expr.expr
end

"""Collect all VarNode names in an expression."""
function collect_var_names(expr::ExprNode)
    names = Set{Symbol}()
    _collect_vars!(names, expr)
    return names
end

function _collect_vars!(names::Set{Symbol}, expr::ExprNode)
    if expr isa VarNode
        push!(names, expr.name)
    elseif expr isa BinaryOpNode
        _collect_vars!(names, expr.left)
        _collect_vars!(names, expr.right)
    elseif expr isa UnaryOpNode
        _collect_vars!(names, expr.expr)
    elseif expr isa DerivNode
        _collect_vars!(names, expr.expr)
    elseif expr isa SubstitutionNode
        for a in expr.args
            _collect_vars!(names, a)
        end
    end
end
