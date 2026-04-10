#══════════════════════════════════════════════════════════════════════════════#
#  Wavenumber separation: tag, extract k-power, split expression trees        #
#══════════════════════════════════════════════════════════════════════════════#

"""A term in the equation with a known power of k."""
struct KTerm
    k_power::Int
    expr::ExprNode
end

"""Check if an expression tree depends on wavenumber k."""
is_k_dependent(expr::ExprNode) = contains_wavenumber(expr)

"""
    separate_additive_terms(expr)

Flatten an addition tree into a list of individual terms.
`a + b - c` becomes `[a, b, -1*c]`.
"""
function separate_additive_terms(expr::ExprNode)
    terms = ExprNode[]
    _collect_terms!(terms, expr, true)
    return terms
end

function _collect_terms!(terms::Vector{ExprNode}, expr::ExprNode, positive::Bool)
    if expr isa BinaryOpNode && expr.op == :+
        _collect_terms!(terms, expr.left, positive)
        _collect_terms!(terms, expr.right, positive)
    elseif expr isa BinaryOpNode && expr.op == :-
        _collect_terms!(terms, expr.left, positive)
        _collect_terms!(terms, expr.right, !positive)
    elseif expr isa UnaryOpNode && expr.op == :-
        _collect_terms!(terms, expr.expr, !positive)
    else
        if positive
            push!(terms, expr)
        else
            push!(terms, BinaryOpNode(:*, ConstNode(-1.0), expr))
        end
    end
end

"""
    extract_k_power(expr) -> (power::Int, reduced::ExprNode)

Determine the power of k in a multiplicative term and return the term
with k nodes removed. Only works on product terms (not sums).
"""
function extract_k_power(expr::ExprNode)
    return _extract_k(expr)
end

"""
Extract wavenumber power from a multiplicative term.

Design: the `im` factors from Fourier lowering (`dx → im*k`) are intentionally
left in the reduced expression tree. They get folded into the discretized matrix
during `discretize_expr`, so `assemble` only needs `k^p` (not `(im*k)^p`).
"""
function _extract_k(expr::ExprNode)
    if expr isa WavenumberNode
        return 1, ConstNode(1.0)
    elseif expr isa ConstNode && (expr.value === im || expr.value == im)
        # im is part of the k lowering: im*k → keep im in reduced, count k separately
        return 0, expr
    elseif expr isa BinaryOpNode && expr.op == :*
        lp, lr = _extract_k(expr.left)
        rp, rr = _extract_k(expr.right)
        total = lp + rp
        # Simplify: skip ConstNode(1.0) factors produced by wavenumber extraction
        if lr isa ConstNode && lr.value == 1.0
            reduced = rr
        elseif rr isa ConstNode && rr.value == 1.0
            reduced = lr
        elseif lr isa ConstNode && rr isa ConstNode
            # Merge two constant factors into one
            reduced = ConstNode(lr.value * rr.value)
        else
            reduced = BinaryOpNode(:*, lr, rr)
        end
        return total, reduced
    elseif expr isa UnaryOpNode
        p, r = _extract_k(expr.expr)
        return p, UnaryOpNode(expr.op, r)
    elseif expr isa DerivNode
        p, r = _extract_k(expr.expr)
        return p, DerivNode(r, expr.coord)
    else
        return 0, expr
    end
end

"""
    distribute_products(expr)

Distribute multiplication over addition/subtraction so that every additive
term is a pure product chain (no sums inside products).

`a * (b + c)` → `a*b + a*c`
`(a - b) * c` → `a*c - b*c`

This is essential for k-separation: after derivative lowering, expressions like
`im*k * (k²*psi + dy(dy(psi)))` must become `im*k³*psi + im*k*dy(dy(psi))`
so that each additive term has a single k-power.
"""
function distribute_products(expr::ExprNode)
    if expr isa BinaryOpNode && expr.op == :*
        left = distribute_products(expr.left)
        right = distribute_products(expr.right)

        # (a + b) * c → a*c + b*c;  (a - b) * c → a*c - b*c
        if left isa BinaryOpNode && left.op in (:+, :-)
            new_left = distribute_products(BinaryOpNode(:*, left.left, right))
            new_right = distribute_products(BinaryOpNode(:*, left.right, right))
            return BinaryOpNode(left.op, new_left, new_right)
        end

        # a * (b + c) → a*b + a*c;  a * (b - c) → a*b - a*c
        if right isa BinaryOpNode && right.op in (:+, :-)
            new_left = distribute_products(BinaryOpNode(:*, left, right.left))
            new_right = distribute_products(BinaryOpNode(:*, left, right.right))
            return BinaryOpNode(right.op, new_left, new_right)
        end

        return BinaryOpNode(:*, left, right)

    elseif expr isa BinaryOpNode
        left = distribute_products(expr.left)
        right = distribute_products(expr.right)
        return BinaryOpNode(expr.op, left, right)

    elseif expr isa UnaryOpNode
        return UnaryOpNode(expr.op, distribute_products(expr.expr))

    elseif expr isa DerivNode
        # Differentiation is linear: d(A + B) → dA + dB, d(A - B) → dA - dB
        inner = distribute_products(expr.expr)
        if inner isa BinaryOpNode && inner.op in (:+, :-)
            new_left = distribute_products(DerivNode(inner.left, expr.coord))
            new_right = distribute_products(DerivNode(inner.right, expr.coord))
            return BinaryOpNode(inner.op, new_left, new_right)
        end
        return DerivNode(inner, expr.coord)

    else
        return expr
    end
end

"""
    separate_by_k_power(expr) -> Vector{KTerm}

Split an equation RHS expression (after substitution expansion and derivative
lowering) into terms grouped by power of k.

First distributes products over sums so each additive term is a pure product
chain, then extracts the k-power from each term.
"""
function separate_by_k_power(expr::ExprNode)
    distributed = distribute_products(expr)
    terms = separate_additive_terms(distributed)
    k_terms = KTerm[]
    for t in terms
        power, reduced = extract_k_power(t)
        push!(k_terms, KTerm(power, reduced))
    end
    return k_terms
end
