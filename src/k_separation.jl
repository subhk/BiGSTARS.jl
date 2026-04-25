#══════════════════════════════════════════════════════════════════════════════#
#  Wavenumber separation: tag, extract k-power, split expression trees        #
#══════════════════════════════════════════════════════════════════════════════#

const KPowerKey = Tuple{Vararg{Pair{Symbol, Int}}}

"""A term in the equation with a known power of k."""
struct KTerm
    k_power::Int
    k_powers::KPowerKey
    expr::ExprNode

    KTerm(k_power::Int, expr::ExprNode) = new(k_power, (), expr)
    KTerm(k_powers::KPowerKey, expr::ExprNode) =
        new(sum(last, k_powers; init=0), k_powers, expr)
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

"""Like `extract_k_power`, but preserves powers for each wavenumber symbol."""
function extract_k_powers(expr::ExprNode)
    return _extract_k_powers(expr)
end

function _merge_k_powers(a::KPowerKey, b::KPowerKey)
    powers = Dict{Symbol, Int}()
    for (name, p) in a
        powers[name] = get(powers, name, 0) + p
    end
    for (name, p) in b
        powers[name] = get(powers, name, 0) + p
    end
    return Tuple(name => powers[name] for name in sort(collect(keys(powers))) if powers[name] != 0)
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

function _extract_k_powers(expr::ExprNode)
    if expr isa WavenumberNode
        return (expr.name => 1,), ConstNode(1.0)
    elseif expr isa ConstNode && (expr.value === im || expr.value == im)
        # im is part of derivative lowering; keep it in the matrix coefficient.
        return (), expr
    elseif expr isa BinaryOpNode && expr.op == :*
        lp, lr = _extract_k_powers(expr.left)
        rp, rr = _extract_k_powers(expr.right)
        powers = _merge_k_powers(lp, rp)

        if lr isa ConstNode && lr.value == 1.0
            reduced = rr
        elseif rr isa ConstNode && rr.value == 1.0
            reduced = lr
        elseif lr isa ConstNode && rr isa ConstNode
            reduced = ConstNode(lr.value * rr.value)
        else
            reduced = BinaryOpNode(:*, lr, rr)
        end
        return powers, reduced
    elseif expr isa UnaryOpNode
        powers, reduced = _extract_k_powers(expr.expr)
        return powers, UnaryOpNode(expr.op, reduced)
    elseif expr isa DerivNode
        powers, reduced = _extract_k_powers(expr.expr)
        return powers, DerivNode(reduced, expr.coord)
    else
        return (), expr
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
        powers, reduced = extract_k_powers(t)
        push!(k_terms, KTerm(powers, reduced))
    end
    return k_terms
end
