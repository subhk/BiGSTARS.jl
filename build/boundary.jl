#══════════════════════════════════════════════════════════════════════════════#
#  Generalized boundary conditions with boundary bordering                    #
#══════════════════════════════════════════════════════════════════════════════#

"""
    chebyshev_T(n, x)

Evaluate Chebyshev polynomial T_n(x). At boundaries: T_n(1)=1, T_n(-1)=(-1)^n.
"""
function chebyshev_T(n::Int, x::Float64)
    if x == 1.0
        return 1.0
    elseif x == -1.0
        return (-1.0)^n
    else
        return cos(n * acos(x))
    end
end

"""
    chebyshev_T_deriv(n, p, x)

Evaluate the p-th derivative of T_n at x = ±1 (boundary points).

Uses the closed-form: T_n^(p)(1) = ∏_{j=0}^{p-1} (n² - j²) / (2j + 1)
and T_n^(p)(-1) = (-1)^(n+p) * T_n^(p)(1).
"""
function chebyshev_T_deriv(n::Int, p::Int, x::Float64)
    p == 0 && return chebyshev_T(n, x)
    n < p && return 0.0  # derivative order > polynomial degree

    # T_n^(p)(1) = ∏_{j=0}^{p-1} (n² - j²) / (2j + 1)
    val_at_1 = 1.0
    for j in 0:p-1
        val_at_1 *= (n^2 - j^2) / (2j + 1)
    end

    if x == 1.0
        return val_at_1
    elseif x == -1.0
        return (-1.0)^(n + p) * val_at_1
    else
        error("chebyshev_T_deriv only implemented at boundary points x = ±1")
    end
end

"""
    chebyshev_boundary_row(side, deriv_order, N; a=-1.0, b=1.0) -> Vector{Float64}

Construct a row vector for evaluating a Chebyshev expansion (or its derivative)
at a boundary. Used for boundary bordering in the ultraspherical method.

- `side`: `:left` (lower bound a) or `:right` (upper bound b)
- `deriv_order`: 0 for value, 1 for first derivative, etc.
- `N`: number of Chebyshev coefficients
"""
function chebyshev_boundary_row(side::Symbol, deriv_order::Int, N::Int;
                                 a::Float64=-1.0, b::Float64=1.0)
    scale = 2.0 / (b - a)
    # Evaluate on reference domain [-1, 1]
    x_eval = side == :left ? -1.0 : 1.0

    return [scale^deriv_order * chebyshev_T_deriv(n - 1, deriv_order, x_eval) for n in 1:N]
end

"""
    count_bc_deriv_order(expr)

Count the derivative order in a BC expression (e.g., dz(dz(psi)) → 2).
"""
function count_bc_deriv_order(expr::ExprNode)
    if expr isa DerivNode
        return 1 + count_bc_deriv_order(expr.expr)
    elseif expr isa BinaryOpNode
        return max(count_bc_deriv_order(expr.left), count_bc_deriv_order(expr.right))
    elseif expr isa UnaryOpNode
        return count_bc_deriv_order(expr.expr)
    else
        return 0
    end
end
