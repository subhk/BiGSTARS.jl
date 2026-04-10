#══════════════════════════════════════════════════════════════════════════════#
#  DSL Macros: @equation, @bc, @substitution                                  #
#══════════════════════════════════════════════════════════════════════════════#

# ──────────────────────────────────────────────────────────────────────────────
#  Runtime symbol resolution
# ──────────────────────────────────────────────────────────────────────────────

"""
    resolve_symbol(prob, name)

At runtime, determine whether a symbol is a variable, parameter, or eigenvalue.
"""
function resolve_symbol(prob::EVP, name::Symbol)
    if name == prob.eigenvalue
        return EigenvalueNode(name)
    elseif name in prob.variables
        return VarNode(name)
    elseif haskey(prob.parameters, name)
        return ParamNode(name)
    elseif haskey(prob.substitutions, name)
        return ParamNode(name)
    else
        error("Unknown symbol '$name': not a variable, parameter, eigenvalue, or substitution")
    end
end

# ──────────────────────────────────────────────────────────────────────────────
#  AST → ExprNode parsing (for equations)
# ──────────────────────────────────────────────────────────────────────────────

"""
    parse_expr_ast(ex, prob_sym)

Walk a Julia AST expression and emit ExprNode constructor calls.
`prob_sym` is the symbol of the EVP variable for runtime resolution.
"""
function parse_expr_ast(ex, prob_sym::Symbol)
    if ex isa Number
        return :(ConstNode($ex))
    elseif ex isa Symbol
        return :(resolve_symbol($(esc(prob_sym)), $(QuoteNode(ex))))
    elseif ex isa Expr
        if ex.head == :call
            func = ex.args[1]
            func_str = string(func)

            # Derivative calls: dx(expr), dy(expr), dz(expr), etc.
            if length(func_str) == 2 && func_str[1] == 'd' && length(ex.args) == 2
                coord = Symbol(func_str[2])
                inner = parse_expr_ast(ex.args[2], prob_sym)
                return :(DerivNode($inner, $(QuoteNode(coord))))
            end

            # Arithmetic operators
            if func in (:+, :-, :*)
                if length(ex.args) == 2
                    # Unary (e.g., -x)
                    inner = parse_expr_ast(ex.args[2], prob_sym)
                    return :(UnaryOpNode($(QuoteNode(func)), $inner))
                elseif length(ex.args) == 3
                    left = parse_expr_ast(ex.args[2], prob_sym)
                    right = parse_expr_ast(ex.args[3], prob_sym)
                    return :(BinaryOpNode($(QuoteNode(func)), $left, $right))
                else
                    # Multi-arg: fold left
                    result = parse_expr_ast(ex.args[2], prob_sym)
                    for i in 3:length(ex.args)
                        next = parse_expr_ast(ex.args[i], prob_sym)
                        result = :(BinaryOpNode($(QuoteNode(func)), $result, $next))
                    end
                    return result
                end
            end

            # Any other function call → assume substitution
            if func isa Symbol && length(ex.args) >= 2
                args_parsed = [parse_expr_ast(a, prob_sym) for a in ex.args[2:end]]
                return :(SubstitutionNode($(QuoteNode(func)), ExprNode[$(args_parsed...)]))
            end
        end
    end
    error("Cannot parse expression: $ex")
end

# ──────────────────────────────────────────────────────────────────────────────
#  AST → ExprNode parsing (for substitution bodies)
# ──────────────────────────────────────────────────────────────────────────────

"""Parse a substitution body. Argument names become VarNode placeholders."""
function parse_substitution_body(ex, arg_names::Vector)
    if ex isa Number
        return :(ConstNode($ex))
    elseif ex isa Symbol
        if ex in arg_names
            return :(VarNode($(QuoteNode(ex))))
        else
            return :(ParamNode($(QuoteNode(ex))))
        end
    elseif ex isa Expr && ex.head == :call
        func = ex.args[1]
        func_str = string(func)

        # Derivative
        if length(func_str) == 2 && func_str[1] == 'd' && length(ex.args) == 2
            coord = Symbol(func_str[2])
            inner = parse_substitution_body(ex.args[2], arg_names)
            return :(DerivNode($inner, $(QuoteNode(coord))))
        end

        # Arithmetic
        if func in (:+, :-, :*)
            if length(ex.args) == 2
                inner = parse_substitution_body(ex.args[2], arg_names)
                return :(UnaryOpNode($(QuoteNode(func)), $inner))
            elseif length(ex.args) == 3
                left = parse_substitution_body(ex.args[2], arg_names)
                right = parse_substitution_body(ex.args[3], arg_names)
                return :(BinaryOpNode($(QuoteNode(func)), $left, $right))
            else
                result = parse_substitution_body(ex.args[2], arg_names)
                for i in 3:length(ex.args)
                    next = parse_substitution_body(ex.args[i], arg_names)
                    result = :(BinaryOpNode($(QuoteNode(func)), $result, $next))
                end
                return result
            end
        end

        # Substitution call in body (nested substitutions)
        args_parsed = [parse_substitution_body(a, arg_names) for a in ex.args[2:end]]
        return :(SubstitutionNode($(QuoteNode(func)), ExprNode[$(args_parsed...)]))
    end
    # Handle begin...end blocks (Julia may wrap RHS of `=` in a block)
    if ex isa Expr && ex.head == :block
        # Filter out LineNumberNodes, recurse on the remaining expression
        exprs = filter(a -> !(a isa LineNumberNode), ex.args)
        if length(exprs) == 1
            return parse_substitution_body(exprs[1], arg_names)
        end
    end
    error("Cannot parse substitution body: $ex")
end

# ──────────────────────────────────────────────────────────────────────────────
#  @substitution
# ──────────────────────────────────────────────────────────────────────────────

"""
    @substitution prob Lap(A) = dx(dx(A)) + dz(dz(A))

Define a substitution template on the problem.
"""
macro substitution(prob, expr)
    expr.head == :(=) || error("@substitution expects: @substitution prob Name(A) = expr")

    lhs = expr.args[1]
    rhs_expr = expr.args[2]

    name = lhs.args[1]
    arg_names = [a for a in lhs.args[2:end]]

    prob_sym = prob::Symbol
    body_ast = parse_substitution_body(rhs_expr, arg_names)

    return quote
        add_substitution!($(esc(prob_sym)), $(QuoteNode(name)),
                         Symbol[$(QuoteNode.(arg_names)...)],
                         $body_ast)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
#  @equation
# ──────────────────────────────────────────────────────────────────────────────

"""
    @equation prob sigma * psi == U * dx(psi) - E * dz(dz(psi))

Add an equation LHS == RHS to the problem.
"""
macro equation(prob, expr)
    (expr isa Expr && expr.head == :call && expr.args[1] == :(==) && length(expr.args) == 3) ||
        error("@equation expects: @equation prob lhs == rhs")

    prob_sym = prob::Symbol
    lhs_ast = parse_expr_ast(expr.args[2], prob_sym)
    rhs_ast = parse_expr_ast(expr.args[3], prob_sym)

    return quote
        add_equation!($(esc(prob_sym)), $lhs_ast, $rhs_ast)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
#  @bc
# ──────────────────────────────────────────────────────────────────────────────

"""
    @bc prob left(psi) == 0
    @bc prob right(dz(psi)) == 0

Add a boundary condition to the problem.
"""
macro bc(prob, expr)
    (expr isa Expr && expr.head == :call && expr.args[1] == :(==) && length(expr.args) == 3) ||
        error("@bc expects: @bc prob left(expr) == value")

    prob_sym = prob::Symbol
    lhs = expr.args[2]
    rhs_val = expr.args[3]

    # lhs should be left(...) or right(...)
    (lhs isa Expr && lhs.head == :call && lhs.args[1] in (:left, :right)) ||
        error("@bc LHS must be left(expr) or right(expr)")

    side = QuoteNode(lhs.args[1])
    inner_expr = lhs.args[2]
    inner_ast = parse_expr_ast(inner_expr, prob_sym)

    # Optional coordinate qualification: left(expr, :z)
    coord_ast = if length(lhs.args) >= 3
        QuoteNode(lhs.args[3])
    else
        :(first_chebyshev_coord($(esc(prob_sym))))
    end

    return quote
        add_bc!($(esc(prob_sym)), $side, $coord_ast, $inner_ast, Float64($rhs_val))
    end
end
