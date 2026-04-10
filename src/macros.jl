#══════════════════════════════════════════════════════════════════════════════#
#  DSL Macros: @equation, @bc, @substitution                                  #
#══════════════════════════════════════════════════════════════════════════════#

# ──────────────────────────────────────────────────────────────────────────────
#  Macro helper: resolve prob argument (explicit or active)
# ──────────────────────────────────────────────────────────────────────────────

"""
Determine the prob symbol for macro-generated code.
If `first_arg` is a Symbol (variable name), it's the explicit prob.
Otherwise, use the active prob via `_get_active_prob()`.
Returns `(prob_expression, remaining_args)`.
"""
function _resolve_prob_arg(first_arg, rest_args)
    if first_arg isa Symbol && isempty(rest_args)
        # Single arg — it's the expression, use active prob
        return :(_get_active_prob()), first_arg
    elseif first_arg isa Symbol
        # First arg is prob symbol, second is the expression
        return esc(first_arg), length(rest_args) == 1 ? rest_args[1] : rest_args
    else
        # First arg is the expression, use active prob
        return :(_get_active_prob()), first_arg
    end
end

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
    elseif haskey(prob.derived_vars, name)
        return VarNode(name)  # treated as variable for parsing; resolved during discretize
    elseif haskey(prob.parameters, name)
        return ParamNode(name)
    elseif haskey(prob.substitutions, name)
        return ParamNode(name)
    else
        error("Unknown symbol '$name': not a variable, parameter, eigenvalue, derived variable, or substitution")
    end
end

# ──────────────────────────────────────────────────────────────────────────────
#  AST → ExprNode parsing (for equations)
# ──────────────────────────────────────────────────────────────────────────────

"""
    parse_expr_ast(ex, prob_ref)

Walk a Julia AST expression and emit ExprNode constructor calls.
`prob_ref` is either a Symbol (variable name, will be esc'd) or an Expr
(like `_get_active_prob()`) that evaluates to the EVP at runtime.
"""
function parse_expr_ast(ex, prob_ref)
    # Build the runtime expression for the prob reference
    prob_runtime = prob_ref isa Symbol ? esc(prob_ref) : prob_ref

    if ex isa Number
        return :(ConstNode($ex))
    elseif ex isa Symbol
        return :(resolve_symbol($prob_runtime, $(QuoteNode(ex))))
    elseif ex isa Expr
        if ex.head == :call
            func = ex.args[1]
            func_str = string(func)

            # Derivative calls: dx(expr), dy(expr), dz(expr), etc.
            if length(func_str) == 2 && func_str[1] == 'd' && length(ex.args) == 2
                coord = Symbol(func_str[2])
                inner = parse_expr_ast(ex.args[2], prob_ref)
                return :(DerivNode($inner, $(QuoteNode(coord))))
            end

            # Arithmetic operators
            if func in (:+, :-, :*)
                if length(ex.args) == 2
                    # Unary (e.g., -x)
                    inner = parse_expr_ast(ex.args[2], prob_ref)
                    return :(UnaryOpNode($(QuoteNode(func)), $inner))
                elseif length(ex.args) == 3
                    left = parse_expr_ast(ex.args[2], prob_ref)
                    right = parse_expr_ast(ex.args[3], prob_ref)
                    return :(BinaryOpNode($(QuoteNode(func)), $left, $right))
                else
                    # Multi-arg: fold left
                    result = parse_expr_ast(ex.args[2], prob_ref)
                    for i in 3:length(ex.args)
                        next = parse_expr_ast(ex.args[i], prob_ref)
                        result = :(BinaryOpNode($(QuoteNode(func)), $result, $next))
                    end
                    return result
                end
            end

            # Any other function call → assume substitution
            if func isa Symbol && length(ex.args) >= 2
                args_parsed = [parse_expr_ast(a, prob_ref) for a in ex.args[2:end]]
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
    @substitution Lap(A) = dx(dx(A)) + dz(dz(A))
    @substitution prob Lap(A) = dx(dx(A)) + dz(dz(A))   # explicit prob

Define a substitution template. `prob` is optional — uses the active EVP if omitted.
"""
macro substitution(args...)
    if length(args) == 1
        prob_expr = :(_get_active_prob())
        expr = args[1]
    elseif length(args) == 2 && args[1] isa Symbol
        prob_expr = esc(args[1])
        expr = args[2]
    else
        error("@substitution expects: @substitution [prob] Name(A) = expr")
    end

    expr.head == :(=) || error("@substitution expects: Name(A) = expr")

    lhs = expr.args[1]
    rhs_expr = expr.args[2]

    name = lhs.args[1]
    arg_names = [a for a in lhs.args[2:end]]

    body_ast = parse_substitution_body(rhs_expr, arg_names)

    return quote
        add_substitution!($prob_expr, $(QuoteNode(name)),
                         Symbol[$(QuoteNode.(arg_names)...)],
                         $body_ast)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
#  @derive
# ──────────────────────────────────────────────────────────────────────────────

"""
    @derive prob Dh2_v = rhs                              # Form 1: named operator
    @derive prob v where dx(dx(v)) + dy(dy(v)) = rhs     # Form 2: inline operator

Define a derived (auxiliary) variable implicitly. The code computes `Op^{-1}`
and substitutes `v = Op^{-1} * rhs_expr` into any equation that references `v`.

**Form 1** (`Op_var = rhs`): The name is split on `_` — everything before the
last `_` is a `@substitution` name, the part after is the variable. `Dh2_v`
means "Dh2(v) = rhs".

**Form 2** (`v where lhs_expr = rhs`): The operator is defined inline as any
expression involving `v`. No separate `@substitution` needed.

`v` is NOT added to the variable list — it's eliminated automatically.

Examples:
```julia
# Form 1 (requires @substitution Dh2):
@derive prob Dh2_v = -dy(dz(w)) + dx(zeta)

# Form 2 (inline, no @substitution needed):
@derive prob v where dx(dx(v)) + dy(dy(v)) = -dy(dz(w)) + dx(zeta)

# General operator:
@derive prob v where 3*dz(dz(v)) + alpha*v = dz(w)
```
"""
macro derive(args...)
    # Detect if first arg is an explicit prob symbol
    # Form 1: @derive [prob] Op_var = rhs                 (1 or 2 args)
    # Form 2: @derive [prob] v lhs_expr(v) = rhs_expr     (2 or 3 args)
    all_args = collect(args)

    # Check if first arg is a prob symbol (it's a bare Symbol that's not an `=` expression)
    has_prob = length(all_args) >= 2 && all_args[1] isa Symbol &&
               !(all_args[1] isa Expr)
    # But in Form 1, `@derive Op_var = rhs` — the single arg is an Expr(:=)
    # and `@derive prob Op_var = rhs` — two args where first is Symbol, second is Expr(:=)
    # In Form 2, `@derive v lhs = rhs` — two args, first is Symbol, second is Expr(:=)
    # So we can't distinguish `prob Op_var = rhs` from `v lhs(v) = rhs` just by arg count.
    # Resolution: check if the second arg (if it's an Expr(:=)) has a Symbol LHS with underscore.
    # If yes → Form 1 with explicit prob. If the second arg has an expression LHS → Form 2.

    if length(all_args) == 1
        # @derive Op_var = rhs (no prob, Form 1)
        prob_ref = :(_get_active_prob())
        prob_expr = :(_get_active_prob())
        return _derive_form1(prob_ref, prob_expr, all_args[1])

    elseif length(all_args) == 2
        a1, a2 = all_args
        if a1 isa Symbol && a2 isa Expr && a2.head == :(=)
            # Could be: @derive prob Op_var = rhs (Form 1 with prob)
            # Or:       @derive v lhs(v) = rhs   (Form 2 without prob)
            # Distinguish: if a2.args[1] is a simple Symbol with underscore → Form 1 with prob
            if a2.args[1] isa Symbol && occursin('_', string(a2.args[1]))
                # Form 1 with explicit prob
                return _derive_form1(a1, esc(a1), a2)
            else
                # Form 2 without prob: a1 = var, a2 = lhs_expr = rhs
                return _derive_form2(:(_get_active_prob()), :(_get_active_prob()), a1, a2)
            end
        elseif a1 isa Symbol && a2 isa Expr
            # Form 2 without prob
            return _derive_form2(:(_get_active_prob()), :(_get_active_prob()), a1, a2)
        end

    elseif length(all_args) == 3
        # @derive prob v lhs(v) = rhs (Form 2 with explicit prob)
        prob_sym, var, eq = all_args
        return _derive_form2(prob_sym, esc(prob_sym), var, eq)
    end

    error("@derive expects: @derive [prob] Op_var = rhs  OR  @derive [prob] v lhs_expr(v) = rhs")
end

function _derive_form1(prob_ref, prob_expr, expr)
    (expr isa Expr && expr.head == :(=) && length(expr.args) == 2) ||
        error("@derive Form 1 expects: Op_var = rhs")

    lhs_sym = expr.args[1]
    rhs = expr.args[2]
    if rhs isa Expr && rhs.head == :block
        stmts = filter(a -> !(a isa LineNumberNode), rhs.args)
        length(stmts) == 1 && (rhs = stmts[1])
    end

    lhs_str = string(lhs_sym)
    last_underscore = findlast('_', lhs_str)
    isnothing(last_underscore) &&
        error("@derive LHS must be Op_var (e.g., Dh2_v), got: $lhs_sym")

    operator_name = QuoteNode(Symbol(lhs_str[1:last_underscore-1]))
    var_name = QuoteNode(Symbol(lhs_str[last_underscore+1:end]))

    rhs_ast = parse_expr_ast(rhs, prob_ref)

    return quote
        add_derived!($prob_expr, $var_name, $operator_name, $rhs_ast)
    end
end

function _derive_form2(prob_ref, prob_expr, var, eq)
    var isa Symbol || error("@derive Form 2: first arg must be a variable name")
    var_name = QuoteNode(var)

    (eq isa Expr && eq.head == :(=) && length(eq.args) == 2) ||
        error("@derive Form 2 expects: v lhs_expr(v) = rhs")

    lhs_op_expr = eq.args[1]
    rhs = eq.args[2]
    # Strip begin...end block from assignment RHS
    if rhs isa Expr && rhs.head == :block
        stmts = filter(a -> !(a isa LineNumberNode), rhs.args)
        length(stmts) == 1 && (rhs = stmts[1])
    end

    lhs_ast = parse_substitution_body(lhs_op_expr, [var])
    rhs_ast = parse_expr_ast(rhs, prob_ref)

    auto_sub_name = QuoteNode(Symbol(:_derive_, var))

    return quote
        add_substitution!($prob_expr, $auto_sub_name,
                         [$(QuoteNode(var))], $lhs_ast)
        add_derived!($prob_expr, $var_name, $auto_sub_name, $rhs_ast)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
#  @equation
# ──────────────────────────────────────────────────────────────────────────────

"""
    @equation sigma * psi == U * dx(psi)
    @equation sigma * psi = U * dx(psi)     # = also works

Add an equation LHS = RHS (or LHS == RHS). `prob` is optional.
"""
macro equation(args...)
    if length(args) == 1
        prob_ref = :(_get_active_prob())
        prob_expr = :(_get_active_prob())
        expr = args[1]
    elseif length(args) == 2 && args[1] isa Symbol
        prob_ref = args[1]
        prob_expr = esc(args[1])
        expr = args[2]
    else
        error("@equation expects: @equation [prob] lhs = rhs")
    end

    # Accept both == (comparison) and = (assignment) syntax
    lhs, rhs = _parse_equation_expr(expr)

    lhs_ast = parse_expr_ast(lhs, prob_ref)
    rhs_ast = parse_expr_ast(rhs, prob_ref)

    return quote
        add_equation!($prob_expr, $lhs_ast, $rhs_ast)
    end
end

"""Extract LHS and RHS from either `lhs == rhs` or `lhs = rhs` AST."""
function _parse_equation_expr(expr)
    # == form: Expr(:call, :(==), lhs, rhs)
    if expr isa Expr && expr.head == :call && expr.args[1] == :(==) && length(expr.args) == 3
        return expr.args[2], expr.args[3]
    end
    # = form: Expr(:(=), lhs, rhs)  — RHS may be wrapped in begin...end block
    if expr isa Expr && expr.head == :(=) && length(expr.args) == 2
        rhs = expr.args[2]
        # Strip begin...end block that Julia inserts around assignment RHS
        if rhs isa Expr && rhs.head == :block
            stmts = filter(a -> !(a isa LineNumberNode), rhs.args)
            length(stmts) == 1 && (rhs = stmts[1])
        end
        return expr.args[1], rhs
    end
    error("@equation expects: lhs = rhs  or  lhs == rhs")
end

# ──────────────────────────────────────────────────────────────────────────────
#  @derive_bc
# ──────────────────────────────────────────────────────────────────────────────

"""
    @derive_bc v left(v) == 0
    @derive_bc v right(dz(v)) == 0
    @derive_bc prob v left(v) == 0   # explicit prob

Add BCs for a derived variable's inversion. `prob` is optional.
"""
macro derive_bc(args...)
    if length(args) == 2
        prob_expr = :(_get_active_prob())
        var_name = args[1]::Symbol
        expr = args[2]
    elseif length(args) == 3 && args[1] isa Symbol
        prob_expr = esc(args[1])
        var_name = args[2]::Symbol
        expr = args[3]
    else
        error("@derive_bc expects: @derive_bc [prob] v left(v) == value")
    end

    lhs, rhs_val = _parse_equation_expr(expr)

    (lhs isa Expr && lhs.head == :call && lhs.args[1] in (:left, :right)) ||
        error("@derive_bc LHS must be left(expr) or right(expr)")

    side = QuoteNode(lhs.args[1])
    inner = lhs.args[2]

    # Count derivative order from the inner expression
    deriv_order = 0
    while inner isa Expr && inner.head == :call && length(inner.args) == 2
        fname = string(inner.args[1])
        if length(fname) == 2 && fname[1] == 'd'
            deriv_order += 1
            inner = inner.args[2]
        else
            break
        end
    end

    coord_ast = :(first_chebyshev_coord($prob_expr))

    return quote
        c = $coord_ast
        isnothing(c) && error("No Chebyshev direction for @derive_bc")
        add_derived_bc!($prob_expr, $(QuoteNode(var_name)), $side, c, $deriv_order, Float64($rhs_val))
    end
end

# ──────────────────────────────────────────────────────────────────────────────
#  @bc
# ──────────────────────────────────────────────────────────────────────────────

"""
    @bc left(psi) = 0
    @bc right(dz(psi)) = 0
    @bc left(psi) == 0        # == also works

Add a boundary condition. `prob` is optional.
"""
macro bc(args...)
    if length(args) == 1
        prob_ref = :(_get_active_prob())
        prob_expr = :(_get_active_prob())
        expr = args[1]
    elseif length(args) == 2 && args[1] isa Symbol
        prob_ref = args[1]
        prob_expr = esc(args[1])
        expr = args[2]
    else
        error("@bc expects: @bc [prob] left(expr) = value")
    end

    lhs, rhs_val = _parse_equation_expr(expr)

    (lhs isa Expr && lhs.head == :call && lhs.args[1] in (:left, :right)) ||
        error("@bc LHS must be left(expr) or right(expr)")

    side = QuoteNode(lhs.args[1])
    inner_expr = lhs.args[2]
    inner_ast = parse_expr_ast(inner_expr, prob_ref)

    coord_ast = if length(lhs.args) >= 3
        QuoteNode(lhs.args[3])
    else
        :(let c = first_chebyshev_coord($prob_expr);
            isnothing(c) && error("No Chebyshev direction found.");
            c
        end)
    end

    return quote
        add_bc!($prob_expr, $side, $coord_ast, $inner_ast, Float64($rhs_val))
    end
end
