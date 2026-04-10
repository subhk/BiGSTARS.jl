#══════════════════════════════════════════════════════════════════════════════#
#  Discretize: expression tree → sparse matrices, k-separated caching         #
#══════════════════════════════════════════════════════════════════════════════#

# ──────────────────────────────────────────────────────────────────────────────
#  Cache type
# ──────────────────────────────────────────────────────────────────────────────

"""
Cache of pre-discretized sparse matrix components, separated by k-power.
`A(k) = Σ_p k^p * A_components[p]`, same for B.

Note: the `im` factors from Fourier lowering (`dx → im*k`) are already baked
into the discretized matrices in each component, so assembly uses `k^p` not `(im*k)^p`.
"""
struct DiscretizationCache
    A_components::Dict{Int, SparseMatrixCSC{ComplexF64, Int}}
    B_components::Dict{Int, SparseMatrixCSC{ComplexF64, Int}}
    N_total::Int
    domain::Domain
end

# ──────────────────────────────────────────────────────────────────────────────
#  Validation
# ──────────────────────────────────────────────────────────────────────────────

function validate_problem(prob::EVP)
    length(prob.equations) == length(prob.variables) ||
        error("Need $(length(prob.variables)) equations, got $(length(prob.equations))")
end

# ──────────────────────────────────────────────────────────────────────────────
#  Core: expression tree → sparse matrix for a single resolved-direction term
# ──────────────────────────────────────────────────────────────────────────────

"""
    discretize_expr(expr, prob, N_per_var, highest_cheb_order) -> SparseMatrixCSC

Convert an expression tree (no wavenumber or substitution nodes) to a sparse
matrix in coefficient space. This handles the 1D Chebyshev case; for 2D with
both Fourier and Chebyshev, Kronecker products are needed.
"""
function _try_scalar(expr::ExprNode, prob::EVP)::Union{ComplexF64, Nothing}
    if expr isa ConstNode
        return ComplexF64(expr.value)
    elseif expr isa ParamNode && haskey(prob.parameters, expr.name) &&
           prob.parameters[expr.name] isa Number
        return ComplexF64(prob.parameters[expr.name])
    elseif expr isa UnaryOpNode && expr.op == :-
        inner = _try_scalar(expr.expr, prob)
        return isnothing(inner) ? nothing : -inner
    elseif expr isa BinaryOpNode && expr.op == :*
        l = _try_scalar(expr.left, prob)
        r = _try_scalar(expr.right, prob)
        return (isnothing(l) || isnothing(r)) ? nothing : l * r
    end
    return nothing
end

function discretize_expr(expr::ExprNode, prob::EVP, N_per_var::Int, highest_cheb_order::Int)
    domain = prob.domain

    if expr isa VarNode
        # Identity converted to C^(highest_cheb_order), lifted to full grid
        return _full_conversion(domain, highest_cheb_order)

    elseif expr isa ParamNode
        val = prob.parameters[expr.name]
        if val isa Number
            return ComplexF64(val) .* _full_conversion(domain, highest_cheb_order)
        else
            # Field parameter: build multiplication operator, lift, apply conversion
            cheb_dim = _find_chebyshev_dim(domain)
            if !isnothing(cheb_dim)
                spec = domain.coords[cheb_dim]
                f_ref = Float64.(vec(val))
                c = chebyshev_coefficients(f_ref)
                M_1d = ComplexF64.(multiplication_operator(c, spec.N))
                S_1d = ComplexF64.(get_conversion_operator(domain, cheb_dim, 0, highest_cheb_order))
                # S * M in 1D, then lift to full grid
                op_1d = S_1d * M_1d
                return _lift_to_2d(op_1d, cheb_dim, domain)
            else
                return ComplexF64.(spdiagm(0 => vec(val)))
            end
        end

    elseif expr isa ConstNode
        return ComplexF64(expr.value) .* _full_conversion(domain, highest_cheb_order)

    elseif expr isa DerivNode
        coord = expr.coord
        spec = domain.coords[coord]

        if spec isa ChebyshevBasisSpec
            order = count_chained_derivs(expr, coord)
            inner_expr = unwrap_chained_derivs(expr, coord)

            scale = 2.0 / (spec.upper - spec.lower)
            N = spec.N

            # Build 1D derivative chain: D_{order-1} * ... * D_0
            D_chain_1d = sparse(ComplexF64(1.0) * I, N, N)
            for p in 0:order-1
                D_chain_1d = ComplexF64(scale) .* differentiation_operator(p, N) * D_chain_1d
            end

            # Convert from C^(order) to C^(highest)
            S_rest_1d = ComplexF64.(get_conversion_operator(domain, coord, order, highest_cheb_order))

            # Lift the combined 1D operator to full grid
            op_1d = S_rest_1d * D_chain_1d
            D_full = _lift_to_2d(op_1d, coord, domain)

            # Inner expression in T basis (no conversion)
            inner_mat_T = discretize_expr_in_T(inner_expr, prob, N_per_var)

            return D_full * inner_mat_T

        elseif spec isa FourierBasisSpec
            inner_mat = discretize_expr(expr.expr, prob, N_per_var, highest_cheb_order)
            D_1d = fourier_diff_operator(spec.N, spec.L, 1)
            D_full = _lift_to_2d(D_1d, coord, domain)
            return D_full * inner_mat
        else
            error("Cannot differentiate in FourierTransformed direction (should have been lowered)")
        end

    elseif expr isa BinaryOpNode
        if expr.op == :+
            return discretize_expr(expr.left, prob, N_per_var, highest_cheb_order) +
                   discretize_expr(expr.right, prob, N_per_var, highest_cheb_order)
        elseif expr.op == :-
            return discretize_expr(expr.left, prob, N_per_var, highest_cheb_order) -
                   discretize_expr(expr.right, prob, N_per_var, highest_cheb_order)
        elseif expr.op == :*
            lv = _try_scalar(expr.left, prob)
            rv = _try_scalar(expr.right, prob)
            if !isnothing(lv) && !isnothing(rv)
                return (lv * rv) .* _full_conversion(domain, highest_cheb_order)
            elseif !isnothing(lv)
                return lv .* discretize_expr(expr.right, prob, N_per_var, highest_cheb_order)
            elseif !isnothing(rv)
                return rv .* discretize_expr(expr.left, prob, N_per_var, highest_cheb_order)
            else
                # Field parameter × operator: S * M_f * S^{-1} * G
                cheb_dim = _find_chebyshev_dim(domain)
                left_fp = _extract_field_param(expr.left, prob)
                right_fp = _extract_field_param(expr.right, prob)

                if !isnothing(left_fp) && !isnothing(cheb_dim)
                    param, sc = left_fp
                    M_f = _field_multiply_T(param, sc, prob, cheb_dim)
                    G = discretize_expr(expr.right, prob, N_per_var, highest_cheb_order)
                    return _apply_field_multiply(M_f, G, domain, cheb_dim, highest_cheb_order, N_per_var)
                elseif !isnothing(right_fp) && !isnothing(cheb_dim)
                    param, sc = right_fp
                    M_f = _field_multiply_T(param, sc, prob, cheb_dim)
                    G = discretize_expr(expr.left, prob, N_per_var, highest_cheb_order)
                    return _apply_field_multiply(M_f, G, domain, cheb_dim, highest_cheb_order, N_per_var)
                else
                    return discretize_expr(expr.left, prob, N_per_var, highest_cheb_order) *
                           discretize_expr(expr.right, prob, N_per_var, highest_cheb_order)
                end
            end
        end

    elseif expr isa UnaryOpNode
        if expr.op == :-
            return -discretize_expr(expr.expr, prob, N_per_var, highest_cheb_order)
        end
    end

    error("Cannot discretize: $(typeof(expr))")
end

"""
Discretize expression in the T basis (no conversion to C^(p)).
Used as input to derivative operators.
"""
function discretize_expr_in_T(expr::ExprNode, prob::EVP, N_per_var::Int)
    domain = prob.domain

    if expr isa VarNode
        return sparse(ComplexF64(1.0) * I, N_per_var, N_per_var)

    elseif expr isa ParamNode
        val = prob.parameters[expr.name]
        cheb_dim = _find_chebyshev_dim(domain)
        if val isa Number
            return ComplexF64(val) * sparse(I, N_per_var, N_per_var)
        else
            if !isnothing(cheb_dim)
                spec = domain.coords[cheb_dim]
                f_ref = Float64.(vec(val))
                c = chebyshev_coefficients(f_ref)
                M_1d = ComplexF64.(multiplication_operator(c, spec.N))
                return _lift_to_2d(M_1d, cheb_dim, domain)
            else
                return ComplexF64.(spdiagm(0 => vec(val)))
            end
        end

    elseif expr isa ConstNode
        return ComplexF64(expr.value) * sparse(I, N_per_var, N_per_var)

    elseif expr isa BinaryOpNode
        if expr.op == :*
            return discretize_expr_in_T(expr.left, prob, N_per_var) *
                   discretize_expr_in_T(expr.right, prob, N_per_var)
        elseif expr.op == :+
            return discretize_expr_in_T(expr.left, prob, N_per_var) +
                   discretize_expr_in_T(expr.right, prob, N_per_var)
        elseif expr.op == :-
            return discretize_expr_in_T(expr.left, prob, N_per_var) -
                   discretize_expr_in_T(expr.right, prob, N_per_var)
        end

    elseif expr isa UnaryOpNode && expr.op == :-
        return -discretize_expr_in_T(expr.expr, prob, N_per_var)

    elseif expr isa DerivNode
        coord = expr.coord
        spec = domain.coords[coord]
        if spec isa FourierBasisSpec
            # Fourier derivatives: diagonal operator in coefficient space
            inner_mat = discretize_expr_in_T(expr.expr, prob, N_per_var)
            D_1d = fourier_diff_operator(spec.N, spec.L, 1)
            D_full = _lift_to_2d(D_1d, coord, domain)
            return D_full * inner_mat
        elseif spec isa ChebyshevBasisSpec
            # Nested Chebyshev derivatives (e.g., from D4 = D2(D2(...))).
            # Apply D_chain then convert back to T via S^{-1} so the
            # result stays in T basis for the caller.
            order = count_chained_derivs(expr, coord)
            inner = unwrap_chained_derivs(expr, coord)
            scale = 2.0 / (spec.upper - spec.lower)
            N = spec.N

            D_chain_1d = sparse(ComplexF64(1.0) * I, N, N)
            for p in 0:order-1
                D_chain_1d = ComplexF64(scale) .* differentiation_operator(p, N) * D_chain_1d
            end

            # D_chain maps T → C^(order). Convert back to T via S^{-1}.
            S_1d = ComplexF64.(get_conversion_operator(domain, coord, 0, order))
            S_inv_1d = sparse(Matrix(S_1d) \ Matrix(ComplexF64(1.0) * I(N)))

            # Lift the combined 1D operator to full grid (Kronecker product)
            op_1d = S_inv_1d * D_chain_1d
            D_full = _lift_to_2d(op_1d, coord, domain)

            inner_T = discretize_expr_in_T(inner, prob, N_per_var)
            return D_full * inner_T
        else
            error("Cannot differentiate in FourierTransformed direction in T basis")
        end
    end

    error("Cannot discretize in T basis: $(typeof(expr))")
end

function _find_chebyshev_dim(domain::Domain)
    for dim in domain.resolved_dims
        if domain.coords[dim] isa ChebyshevBasisSpec
            return dim
        end
    end
    return nothing
end

function _find_fourier_dim(domain::Domain)
    for dim in domain.resolved_dims
        if domain.coords[dim] isa FourierBasisSpec
            return dim
        end
    end
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
#  Kronecker product helpers for multi-dimensional lifting
# ──────────────────────────────────────────────────────────────────────────────

"""
    _lift_to_2d(op_1d, dim, domain) -> SparseMatrixCSC

Lift a 1D operator to the full resolved 2D space using Kronecker products.

For a domain with resolved dims [y(Fourier, N_y), z(Chebyshev, N_z)]:
- A Chebyshev operator `op_z` (N_z × N_z) becomes `I_y ⊗ op_z` (N_y*N_z × N_y*N_z)
- A Fourier operator `op_y` (N_y × N_y) becomes `op_y ⊗ I_z` (N_y*N_z × N_y*N_z)

Ordering convention: z varies fastest (column-major), consistent with vec(Y × Z grid).
"""
function _lift_to_2d(op_1d::AbstractMatrix, dim::Symbol, domain::Domain)
    resolved = domain.resolved_dims
    length(resolved) == 1 && return op_1d  # 1D: no lifting needed

    # 2D case: find which dimension this operator acts on
    @assert length(resolved) == 2 "Only 1D and 2D resolved domains supported"

    dim1, dim2 = resolved[1], resolved[2]
    N1 = get_N(domain, dim1)
    N2 = get_N(domain, dim2)

    if dim == dim1
        # First resolved dim: op ⊗ I_2
        return kron(sparse(op_1d), sparse(ComplexF64(1.0) * I, N2, N2))
    elseif dim == dim2
        # Second resolved dim: I_1 ⊗ op
        return kron(sparse(ComplexF64(1.0) * I, N1, N1), sparse(op_1d))
    else
        error("Dimension :$dim not in resolved dims $resolved")
    end
end

"""Identity operator on the full resolved grid."""
function _full_identity(domain::Domain)
    N = total_grid_size(domain)
    return sparse(ComplexF64(1.0) * I, N, N)
end

"""Conversion operator S_{0→p} lifted to the full resolved grid."""
function _full_conversion(domain::Domain, highest_cheb_order::Int)
    cheb_dim = _find_chebyshev_dim(domain)
    if isnothing(cheb_dim)
        return _full_identity(domain)
    end
    S = ComplexF64.(get_conversion_operator(domain, cheb_dim, 0, highest_cheb_order))
    return _lift_to_2d(S, cheb_dim, domain)
end

"""
Extract field parameter info from an expression that may be wrapped in negation
or scalar multiplication. Returns (ParamNode, scalar_factor) or nothing.
"""
function _extract_field_param(expr::ExprNode, prob::EVP)
    if expr isa ParamNode && haskey(prob.parameters, expr.name) &&
       !(prob.parameters[expr.name] isa Number)
        return (expr, ComplexF64(1.0))
    elseif expr isa UnaryOpNode && expr.op == :-
        inner = _extract_field_param(expr.expr, prob)
        return isnothing(inner) ? nothing : (inner[1], -inner[2])
    elseif expr isa BinaryOpNode && expr.op == :*
        # Check for scalar * field_param
        sv = _try_scalar(expr.left, prob)
        if !isnothing(sv)
            fp = _extract_field_param(expr.right, prob)
            return isnothing(fp) ? nothing : (fp[1], sv * fp[2])
        end
        sv = _try_scalar(expr.right, prob)
        if !isnothing(sv)
            fp = _extract_field_param(expr.left, prob)
            return isnothing(fp) ? nothing : (fp[1], sv * fp[2])
        end
    end
    return nothing
end

"""Build the T-basis multiplication operator for a field parameter with scaling."""
function _field_multiply_T(param::ParamNode, scale::ComplexF64, prob::EVP, cheb_dim::Symbol)
    spec = prob.domain.coords[cheb_dim]
    f_ref = Float64.(vec(prob.parameters[param.name]))
    c = chebyshev_coefficients(f_ref)
    return scale .* ComplexF64.(multiplication_operator(c, spec.N))
end

"""
    _apply_field_multiply(M_f, G, domain, cheb_dim, highest, N_per_var)

Apply field multiplication in the correct basis: `S * M_f * S^{-1} * G`.

M_f is the 1D T-basis multiplication operator (N_z × N_z).
G is the full-grid operator (N_per_var × N_per_var).
S^{-1} converts C^(h) output back to T basis, M_f multiplies, then S converts back.
All 1D operators are lifted to full grid via Kronecker products.
"""
function _apply_field_multiply(M_f, G, domain, cheb_dim, highest_cheb_order, N_per_var)
    spec = domain.coords[cheb_dim]
    N_z = spec.N
    S_1d = ComplexF64.(get_conversion_operator(domain, cheb_dim, 0, highest_cheb_order))
    S_inv_1d = sparse(Matrix(S_1d) \ Matrix(ComplexF64(1.0) * I, N_z, N_z))

    # Compose in 1D: S * M_f * S_inv, then lift to full grid
    op_1d = S_1d * M_f * S_inv_1d
    op_full = _lift_to_2d(op_1d, cheb_dim, domain)
    return op_full * G
end

# ──────────────────────────────────────────────────────────────────────────────
#  Block placement for multi-variable systems
# ──────────────────────────────────────────────────────────────────────────────

"""Find which variable a term acts on."""
function find_target_variable(expr::ExprNode, variables::Vector{Symbol})
    vars = collect_var_names(expr)
    isempty(vars) && return 1
    length(vars) == 1 || error("Term acts on multiple variables: $vars")
    idx = findfirst(==(first(vars)), variables)
    isnothing(idx) && error("Variable $(first(vars)) not in problem")
    return idx
end

"""Place a single-variable block into the full multi-variable system."""
function place_in_block(mat::AbstractMatrix, eq_idx::Int, var_idx::Int,
                        N_vars::Int, N_per_var::Int)
    N_total = N_per_var * N_vars
    result = spzeros(ComplexF64, N_total, N_total)
    rs = (eq_idx - 1) * N_per_var + 1
    cs = (var_idx - 1) * N_per_var + 1
    result[rs:rs+N_per_var-1, cs:cs+N_per_var-1] = mat
    return result
end

# ──────────────────────────────────────────────────────────────────────────────
#  Build BC rows for the full system
# ──────────────────────────────────────────────────────────────────────────────

function build_bc_rows(prob::EVP, N_per_var::Int, N_total::Int)
    bc_rows_A = Tuple{Int, Vector{Float64}}[]
    bc_rows_B = Tuple{Int, Vector{Float64}}[]
    rhs_values = Float64[]
    bc_count = Dict{Symbol, Int}()

    for bc in prob.bcs
        # Determine which variable "owns" this BC for row placement.
        # Use the first variable (in declaration order) that appears in the expression.
        # This is deterministic, unlike first(Set{Symbol}).
        var_names = collect_var_names(bc.expr)
        primary_var = prob.variables[1]  # default
        for v in prob.variables
            if v in var_names
                primary_var = v
                break
            end
        end
        var_idx = findfirst(==(primary_var), prob.variables)

        count = get(bc_count, primary_var, 0) + 1
        bc_count[primary_var] = count

        # Build the full row vector by walking the BC expression tree.
        row_vec = zeros(N_total)
        _build_bc_row!(row_vec, bc.expr, bc.side, bc.coord, prob, N_per_var)

        # Row to replace: last rows of this variable's equation block
        row_idx = var_idx * N_per_var - count + 1

        push!(bc_rows_A, (row_idx, row_vec))
        push!(bc_rows_B, (row_idx, zeros(N_total)))
        push!(rhs_values, Float64(real(bc.rhs)))
    end

    return bc_rows_A, bc_rows_B, rhs_values
end

"""
    _build_bc_row!(row_vec, expr, side, coord, prob, N_per_var)

Walk the BC expression tree and accumulate boundary evaluation rows into `row_vec`.

Handles:
- VarNode: adds the evaluation row (deriv_order=0) for that variable's block
- DerivNode(expr, coord): counts chained derivatives, adds the deriv-evaluation row
- BinaryOpNode(:+, l, r): recurse on both sides
- BinaryOpNode(:-, l, r): recurse, negate the right side
- BinaryOpNode(:*, ConstNode(c), expr): scale by constant c
- BinaryOpNode(:*, ParamNode(scalar), expr): scale by scalar parameter
"""
function _build_bc_row!(row_vec::Vector{Float64}, expr::ExprNode,
                        side::Symbol, coord::Symbol, prob::EVP, N_per_var::Int;
                        scale::Float64=1.0)
    spec = prob.domain.coords[coord]
    N = spec.N

    if expr isa VarNode
        # Evaluate variable at boundary (deriv_order = 0)
        var_idx = findfirst(==(expr.name), prob.variables)
        isnothing(var_idx) && error("Unknown variable in BC: $(expr.name)")
        bc_row = chebyshev_boundary_row(side, 0, N; a=spec.lower, b=spec.upper)
        col_start = (var_idx - 1) * N_per_var + 1
        row_vec[col_start:col_start+N-1] .+= scale .* bc_row

    elseif expr isa DerivNode
        # Guard: derivative must be in the BC coordinate direction
        if expr.coord != coord
            error("BC expression contains derivative in direction :$(expr.coord), " *
                  "but boundary is on :$(coord). Move the :$(expr.coord) derivative outside the BC.")
        end
        # Count chained derivatives in the BC coordinate
        deriv_order = count_chained_derivs_any(expr, coord)
        inner = unwrap_chained_derivs_any(expr, coord)

        if inner isa VarNode
            var_idx = findfirst(==(inner.name), prob.variables)
            isnothing(var_idx) && error("Unknown variable in BC: $(inner.name)")
            bc_row = chebyshev_boundary_row(side, deriv_order, N; a=spec.lower, b=spec.upper)
            col_start = (var_idx - 1) * N_per_var + 1
            row_vec[col_start:col_start+N-1] .+= scale .* bc_row
        else
            # Derivative of a complex expression: dz(a*psi + b*phi) = a*dz(psi) + b*dz(phi)
            # For linear BCs, derivatives distribute. Wrap each leaf VarNode with
            # the derivative and recurse. This handles dz(3*psi), dz(psi + b), etc.
            distributed = _distribute_deriv(inner, coord, deriv_order)
            _build_bc_row!(row_vec, distributed, side, coord, prob, N_per_var; scale=scale)
        end

    elseif expr isa BinaryOpNode && expr.op == :+
        _build_bc_row!(row_vec, expr.left, side, coord, prob, N_per_var; scale=scale)
        _build_bc_row!(row_vec, expr.right, side, coord, prob, N_per_var; scale=scale)

    elseif expr isa BinaryOpNode && expr.op == :-
        _build_bc_row!(row_vec, expr.left, side, coord, prob, N_per_var; scale=scale)
        _build_bc_row!(row_vec, expr.right, side, coord, prob, N_per_var; scale=-scale)

    elseif expr isa BinaryOpNode && expr.op == :*
        # One side should be a scalar (ConstNode or scalar ParamNode)
        if expr.left isa ConstNode
            c = Float64(real(expr.left.value))
            _build_bc_row!(row_vec, expr.right, side, coord, prob, N_per_var; scale=scale * c)
        elseif expr.right isa ConstNode
            c = Float64(real(expr.right.value))
            _build_bc_row!(row_vec, expr.left, side, coord, prob, N_per_var; scale=scale * c)
        elseif expr.left isa ParamNode && haskey(prob.parameters, expr.left.name) &&
               prob.parameters[expr.left.name] isa Number
            c = Float64(real(prob.parameters[expr.left.name]))
            _build_bc_row!(row_vec, expr.right, side, coord, prob, N_per_var; scale=scale * c)
        elseif expr.right isa ParamNode && haskey(prob.parameters, expr.right.name) &&
               prob.parameters[expr.right.name] isa Number
            c = Float64(real(prob.parameters[expr.right.name]))
            _build_bc_row!(row_vec, expr.left, side, coord, prob, N_per_var; scale=scale * c)
        else
            error("BC multiplication must involve a scalar: got $(expr.left) * $(expr.right)")
        end

    elseif expr isa UnaryOpNode && expr.op == :-
        _build_bc_row!(row_vec, expr.expr, side, coord, prob, N_per_var; scale=-scale)

    else
        error("Unsupported expression in BC: $(typeof(expr))")
    end
end

"""Count chained DerivNodes in any direction (not just matching coord)."""
function count_chained_derivs_any(expr::DerivNode, coord::Symbol)
    if expr.coord == coord
        if expr.expr isa DerivNode && expr.expr.coord == coord
            return 1 + count_chained_derivs_any(expr.expr, coord)
        end
        return 1
    end
    return 0
end

"""Unwrap chained DerivNodes in the BC coordinate direction."""
function unwrap_chained_derivs_any(expr::DerivNode, coord::Symbol)
    if expr.coord == coord
        if expr.expr isa DerivNode && expr.expr.coord == coord
            return unwrap_chained_derivs_any(expr.expr, coord)
        end
        return expr.expr
    end
    return expr
end

"""
    _distribute_deriv(expr, coord, order)

Distribute a derivative through a linear expression tree.
dz(a*psi + b*phi) → a*dz(psi) + b*dz(phi).
Wraps each VarNode leaf with `order` DerivNodes.
"""
function _distribute_deriv(expr::ExprNode, coord::Symbol, order::Int)
    if expr isa VarNode
        # Wrap with derivative chain
        result = expr
        for _ in 1:order
            result = DerivNode(result, coord)
        end
        return result
    elseif expr isa BinaryOpNode && expr.op in (:+, :-)
        left = _distribute_deriv(expr.left, coord, order)
        right = _distribute_deriv(expr.right, coord, order)
        return BinaryOpNode(expr.op, left, right)
    elseif expr isa BinaryOpNode && expr.op == :*
        # For scalar * expr, keep scalar outside: c * dz(expr)
        if expr.left isa ConstNode || expr.left isa ParamNode
            right = _distribute_deriv(expr.right, coord, order)
            return BinaryOpNode(:*, expr.left, right)
        elseif expr.right isa ConstNode || expr.right isa ParamNode
            left = _distribute_deriv(expr.left, coord, order)
            return BinaryOpNode(:*, left, expr.right)
        else
            error("Cannot distribute derivative through non-scalar product in BC: $expr")
        end
    elseif expr isa DerivNode
        # Derivative of a different coordinate inside: d_coord^order(d_other(psi))
        # Distribute the outer derivative through, keeping the inner one intact
        inner_distributed = _distribute_deriv(expr.expr, coord, order)
        return DerivNode(inner_distributed, expr.coord)
    elseif expr isa UnaryOpNode && expr.op == :-
        inner = _distribute_deriv(expr.expr, coord, order)
        return UnaryOpNode(:-, inner)
    else
        error("Cannot distribute derivative through BC expression: $(typeof(expr))")
    end
end

# ──────────────────────────────────────────────────────────────────────────────
#  Main discretize function
# ──────────────────────────────────────────────────────────────────────────────

"""
    discretize(prob::EVP) -> DiscretizationCache

Validate the problem, expand substitutions, lower derivatives, separate by k-power,
and build sparse matrix components. Returns a cache for fast wavenumber assembly.
"""
function discretize(prob::EVP)
    validate_problem(prob)

    N_per_var = total_grid_size(prob.domain)
    N_vars = length(prob.variables)
    N_total = N_per_var * N_vars

    A_components = Dict{Int, SparseMatrixCSC{ComplexF64, Int}}()
    B_components = Dict{Int, SparseMatrixCSC{ComplexF64, Int}}()

    # Determine highest Chebyshev derivative order across ALL equations
    highest_cheb_order = 0
    for eq in prob.equations
        rhs_exp = expand_substitutions(eq.rhs, prob.substitutions)
        lhs_exp = expand_substitutions(eq.lhs, prob.substitutions)
        rhs_low = lower_derivatives(rhs_exp, prob.domain)
        lhs_low = lower_derivatives(lhs_exp, prob.domain)
        for dim in prob.domain.resolved_dims
            if prob.domain.coords[dim] isa ChebyshevBasisSpec
                highest_cheb_order = max(highest_cheb_order, max_deriv_order(rhs_low, dim))
                highest_cheb_order = max(highest_cheb_order, max_deriv_order(lhs_low, dim))
            end
        end
    end

    for (eq_idx, eq) in enumerate(prob.equations)
        # Expand and lower
        rhs_expanded = expand_substitutions(eq.rhs, prob.substitutions)
        lhs_expanded = expand_substitutions(eq.lhs, prob.substitutions)
        rhs_lowered = lower_derivatives(rhs_expanded, prob.domain)
        lhs_lowered = lower_derivatives(lhs_expanded, prob.domain)

        # RHS: separate by k-power, discretize each group
        rhs_terms = separate_by_k_power(rhs_lowered)

        for kt in rhs_terms
            mat = discretize_expr(kt.expr, prob, N_per_var, highest_cheb_order)
            var_idx = find_target_variable(kt.expr, prob.variables)
            block = place_in_block(mat, eq_idx, var_idx, N_vars, N_per_var)

            if haskey(A_components, kt.k_power)
                A_components[kt.k_power] = A_components[kt.k_power] + block
            else
                A_components[kt.k_power] = block
            end
        end

        # LHS: eigenvalue side → B matrix (also needs k-separation)
        # Strip eigenvalue node, then k-separate the remaining expression
        lhs_inner = strip_eigenvalue(lhs_lowered)
        lhs_var_idx = find_target_variable(lhs_inner, prob.variables)
        lhs_terms = separate_by_k_power(lhs_inner)

        for kt in lhs_terms
            mat = discretize_expr(kt.expr, prob, N_per_var, highest_cheb_order)
            b_block = place_in_block(mat, eq_idx, lhs_var_idx, N_vars, N_per_var)

            if haskey(B_components, kt.k_power)
                B_components[kt.k_power] = B_components[kt.k_power] + b_block
            else
                B_components[kt.k_power] = b_block
            end
        end
    end

    # Apply BCs
    bc_rows_A, bc_rows_B, rhs_values = build_bc_rows(prob, N_per_var, N_total)
    bc_row_indices = [row_idx for (row_idx, _) in bc_rows_A]

    # Check for inhomogeneous BCs — not yet supported for eigenvalue problems
    for (i, rhs) in enumerate(rhs_values)
        if abs(rhs) > 0.0
            error("Inhomogeneous boundary condition (rhs = $rhs) is not supported " *
                  "for eigenvalue problems. Use homogeneous BCs (rhs == 0).")
        end
    end

    # Ensure k^0 components exist
    if !haskey(A_components, 0)
        A_components[0] = spzeros(ComplexF64, N_total, N_total)
    end
    if !haskey(B_components, 0)
        B_components[0] = spzeros(ComplexF64, N_total, N_total)
    end

    # Replace BC rows in the k^0 component (sparse-native, no dense conversion)
    for (row_idx, row_vec) in bc_rows_A
        A_components[0][row_idx, :] .= zero(ComplexF64)
        for (j, v) in enumerate(row_vec)
            if v != 0.0
                A_components[0][row_idx, j] = ComplexF64(v)
            end
        end
    end
    for (row_idx, row_vec) in bc_rows_B
        B_components[0][row_idx, :] .= zero(ComplexF64)
        for (j, v) in enumerate(row_vec)
            if v != 0.0
                B_components[0][row_idx, j] = ComplexF64(v)
            end
        end
    end

    # Zero out BC rows in ALL non-zero k-power components (both A and B)
    # so they don't corrupt boundary conditions during assembly
    for (p, M) in A_components
        p == 0 && continue
        for row_idx in bc_row_indices
            M[row_idx, :] .= zero(ComplexF64)
        end
    end
    for (p, M) in B_components
        p == 0 && continue
        for row_idx in bc_row_indices
            M[row_idx, :] .= zero(ComplexF64)
        end
    end

    return DiscretizationCache(A_components, B_components, N_total, prob.domain)
end

"""
    strip_eigenvalue(lhs) -> ExprNode

Strip the eigenvalue node from LHS, returning the remaining expression.
LHS must be of the form `eigenvalue * expr`.
"""
function strip_eigenvalue(lhs::ExprNode)
    if lhs isa BinaryOpNode && lhs.op == :*
        if lhs.left isa EigenvalueNode
            return lhs.right
        elseif lhs.right isa EigenvalueNode
            return lhs.left
        end
        # Recurse into product chains: (sigma * a) * b → a * b
        left_stripped = _try_strip(lhs.left)
        if !isnothing(left_stripped)
            return BinaryOpNode(:*, left_stripped, lhs.right)
        end
        right_stripped = _try_strip(lhs.right)
        if !isnothing(right_stripped)
            return BinaryOpNode(:*, lhs.left, right_stripped)
        end
    end
    error("LHS must be eigenvalue * expr, got: $lhs")
end

function _try_strip(expr::ExprNode)
    if expr isa EigenvalueNode
        return ConstNode(1.0)
    elseif expr isa BinaryOpNode && expr.op == :*
        if expr.left isa EigenvalueNode
            return expr.right
        elseif expr.right isa EigenvalueNode
            return expr.left
        end
        left_stripped = _try_strip(expr.left)
        if !isnothing(left_stripped)
            return BinaryOpNode(:*, left_stripped, expr.right)
        end
        right_stripped = _try_strip(expr.right)
        if !isnothing(right_stripped)
            return BinaryOpNode(:*, expr.left, right_stripped)
        end
    end
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
#  Assemble
# ──────────────────────────────────────────────────────────────────────────────

"""
    assemble(cache, k) -> (A, B)

Assemble the full A and B matrices for a given wavenumber k (allocating).
A(k) = Σ_p k^p * A_p,  B(k) = Σ_p k^p * B_p

Note: the `im` factors from wavenumber lowering (dx → im*k) are already
incorporated into the discretized components, so assembly uses k^p not (im*k)^p.
"""
function assemble(cache::DiscretizationCache, k::Float64)
    N = cache.N_total
    A = spzeros(ComplexF64, N, N)
    for (p, Ap) in cache.A_components
        if p == 0
            A = A + Ap
        else
            A = A + k^p * Ap
        end
    end

    B = spzeros(ComplexF64, N, N)
    for (p, Bp) in cache.B_components
        if p == 0
            B = B + Bp
        else
            B = B + k^p * Bp
        end
    end

    return A, B
end

# ──────────────────────────────────────────────────────────────────────────────
#  In-place assembly workspace (for allocation-free wavenumber loops)
# ──────────────────────────────────────────────────────────────────────────────

"""
Pre-allocated workspace for in-place assembly. One per thread.

Create with `allocate_workspace(cache)`, reuse across wavenumbers.
"""
struct AssemblyWorkspace
    A::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    temp::Matrix{ComplexF64}  # scratch buffer for solver (A - sigma*B)
end

"""
    allocate_workspace(cache) -> AssemblyWorkspace

Allocate a dense (A, B, temp) workspace matching the cache dimensions.
Dense because eigenvalue solvers need dense matrices for factorization anyway,
and in-place overwrites require fixed memory layout.
"""
function allocate_workspace(cache::DiscretizationCache)
    N = cache.N_total
    return AssemblyWorkspace(
        zeros(ComplexF64, N, N),
        zeros(ComplexF64, N, N),
        zeros(ComplexF64, N, N)
    )
end

"""
    assemble!(ws, cache, k)

Assemble A and B into the pre-allocated workspace `ws` for wavenumber `k`.
Overwrites `ws.A` and `ws.B` in-place — zero allocation.
"""
function assemble!(ws::AssemblyWorkspace, cache::DiscretizationCache, k::Float64)
    N = cache.N_total

    # Zero out
    fill!(ws.A, zero(ComplexF64))
    fill!(ws.B, zero(ComplexF64))

    # Accumulate A components
    for (p, Ap) in cache.A_components
        coeff = p == 0 ? ComplexF64(1.0) : ComplexF64(k^p)
        rows = rowvals(Ap)
        vals = nonzeros(Ap)
        for col in 1:N
            for idx in nzrange(Ap, col)
                ws.A[rows[idx], col] += coeff * vals[idx]
            end
        end
    end

    # Accumulate B components
    for (p, Bp) in cache.B_components
        coeff = p == 0 ? ComplexF64(1.0) : ComplexF64(k^p)
        rows = rowvals(Bp)
        vals = nonzeros(Bp)
        for col in 1:N
            for idx in nzrange(Bp, col)
                ws.B[rows[idx], col] += coeff * vals[idx]
            end
        end
    end

    return ws
end
