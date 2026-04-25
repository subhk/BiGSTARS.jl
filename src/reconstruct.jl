#══════════════════════════════════════════════════════════════════════════════#
#  Post-processing: reconstruct derived fields and evaluate expressions       #
#══════════════════════════════════════════════════════════════════════════════#

# ──────────────────────────────────────────────────────────────────────────────
#  @compute — evaluate any expression on eigenvector fields
# ──────────────────────────────────────────────────────────────────────────────

"""
    @compute v = -dy(dz(w)) + dx(zeta)

Evaluate a DSL expression on the solved eigenvector fields and assign the result.
Requires `cache`, `eigvec`, and `k` to be in scope (set via `@compute_setup`).

```julia
cache = discretize(prob)
results = solve(cache, [0.1]; sigma_0=0.02)

@compute_setup cache results[1].eigenvectors[:, 1] 0.1

@compute v = -dy(dz(w)) + dx(zeta)
@compute u = -dx(dz(w)) - dy(zeta)
@compute vort = dx(dx(psi)) + dy(dy(psi))
@compute flux = U * dy(b) - E * D2(zeta)
```
"""
macro compute(expr)
    (expr isa Expr && expr.head == :(=) && length(expr.args) == 2) ||
        error("@compute expects: @compute result = expression")

    result_sym = expr.args[1]
    rhs = expr.args[2]
    # Strip begin...end block from assignment RHS
    if rhs isa Expr && rhs.head == :block
        stmts = filter(a -> !(a isa LineNumberNode), rhs.args)
        length(stmts) == 1 && (rhs = stmts[1])
    end

    prob_ref = :(_get_active_prob())
    expr_ast = parse_expr_ast(rhs, prob_ref)

    return quote
        $(esc(result_sym)) = _evaluate_expr($expr_ast, $prob_ref,
            _compute_cache[], _compute_eigvec[], _compute_k[])
    end
end

"""
    @compute_setup cache eigvec k

Set up the context for `@compute`. Call once after solving.

```julia
@compute_setup cache results[1].eigenvectors[:, 1] 0.1
@compute v = -dy(dz(w)) + dx(zeta)
```
"""
macro compute_setup(cache, eigvec, k)
    return quote
        _compute_cache[] = $(esc(cache))
        _compute_eigvec[] = $(esc(eigvec))
        _compute_k[] = Float64($(esc(k)))
    end
end

# Module-level storage for @compute context
const _compute_cache = Ref{Any}(nothing)
const _compute_eigvec = Ref{Any}(nothing)
const _compute_k = Ref{Float64}(0.0)

"""
Evaluate an expression tree on eigenvector data.
Walks the tree, applying operators to the coefficient vectors extracted from eigvec.
"""
function _evaluate_expr(expr::ExprNode, prob::EVP, cache::DiscretizationCache,
                        eigvec::AbstractVector, k::Float64)
    N_per_var = cache.N_per_var
    domain = cache.domain

    if expr isa VarNode
        # Extract variable's coefficients from eigenvector
        if expr.name in prob.variables
            var_idx = findfirst(==(expr.name), prob.variables)
            start = (var_idx - 1) * N_per_var + 1
            return ComplexF64.(eigvec[start:start+N_per_var-1])
        elseif haskey(prob.derived_vars, expr.name)
            # Derived variable: reconstruct via inverse operator
            return reconstruct(cache, prob, eigvec, k, expr.name)
        else
            error("Unknown variable in @compute: $(expr.name)")
        end

    elseif expr isa ParamNode
        val = prob.parameters[expr.name]
        if val isa Number
            error("Scalar parameter $(expr.name) cannot produce a field alone. " *
                  "Use it as a multiplier: $(expr.name) * variable")
        else
            # Field parameter as a coefficient vector
            # (Not meaningful alone — but could appear in products)
            error("Field parameter $(expr.name) cannot be evaluated alone. " *
                  "Use it as a multiplier: $(expr.name) * variable")
        end

    elseif expr isa ConstNode
        error("Constant $(expr.value) cannot produce a field alone.")

    elseif expr isa DerivNode
        coord = expr.coord
        spec = domain.coords[coord]
        inner_vec = _evaluate_expr(expr.expr, prob, cache, eigvec, k)

        if spec isa FourierTransformed
            # dx → im*k
            return im * k * inner_vec
        elseif spec isa FourierBasisSpec
            # Fourier derivative: diagonal operator
            D_1d = fourier_diff_operator(spec.N, spec.L, 1)
            D_full = _lift_to_2d(D_1d, coord, domain)
            return D_full * inner_vec
        elseif spec isa ChebyshevBasisSpec
            # Chebyshev derivative in T basis: S^{-1} * D_0
            N_z = spec.N
            scale = 2.0 / (spec.upper - spec.lower)
            D_0 = ComplexF64(scale) .* differentiation_operator(0, N_z)
            S_0 = ComplexF64.(conversion_operator(0, N_z))
            S_inv = sparse(Matrix(S_0) \ Matrix(ComplexF64(1.0) * I, N_z, N_z))
            D_T = S_inv * D_0  # T→T derivative
            D_full = _lift_to_2d(D_T, coord, domain)
            return D_full * inner_vec
        end

    elseif expr isa BinaryOpNode
        if expr.op == :+
            return _evaluate_expr(expr.left, prob, cache, eigvec, k) +
                   _evaluate_expr(expr.right, prob, cache, eigvec, k)
        elseif expr.op == :-
            return _evaluate_expr(expr.left, prob, cache, eigvec, k) -
                   _evaluate_expr(expr.right, prob, cache, eigvec, k)
        elseif expr.op == :*
            # Check if either side is a scalar
            lv = _try_scalar(expr.left, prob)
            rv = _try_scalar(expr.right, prob)
            if !isnothing(lv)
                return lv .* _evaluate_expr(expr.right, prob, cache, eigvec, k)
            elseif !isnothing(rv)
                return rv .* _evaluate_expr(expr.left, prob, cache, eigvec, k)
            elseif _is_field_param(expr.left, prob)
                right_vec = _evaluate_expr(expr.right, prob, cache, eigvec, k)
                M = _field_param_operator(expr.left::ParamNode, prob, domain, N_per_var)
                return M * right_vec
            elseif _is_field_param(expr.right, prob)
                left_vec = _evaluate_expr(expr.left, prob, cache, eigvec, k)
                M = _field_param_operator(expr.right::ParamNode, prob, domain, N_per_var)
                return M * left_vec
            else
                # Field * field: multiplication in coefficient space
                left_vec = _evaluate_expr(expr.left, prob, cache, eigvec, k)
                right_vec = _evaluate_expr(expr.right, prob, cache, eigvec, k)
                # Build multiplication operator from left, apply to right
                cheb_dim = _find_chebyshev_dim(domain)
                if !isnothing(cheb_dim)
                    fourier_dim = _find_fourier_dim(domain)
                    if !isnothing(fourier_dim)
                        M = _build_2d_coeff_multiply(left_vec, domain, cheb_dim, fourier_dim)
                    else
                        M = _complex_multiplication_operator(ComplexF64.(left_vec), length(left_vec))
                    end
                    return M * right_vec
                else
                    # Fourier-only: convolution
                    M = fourier_multiply_operator(left_vec, length(left_vec))
                    return M * right_vec
                end
            end
        end

    elseif expr isa UnaryOpNode && expr.op == :-
        return -_evaluate_expr(expr.expr, prob, cache, eigvec, k)

    elseif expr isa SubstitutionNode
        # Expand substitution and evaluate
        expanded = expand_substitutions(expr, prob.substitutions)
        return _evaluate_expr(expanded, prob, cache, eigvec, k)
    end

    error("Cannot evaluate expression: $(typeof(expr))")
end

function _is_field_param(expr::ExprNode, prob::EVP)
    return expr isa ParamNode &&
           haskey(prob.parameters, expr.name) &&
           !(prob.parameters[expr.name] isa Number)
end

function _field_param_operator(param::ParamNode, prob::EVP, domain::Domain, N_per_var::Int)
    val = prob.parameters[param.name]
    if val isa AbstractMatrix && size(val) == (N_per_var, N_per_var)
        return ComplexF64.(val)
    end

    f_vec = vec(val)
    cheb_dim = _find_chebyshev_dim(domain)
    fourier_dim = _find_fourier_dim(domain)
    if !isnothing(cheb_dim) && !isnothing(fourier_dim)
        return _build_2d_field_multiply(f_vec, domain, cheb_dim, fourier_dim, 0)
    elseif !isnothing(cheb_dim)
        c = chebyshev_coefficients(Float64.(f_vec))
        return ComplexF64.(multiplication_operator(c, length(f_vec)))
    else
        return fourier_multiply_operator(fft(f_vec) / length(f_vec), length(f_vec))
    end
end

"""
    reconstruct(cache, prob, eigvec, k, field_name::Symbol) -> Vector{ComplexF64}

Reconstruct a derived variable's coefficient vector from an eigenvector.

Given the solved eigenvector `eigvec` (containing coefficients for all declared
variables) and wavenumber `k`, computes the derived field using its stored
operator inverse: `v = H(k) * rhs(eigvec)`.

```julia
cache = discretize(prob)
results = solve(cache, [0.1]; sigma_0=0.02)
eigvec = results[1].eigenvectors[:, 1]

# Reconstruct derived variables:
v_coeffs = reconstruct(cache, prob, eigvec, 0.1, :v)
u_coeffs = reconstruct(cache, prob, eigvec, 0.1, :u)
```
"""
function reconstruct(cache::DiscretizationCache, prob::EVP,
                     eigvec::AbstractVector, k::Float64, field_name::Symbol)
    haskey(prob.derived_vars, field_name) ||
        error("No derived variable :$field_name. Available: $(keys(prob.derived_vars))")

    dvar = prob.derived_vars[field_name]
    dc = cache.derived_caches[field_name]
    N_per_var = cache.N_per_var
    N_vars = cache.N_vars
    k_vals = Dict{Symbol, Float64}()
    if isempty(cache.domain.transformed_dims)
        k_vals[:_total_k] = k
    else
        for dim in cache.domain.transformed_dims
            k_vals[Symbol(:k_, dim)] = k
        end
    end

    # Build H(k) for this wavenumber
    op_k = dc.op_k0
    for (kp, mat) in dc.op_k_components
        coeff = _k_coeff(kp, k_vals)
        coeff == 0.0 && continue
        op_k = op_k + coeff * mat
    end
    H_k = _sparse_block_inverse(op_k, cache.domain; bcs=dc.bcs)

    # Expand and lower the RHS expression
    rhs_expanded = expand_substitutions(dvar.rhs, prob.substitutions)
    rhs_lowered = lower_derivatives(rhs_expanded, prob.domain)
    rhs_distributed = distribute_products(rhs_lowered)
    rhs_additive = separate_additive_terms(rhs_distributed)

    # Accumulate: result = H(k) * Σ (k^p * rhs_op * var_coeffs)
    rhs_vec = zeros(ComplexF64, N_per_var)

    for rhs_term in rhs_additive
        rhs_kp, rhs_reduced = extract_k_power(rhs_term)
        rhs_mat = _discretize_operator(rhs_reduced, nothing, prob, N_per_var)
        var_idx = find_target_variable(rhs_reduced, prob.variables)

        # Extract this variable's coefficients from the eigenvector
        var_start = (var_idx - 1) * N_per_var + 1
        var_coeffs = eigvec[var_start:var_start+N_per_var-1]

        rhs_vec .+= k^rhs_kp .* (rhs_mat * var_coeffs)
    end

    return H_k * rhs_vec
end

"""
    evaluate_field(cache, prob, eigvec, k, expr_fn) -> Vector{ComplexF64}

Evaluate a general expression on the eigenvector fields.

`expr_fn` is a function that takes a Dict of variable coefficient vectors
and returns the result. Operators from the cache can be applied.

```julia
# Manual field computation:
v_coeffs = evaluate_field(cache, prob, eigvec, 0.1) do fields
    # fields[:w], fields[:zeta], fields[:b] are coefficient vectors
    # Apply operators manually:
    D_y = get_diff_operator(prob.domain, :y, 1)
    D_z_T = ...  # build T-basis z-derivative
    return -D_y * D_z_T * fields[:w] + im * 0.1 * fields[:zeta]
end
```

For derived variables, use `reconstruct(cache, prob, eigvec, k, :v)` instead.
"""
function evaluate_field(fn::Function, cache::DiscretizationCache, prob::EVP,
                        eigvec::AbstractVector, k::Float64)
    N_per_var = cache.N_per_var

    # Build dict of per-variable coefficient vectors
    fields = Dict{Symbol, Vector{ComplexF64}}()
    for (i, var) in enumerate(prob.variables)
        var_start = (i - 1) * N_per_var + 1
        fields[var] = eigvec[var_start:var_start+N_per_var-1]
    end

    # Also add derived variables
    for (dname, _) in prob.derived_vars
        fields[dname] = reconstruct(cache, prob, eigvec, k, dname)
    end

    return fn(fields)
end

"""
    reconstruct_all(cache, prob, eigvec, k) -> Dict{Symbol, Vector{ComplexF64}}

Reconstruct ALL fields (declared variables + derived variables) from an eigenvector.

```julia
fields = reconstruct_all(cache, prob, eigvec, 0.1)
# fields[:w], fields[:zeta], fields[:b] — from eigenvector
# fields[:v], fields[:u] — reconstructed from @derive definitions
```
"""
function reconstruct_all(cache::DiscretizationCache, prob::EVP,
                         eigvec::AbstractVector, k::Float64)
    N_per_var = cache.N_per_var
    fields = Dict{Symbol, Vector{ComplexF64}}()

    # Declared variables: extract from eigenvector
    for (i, var) in enumerate(prob.variables)
        var_start = (i - 1) * N_per_var + 1
        fields[var] = eigvec[var_start:var_start+N_per_var-1]
    end

    # Derived variables: reconstruct via inverse operator
    for (dname, _) in prob.derived_vars
        fields[dname] = reconstruct(cache, prob, eigvec, k, dname)
    end

    return fields
end
