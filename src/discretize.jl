#══════════════════════════════════════════════════════════════════════════════#
#  Discretize: expression tree → sparse matrices, k-separated caching         #
#══════════════════════════════════════════════════════════════════════════════#

# ──────────────────────────────────────────────────────────────────────────────
#  Cache type
# ──────────────────────────────────────────────────────────────────────────────

"""
Mutable memo carried by a `DiscretizationCache` so `reconstruct` does not
rebuild the (eigenvector-independent) inverse operator and RHS matrices for
every eigenvector. `rhs` is k-independent; `Hk` is keyed by `(field, k)` and
size-capped because each H(k) is a near-dense N_per_var² matrix.
"""
struct ReconstructMemo
    rhs::Dict{Symbol, Vector{Tuple{Int, KPowerKey, SparseMatrixCSC{ComplexF64, Int}}}}
    Hk::Dict{Tuple{Symbol, Float64}, SparseMatrixCSC{ComplexF64, Int}}
end
ReconstructMemo() = ReconstructMemo(
    Dict{Symbol, Vector{Tuple{Int, KPowerKey, SparseMatrixCSC{ComplexF64, Int}}}}(),
    Dict{Tuple{Symbol, Float64}, SparseMatrixCSC{ComplexF64, Int}}())

"""Pre-computed info for a derived variable, used to rebuild H(k) during assembly."""
struct DerivedVarCache
    # The operator matrix WITHOUT the k² term (k-independent part only)
    op_k0::SparseMatrixCSC{ComplexF64, Int}
    # Coefficient of k² in the operator (from dx(dx(...)) → -k² * I)
    op_k2_coeff::ComplexF64
    # Per-wavenumber operator components for multi-FourierTransformed domains.
    op_k_components::Dict{KPowerKey, SparseMatrixCSC{ComplexF64, Int}}
    # BCs for the inversion (boundary bordering)
    bcs::Vector{BoundaryCondition}
    # (eq_idx, var_idx, rhs_k_powers, coeff_matrix, rhs_matrix)
    terms::Vector{Tuple{Int, Int, KPowerKey, SparseMatrixCSC{ComplexF64, Int}, SparseMatrixCSC{ComplexF64, Int}}}
end

"""Cache of pre-discretized sparse matrix components, separated by k-power."""
struct DiscretizationCache
    A_components::Dict{KPowerKey, SparseMatrixCSC{ComplexF64, Int}}
    B_components::Dict{KPowerKey, SparseMatrixCSC{ComplexF64, Int}}
    derived_caches::Dict{Symbol, DerivedVarCache}
    N_total::Int
    N_per_var::Int
    N_vars::Int
    domain::Domain
    derived_var_order::Vector{Symbol}
    row_range::Union{Nothing, Tuple{Int, Int}}
    # Memo for `reconstruct`: per-field RHS term matrices (k-independent) and
    # H(k) per (field, k). H(k) is near-dense, so the H memo is size-capped.
    reconstruct_memo::ReconstructMemo
end

const FieldMultiplyCacheKey = Tuple{Symbol, Symbol, Union{Nothing, Symbol}}
const FieldOpCacheKey = Tuple{Symbol, Int}  # (field_name, cheb_order)

"""Scratch cache used while building one `DiscretizationCache`."""
mutable struct DiscretizationContext
    field_t_cache::Dict{FieldMultiplyCacheKey, SparseMatrixCSC{ComplexF64, Int}}
    field_t_hits::Int
    field_t_misses::Int
    s_full_cache::Dict{Int, SparseMatrixCSC{ComplexF64, Int}}      # S_full per cheb_order
    s_inv_full_cache::Dict{Int, SparseMatrixCSC{ComplexF64, Int}}  # S_inv_full (N_per_var) per cheb_order
    sinv_1d_cache::Dict{Int, SparseMatrixCSC{ComplexF64, Int}}     # S_inv_1d (N_z) per cheb_order
    s_mf_sinv_cache::Dict{FieldOpCacheKey, SparseMatrixCSC{ComplexF64, Int}}  # S*M_f*S_inv per (field, order)
    coeff_tol::Float64                                                        # Fourier-block drop tolerance (0 = exact)
end

function DiscretizationContext(coeff_tol::Float64=0.0)
    return DiscretizationContext(
        Dict{FieldMultiplyCacheKey, SparseMatrixCSC{ComplexF64, Int}}(),
        0,
        0,
        Dict{Int, SparseMatrixCSC{ComplexF64, Int}}(),
        Dict{Int, SparseMatrixCSC{ComplexF64, Int}}(),
        Dict{Int, SparseMatrixCSC{ComplexF64, Int}}(),
        Dict{FieldOpCacheKey, SparseMatrixCSC{ComplexF64, Int}}(),
        coeff_tol,
    )
end

# coeff_tol carried on the (per-call) ctx; nothing → exact.
_ctx_coeff_tol(ctx::Union{Nothing, DiscretizationContext}) = ctx === nothing ? 0.0 : ctx.coeff_tol

function DiscretizationCache(A_components::Dict{KPowerKey, SparseMatrixCSC{ComplexF64, Int}},
                             B_components::Dict{KPowerKey, SparseMatrixCSC{ComplexF64, Int}},
                             derived_caches::Dict{Symbol, DerivedVarCache},
                             N_total::Int, N_per_var::Int, N_vars::Int, domain::Domain,
                             derived_var_order::Vector{Symbol}=Symbol[])
    return DiscretizationCache(A_components, B_components, derived_caches,
                               N_total, N_per_var, N_vars, domain, derived_var_order, nothing,
                               ReconstructMemo())
end

function _k_coeff(key::KPowerKey, k_vals::Dict{Symbol, Float64})
    coeff = 1.0
    for (name, power) in key
        coeff *= get(k_vals, name, 0.0)^power
    end
    return coeff
end

function _normalize_k_values(domain::Domain, kwargs)
    transformed = domain.transformed_dims
    valid_dim_names = Set(transformed)
    valid_k_names = Set(Symbol(:k_, dim) for dim in transformed)
    k_vals = Dict{Symbol, Float64}()

    for (name, val) in kwargs
        name_sym = Symbol(name)
        val_float = Float64(val)
        if name_sym in valid_dim_names
            k_name = Symbol(:k_, name_sym)
            if haskey(k_vals, k_name) && k_vals[k_name] != val_float
                error("Conflicting values provided for wavenumber :$k_name")
            end
            k_vals[k_name] = val_float
        elseif name_sym in valid_k_names
            if haskey(k_vals, name_sym) && k_vals[name_sym] != val_float
                error("Conflicting values provided for wavenumber :$name_sym")
            end
            k_vals[name_sym] = val_float
        else
            expected = String[]
            for dim in transformed
                push!(expected, string(dim))
                push!(expected, string(:k_, dim))
            end
            expected_msg = isempty(expected) ? "none; this domain has no FourierTransformed coordinates" :
                           join(expected, ", ")
            error("Unknown wavenumber keyword :$name_sym. Expected one of: $expected_msg")
        end
    end

    return k_vals
end

function _add_component!(components::Dict{KPowerKey, SparseMatrixCSC{ComplexF64, Int}},
                         key::KPowerKey,
                         block::SparseMatrixCSC{ComplexF64, Int})
    if haskey(components, key)
        components[key] = components[key] + block
    else
        components[key] = block
    end
    return components
end

# COO-triplet accumulator for building a k-power component without repeated
# sparse `+=` (which reallocates the whole accumulator each term). Blocks are
# scattered into per-key (I, J, V) buffers and finalized with one `sparse()`,
# which sums duplicate (i, j) entries — identical result to sequential `+=`.
const _COOBuf = Tuple{Vector{Int}, Vector{Int}, Vector{ComplexF64}}

function _scatter_block!(buffers::Dict{KPowerKey, _COOBuf}, key::KPowerKey,
                         mat::SparseMatrixCSC{ComplexF64, Int},
                         eq_idx::Int, var_idx::Int, N_per_var::Int)
    row_off = (eq_idx - 1) * N_per_var
    col_off = (var_idx - 1) * N_per_var
    I, J, V = get!(() -> (Int[], Int[], ComplexF64[]), buffers, key)
    rows = rowvals(mat)
    vals = nonzeros(mat)
    for col in 1:size(mat, 2)
        for idx in nzrange(mat, col)
            push!(I, rows[idx] + row_off)
            push!(J, col + col_off)
            push!(V, vals[idx])
        end
    end
    return buffers
end

# Overload for dense matrices (e.g. matrix-valued parameter operators): convert first.
function _scatter_block!(buffers::Dict{KPowerKey, _COOBuf}, key::KPowerKey,
                         mat::AbstractMatrix,
                         eq_idx::Int, var_idx::Int, N_per_var::Int)
    return _scatter_block!(buffers, key, sparse(ComplexF64.(mat)), eq_idx, var_idx, N_per_var)
end

# Row-restricted scatter: like _scatter_block!, but only entries whose GLOBAL row
# (1-based: rows[idx] + row_off) lies in the owned slice [rstart, rend) (0-based
# half-open) are kept, remapped to the LOCAL row within the slice. Columns stay
# global/full. Used to build only a rank's owned rows without materializing N×N.
function _scatter_block_rows!(buffers::Dict{KPowerKey, _COOBuf}, key::KPowerKey,
                              mat::SparseMatrixCSC{ComplexF64, Int},
                              eq_idx::Int, var_idx::Int, N_per_var::Int,
                              rstart::Int, rend::Int)
    row_off = (eq_idx - 1) * N_per_var
    col_off = (var_idx - 1) * N_per_var
    # Skip this equation block entirely if its rows don't overlap the owned slice
    # [rstart, rend) — half-open overlap test on the block's global row interval.
    max(row_off, rstart) < min(row_off + N_per_var, rend) || return buffers
    I, J, V = get!(() -> (Int[], Int[], ComplexF64[]), buffers, key)
    rows = rowvals(mat)
    vals = nonzeros(mat)
    for col in 1:size(mat, 2)
        for idx in nzrange(mat, col)
            g = rows[idx] + row_off            # 1-based global row
            (rstart < g <= rend) || continue   # owned ⟺ g ∈ [rstart+1, rend]
            push!(I, g - rstart)               # 1-based local row in the slice
            push!(J, col + col_off)            # global 1-based column
            push!(V, vals[idx])
        end
    end
    return buffers
end

# Dense overload, mirroring the dense _scatter_block!.
function _scatter_block_rows!(buffers::Dict{KPowerKey, _COOBuf}, key::KPowerKey,
                              mat::AbstractMatrix,
                              eq_idx::Int, var_idx::Int, N_per_var::Int,
                              rstart::Int, rend::Int)
    return _scatter_block_rows!(buffers, key, sparse(ComplexF64.(mat)),
                                eq_idx, var_idx, N_per_var, rstart, rend)
end

# Zero (and drop) all stored entries in the given rows — one O(nnz) pass.
function _zero_sparse_rows!(M::SparseMatrixCSC{ComplexF64, Int}, rows::Set{Int})
    rv = rowvals(M); vals = nonzeros(M)
    for idx in eachindex(vals)
        (rv[idx] in rows) && (vals[idx] = zero(ComplexF64))
    end
    return dropzeros!(M)
end

function _finalize_components(buffers::Dict{KPowerKey, _COOBuf}, nrows::Int, N_total::Int)
    out = Dict{KPowerKey, SparseMatrixCSC{ComplexF64, Int}}()
    for (key, (I, J, V)) in buffers
        out[key] = sparse(I, J, V, nrows, N_total)
    end
    return out
end

# Square (full-build) convenience: nrows == N_total.
_finalize_components(buffers::Dict{KPowerKey, _COOBuf}, N_total::Int) =
    _finalize_components(buffers, N_total, N_total)

function Base.show(io::IO, c::DiscretizationCache)
    println(io, "DiscretizationCache")
    println(io, "  System size: $(c.N_total) x $(c.N_total) ($(c.N_vars) variables, $(c.N_per_var) per var)")
    a_keys = collect(keys(c.A_components))
    b_keys = collect(keys(c.B_components))
    println(io, "  A components (", length(a_keys), " k-power keys)")
    println(io, "  B components (", length(b_keys), " k-power keys)")
    total_nnz = sum(nnz(v) for v in values(c.A_components)) +
                sum(nnz(v) for v in values(c.B_components))
    density = total_nnz / (2 * c.N_total^2) * 100
    println(io, "  Total nnz: ", total_nnz, " (", @sprintf("%.2f", density), "% dense)")
    if !isempty(c.derived_caches)
        for (name, dc) in c.derived_caches
            println(io, "  Derived :$name — $(length(dc.terms)) terms, " *
                        "k² coeff=$(dc.op_k2_coeff) (H recomputed per-k)")
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
#  Validation
# ──────────────────────────────────────────────────────────────────────────────

function validate_problem(prob::EVP)
    n_vars = length(prob.variables)
    n_eqs = length(prob.equations)
    n_vars == n_eqs ||
        error("Need $n_vars equations for $n_vars variables, got $n_eqs")

    # Validate BC count: for Chebyshev directions, each variable's equation
    # needs enough BCs to match the highest z-derivative order
    cheb_dim = nothing
    for dim in prob.domain.resolved_dims
        if prob.domain.coords[dim] isa ChebyshevBasisSpec
            cheb_dim = dim
            break
        end
    end

    if !isnothing(cheb_dim)
        # Count BCs per variable
        bc_counts = Dict{Symbol, Int}()
        for bc in prob.bcs
            var_names = collect_var_names(bc.expr)
            primary = prob.variables[1]
            for v in prob.variables
                if v in var_names
                    primary = v
                    break
                end
            end
            bc_counts[primary] = get(bc_counts, primary, 0) + 1
        end

        # Check each equation's order vs BC count
        for (eq_idx, eq) in enumerate(prob.equations)
            var = prob.variables[eq_idx]
            rhs_exp = expand_substitutions(eq.rhs, prob.substitutions)
            rhs_low = lower_derivatives(rhs_exp, prob.domain)
            order = max_deriv_order(rhs_low, cheb_dim)

            lhs_exp = expand_substitutions(eq.lhs, prob.substitutions)
            lhs_low = lower_derivatives(lhs_exp, prob.domain)
            order = max(order, max_deriv_order(lhs_low, cheb_dim))

            n_bcs = get(bc_counts, var, 0)
            if n_bcs < order
                @warn "Variable :$var has equation order $order in :$cheb_dim but only $n_bcs BCs " *
                      "(need $order). Missing BCs may cause spurious eigenvalues."
            end
        end
    end
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

function discretize_expr(expr::ExprNode, prob::EVP, N_per_var::Int, highest_cheb_order::Int,
                         ctx::Union{Nothing, DiscretizationContext}=nothing)
    domain = prob.domain

    if expr isa VarNode
        # Identity converted to C^(highest_cheb_order), lifted to full grid
        return _full_conversion(domain, highest_cheb_order, ctx)

    elseif expr isa ParamNode
        val = prob.parameters[expr.name]
        if val isa Number
            return ComplexF64(val) .* _full_conversion(domain, highest_cheb_order, ctx)
        elseif val isa AbstractMatrix && size(val) == (N_per_var, N_per_var)
            # Square matrix parameter: use directly as a pre-computed operator.
            return ComplexF64.(val)
        else
            _field_param_is_zero(expr, prob) && return spzeros(ComplexF64, N_per_var, N_per_var)
            return _converted_field_multiply(expr, prob, N_per_var, highest_cheb_order, ctx)
        end

    elseif expr isa ConstNode
        return ComplexF64(expr.value) .* _full_conversion(domain, highest_cheb_order, ctx)

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
            inner_mat_T = discretize_expr_in_T(inner_expr, prob, N_per_var, ctx)

            return D_full * inner_mat_T

        elseif spec isa FourierBasisSpec
            inner_mat = discretize_expr(expr.expr, prob, N_per_var, highest_cheb_order, ctx)
            D_1d = fourier_diff_operator(spec.N, spec.L, 1)
            D_full = _lift_to_2d(D_1d, coord, domain)
            return D_full * inner_mat
        else
            error("Cannot differentiate in FourierTransformed direction (should have been lowered)")
        end

    elseif expr isa BinaryOpNode
        if expr.op == :+
            return discretize_expr(expr.left, prob, N_per_var, highest_cheb_order, ctx) +
                   discretize_expr(expr.right, prob, N_per_var, highest_cheb_order, ctx)
        elseif expr.op == :-
            return discretize_expr(expr.left, prob, N_per_var, highest_cheb_order, ctx) -
                   discretize_expr(expr.right, prob, N_per_var, highest_cheb_order, ctx)
        elseif expr.op == :*
            lv = _try_scalar(expr.left, prob)
            rv = _try_scalar(expr.right, prob)
            if !isnothing(lv) && !isnothing(rv)
                return (lv * rv) .* _full_conversion(domain, highest_cheb_order, ctx)
            elseif !isnothing(lv)
                return lv .* discretize_expr(expr.right, prob, N_per_var, highest_cheb_order, ctx)
            elseif !isnothing(rv)
                return rv .* discretize_expr(expr.left, prob, N_per_var, highest_cheb_order, ctx)
            else
                # Field parameter × operator: S * M_f * S^{-1} * G
                cheb_dim = _find_chebyshev_dim(domain)
                left_fp = _extract_field_param(expr.left, prob)
                right_fp = _extract_field_param(expr.right, prob)

                if !isnothing(left_fp) && !isnothing(cheb_dim)
                    param, sc = left_fp
                    _field_param_is_zero(param, prob) && return spzeros(ComplexF64, N_per_var, N_per_var)
                    G = discretize_expr(expr.right, prob, N_per_var, highest_cheb_order, ctx)
                    result = _field_times_operator(param, G, prob, cheb_dim, highest_cheb_order, N_per_var, ctx)
                    return sc == ComplexF64(1.0) ? result : sc .* result
                elseif !isnothing(right_fp) && !isnothing(cheb_dim)
                    param, sc = right_fp
                    _field_param_is_zero(param, prob) && return spzeros(ComplexF64, N_per_var, N_per_var)
                    G = discretize_expr(expr.left, prob, N_per_var, highest_cheb_order, ctx)
                    result = _field_times_operator(param, G, prob, cheb_dim, highest_cheb_order, N_per_var, ctx)
                    return sc == ComplexF64(1.0) ? result : sc .* result
                else
                    return discretize_expr(expr.left, prob, N_per_var, highest_cheb_order, ctx) *
                           discretize_expr(expr.right, prob, N_per_var, highest_cheb_order, ctx)
                end
            end
        end

    elseif expr isa UnaryOpNode
        if expr.op == :-
            return -discretize_expr(expr.expr, prob, N_per_var, highest_cheb_order, ctx)
        end
    end

    error("Cannot discretize: $(typeof(expr))")
end

"""
Discretize expression in the T basis (no conversion to C^(p)).
Used as input to derivative operators.
"""
function discretize_expr_in_T(expr::ExprNode, prob::EVP, N_per_var::Int,
                              ctx::Union{Nothing, DiscretizationContext}=nothing)
    domain = prob.domain

    if expr isa VarNode
        return sparse(ComplexF64(1.0) * I, N_per_var, N_per_var)

    elseif expr isa ParamNode
        val = prob.parameters[expr.name]
        cheb_dim = _find_chebyshev_dim(domain)
        if val isa Number
            return ComplexF64(val) * sparse(I, N_per_var, N_per_var)
        elseif val isa AbstractMatrix && size(val) == (N_per_var, N_per_var)
            return ComplexF64.(val)
        else
            isnothing(cheb_dim) && return ComplexF64.(spdiagm(0 => vec(val)))
            return _field_multiply_T_lifted(expr, prob, N_per_var, cheb_dim, ctx)
        end

    elseif expr isa ConstNode
        return ComplexF64(expr.value) * sparse(I, N_per_var, N_per_var)

    elseif expr isa BinaryOpNode
        if expr.op == :*
            return discretize_expr_in_T(expr.left, prob, N_per_var, ctx) *
                   discretize_expr_in_T(expr.right, prob, N_per_var, ctx)
        elseif expr.op == :+
            return discretize_expr_in_T(expr.left, prob, N_per_var, ctx) +
                   discretize_expr_in_T(expr.right, prob, N_per_var, ctx)
        elseif expr.op == :-
            return discretize_expr_in_T(expr.left, prob, N_per_var, ctx) -
                   discretize_expr_in_T(expr.right, prob, N_per_var, ctx)
        end

    elseif expr isa UnaryOpNode && expr.op == :-
        return -discretize_expr_in_T(expr.expr, prob, N_per_var, ctx)

    elseif expr isa DerivNode
        coord = expr.coord
        spec = domain.coords[coord]
        if spec isa FourierBasisSpec
            # Fourier derivatives: diagonal operator in coefficient space
            inner_mat = discretize_expr_in_T(expr.expr, prob, N_per_var, ctx)
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

            # D_chain maps T → C^(order). Convert back to T via S^{-1}
            # (memoized per order on the build context — dense O(N³) solve).
            S_inv_1d = if !isnothing(ctx)
                get!(ctx.sinv_1d_cache, order) do
                    S_1d = ComplexF64.(get_conversion_operator(domain, coord, 0, order))
                    sparse(Matrix(S_1d) \ Matrix(ComplexF64(1.0) * I, N, N))
                end
            else
                S_1d = ComplexF64.(get_conversion_operator(domain, coord, 0, order))
                sparse(Matrix(S_1d) \ Matrix(ComplexF64(1.0) * I, N, N))
            end

            # Lift the combined 1D operator to full grid (Kronecker product)
            op_1d = S_inv_1d * D_chain_1d
            D_full = _lift_to_2d(op_1d, coord, domain)

            inner_T = discretize_expr_in_T(inner, prob, N_per_var, ctx)
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
    _build_2d_field_multiply(f_vec, domain, cheb_dim, fourier_dim, highest_cheb_order)

Build the full 2D multiplication operator for a field f(y,z) in mixed
Fourier-Chebyshev coefficient space with conversion to C^(highest_cheb_order).

The operator has block-Toeplitz structure:
    M_f[block(m₁, m₂)] = S · M_z(f̂_{(m₁-m₂) mod Ny})

where f̂_m(z) are the Fourier modes of f, and M_z builds the Chebyshev
multiplication operator from z-coefficients.

State vector ordering: z varies fastest.
"""
function _build_2d_field_multiply(f_vec::AbstractVector, domain::Domain,
                                   cheb_dim::Symbol, fourier_dim::Symbol,
                                   highest_cheb_order::Int; coeff_tol::Float64=0.0)
    spec_z = domain.coords[cheb_dim]
    spec_y = domain.coords[fourier_dim]
    N_z = spec_z.N
    N_y = spec_y.N

    S_1d = ComplexF64.(get_conversion_operator(domain, cheb_dim, 0, highest_cheb_order))

    # Reshape f to (N_z, N_y) — z fastest
    f_physical = reshape(Float64.(f_vec), N_z, N_y)

    # FFT along y for each z-point → f_hat[iz, m] (complex Fourier coefficients)
    f_hat = fft(f_physical, 2) ./ N_y

    # For each Fourier mode shift Δm, compute Chebyshev coefficients of f̂_Δm(z)
    # and build M_z(f̂_Δm) — the N_z × N_z Chebyshev multiplication operator
    Mz_blocks = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, N_y)
    for dm in 0:N_y-1
        # f̂_dm(z) at Chebyshev points
        f_dm_at_z = f_hat[:, dm + 1]
        # Chebyshev coefficients (handle complex: do real and imag separately)
        c_real = chebyshev_coefficients(real.(f_dm_at_z))
        c_imag = chebyshev_coefficients(imag.(f_dm_at_z))
        c_complex = ComplexF64.(c_real) .+ im .* ComplexF64.(c_imag)
        # Build Chebyshev multiplication operator (extend to complex coefficients)
        M_z = _complex_multiplication_operator(c_complex, N_z)
        Mz_blocks[dm + 1] = S_1d * M_z
    end

    return _assemble_2d_block_toeplitz(Mz_blocks, N_z, N_y; coeff_tol=coeff_tol)
end

"""
Banded 2D field multiplication operator directly in the C^(λ) ultraspherical
basis (λ ≥ 1). Same Fourier-block-Toeplitz structure as `_build_2d_field_multiply`,
but each z-block is the banded ultraspherical multiplication operator for that
Fourier mode of the field — avoiding the dense inverse-conversion `S_z⁻¹` and
keeping the operator sparse (fill ∝ band/N_z).
"""
function _build_2d_field_multiply_banded(f_vec::AbstractVector, domain::Domain,
                                         cheb_dim::Symbol, fourier_dim::Symbol, λ::Int;
                                         coeff_tol::Float64=0.0)
    N_z = domain.coords[cheb_dim].N
    N_y = domain.coords[fourier_dim].N

    f_physical = reshape(Float64.(f_vec), N_z, N_y)
    f_hat = fft(f_physical, 2) ./ N_y

    Mz_blocks = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, N_y)
    for dm in 0:N_y-1
        f_dm = f_hat[:, dm + 1]
        c = ComplexF64.(chebyshev_coefficients(real.(f_dm))) .+
            im .* ComplexF64.(chebyshev_coefficients(imag.(f_dm)))
        Mz_blocks[dm + 1] = ultraspherical_multiplication_operator(c, N_z, λ)
    end

    return _assemble_2d_block_toeplitz(Mz_blocks, N_z, N_y; coeff_tol=coeff_tol)
end

"""
Banded full-grid C^(λ) multiplication operator for a field parameter, or
`nothing` when `highest_cheb_order == 0` (T basis — no conversion, handled by
the caller's fallback). Unifies the 1D (z-only, lifted) and 2D (varies in y and
z) cases; both stay sparse (banded ultraspherical, no dense `S⁻¹`).
"""
function _field_banded_operator(param::ParamNode, prob::EVP, cheb_dim::Symbol,
                                highest_cheb_order::Int, N_per_var::Int;
                                coeff_tol::Float64=0.0)
    highest_cheb_order >= 1 || return nothing
    domain = prob.domain
    N_z = domain.coords[cheb_dim].N
    f_vec = vec(prob.parameters[param.name])
    fourier_dim = _find_fourier_dim(domain)

    if isnothing(fourier_dim) || length(f_vec) == N_z
        # 1D (z-only) field
        c = chebyshev_coefficients(Float64.(real.(f_vec)))
        M_z = ComplexF64.(ultraspherical_multiplication_operator(c, N_z, highest_cheb_order))
        return size(M_z, 1) == N_per_var ? M_z : _lift_to_2d(M_z, cheb_dim, domain)
    else
        # 2D (y, z) field
        return _build_2d_field_multiply_banded(f_vec, domain, cheb_dim, fourier_dim, highest_cheb_order;
                                               coeff_tol=coeff_tol)
    end
end

"""
Discretize `f · G`, a field parameter `f` times an already-built operator `G`.
Uses the banded C^(λ) multiplication (1D or 2D field, sparse) for Chebyshev
order ≥ 1; falls back to the T-basis multiply (`S·M_f·S⁻¹`) for order 0.
"""
function _field_times_operator(param::ParamNode, G, prob::EVP, cheb_dim::Symbol,
                               highest_cheb_order::Int, N_per_var::Int,
                               ctx::Union{Nothing, DiscretizationContext})
    banded = _field_banded_operator(param, prob, cheb_dim, highest_cheb_order, N_per_var;
                                    coeff_tol=_ctx_coeff_tol(ctx))
    isnothing(banded) || return banded * G
    M_f = _field_multiply_T(param, ComplexF64(1.0), prob, cheb_dim, ctx)
    return _apply_field_multiply(M_f, G, prob.domain, cheb_dim, highest_cheb_order,
                                 N_per_var, ctx, param.name)
end

"""
Build the full 2D multiplication operator from mixed Fourier-Chebyshev
coefficient data. Unlike `_build_2d_field_multiply`, `c_vec` is already in
coefficient space with z coefficients varying fastest.
"""
function _build_2d_coeff_multiply(c_vec::AbstractVector, domain::Domain,
                                  cheb_dim::Symbol, fourier_dim::Symbol)
    spec_z = domain.coords[cheb_dim]
    spec_y = domain.coords[fourier_dim]
    N_z = spec_z.N
    N_y = spec_y.N
    N_total = N_y * N_z

    length(c_vec) == N_total ||
        throw(DimensionMismatch("Coefficient vector has length $(length(c_vec)), expected $N_total"))

    coeffs = reshape(ComplexF64.(c_vec), N_z, N_y)
    Mz_blocks = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, N_y)
    for dm in 0:N_y-1
        Mz_blocks[dm + 1] = _complex_multiplication_operator(coeffs[:, dm + 1], N_z)
    end

    return _assemble_2d_block_toeplitz(Mz_blocks, N_z, N_y)
end

function _assemble_2d_block_toeplitz(Mz_blocks::AbstractVector{<:SparseMatrixCSC{ComplexF64, Int}},
                                     N_z::Int, N_y::Int; coeff_tol::Float64=0.0)
    length(Mz_blocks) == N_y ||
        throw(DimensionMismatch("Expected $N_y Fourier blocks, got $(length(Mz_blocks))"))

    # Relative drop of near-zero Fourier-mode blocks: treat block dm as zero when its norm is
    # ≤ coeff_tol × the largest block norm — truncating the field's y-spectrum (smooth field ⇒
    # exponential decay ⇒ tiny error). Reference is the MAX block (not DC: zero-mean fields have
    # DC ≈ 0). coeff_tol = 0 keeps every nonzero block (exact, current behavior).
    keep = trues(N_y)
    if coeff_tol > 0
        norms = [iszero(nnz(b)) ? 0.0 : norm(nonzeros(b)) for b in Mz_blocks]
        mx = maximum(norms)
        if mx > 0
            for dm in 1:N_y
                keep[dm] = norms[dm] > coeff_tol * mx
            end
            ndrop = count(!, keep)
            if ndrop > 0
                kept_frac = sqrt(sum(abs2, norms[keep]) / sum(abs2, norms))
                @debug "block-Toeplitz coeff_tol drop" coeff_tol kept=(N_y - ndrop) total=N_y dropped_frac=(1 - kept_frac)
            end
        end
    end

    N_total = N_z * N_y
    estimated_nnz = N_y * sum(dm -> keep[dm] ? nnz(Mz_blocks[dm]) : 0, 1:N_y)
    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]
    sizehint!(rows, estimated_nnz)
    sizehint!(cols, estimated_nnz)
    sizehint!(vals, estimated_nnz)

    # Assemble the full block-Toeplitz matrix:
    # M_f[block(m1, m2)] = Mz_blocks[(m1 - m2) mod N_y + 1].
    for m1 in 0:N_y-1
        for m2 in 0:N_y-1
            dm = mod(m1 - m2, N_y)
            block = Mz_blocks[dm + 1]
            size(block) == (N_z, N_z) ||
                throw(DimensionMismatch("Block $(dm + 1) has size $(size(block)), expected ($N_z, $N_z)"))
            (nnz(block) == 0 || !keep[dm + 1]) && continue

            row_offset = m1 * N_z
            col_offset = m2 * N_z
            block_rows = rowvals(block)
            block_vals = nonzeros(block)
            for block_col in 1:N_z
                full_col = col_offset + block_col
                for idx in nzrange(block, block_col)
                    push!(rows, row_offset + block_rows[idx])
                    push!(cols, full_col)
                    push!(vals, block_vals[idx])
                end
            end
        end
    end

    return sparse(rows, cols, vals, N_total, N_total)
end

"""
Complex-valued Chebyshev multiplication operator.
Same algorithm as `multiplication_operator` but with complex coefficients.
"""
function _complex_multiplication_operator(c::AbstractVector{ComplexF64}, N::Int; tol::Float64=1e-14)
    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]

    for j in 1:N
        n = j - 1
        for k in 1:length(c)
            m = k - 1
            ck = c[k]
            abs(ck) < tol && continue

            idx1 = abs(m - n) + 1
            idx2 = m + n + 1

            if idx1 <= N
                push!(rows, idx1); push!(cols, j); push!(vals, ck / 2)
            end
            if idx2 <= N
                push!(rows, idx2); push!(cols, j); push!(vals, ck / 2)
            end
        end
    end

    M = sparse(rows, cols, vals, N, N)
    droptol!(M, tol)
    return M
end

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
function _full_conversion(domain::Domain, highest_cheb_order::Int,
                          ctx::Union{Nothing, DiscretizationContext}=nothing)
    if !isnothing(ctx)
        cached = get(ctx.s_full_cache, highest_cheb_order, nothing)
        !isnothing(cached) && return cached
    end
    cheb_dim = _find_chebyshev_dim(domain)
    result = if isnothing(cheb_dim)
        _full_identity(domain)
    else
        S = ComplexF64.(get_conversion_operator(domain, cheb_dim, 0, highest_cheb_order))
        _lift_to_2d(S, cheb_dim, domain)
    end
    isnothing(ctx) || (ctx.s_full_cache[highest_cheb_order] = result)
    return result
end

"""
Extract field parameter info from an expression that may be wrapped in negation
or scalar multiplication. Returns (ParamNode, scalar_factor) or nothing.
"""
# A field parameter whose value array is identically zero contributes a zero
# operator (field · anything = 0). Detecting it lets discretize skip building the
# (potentially dense 2D Fourier block-Toeplitz) multiply operator entirely — a big
# saving when a model sets background fields to zero in some limit.
_field_param_is_zero(param::ParamNode, prob::EVP) =
    haskey(prob.parameters, param.name) &&
    (v = prob.parameters[param.name]; v isa AbstractArray && all(iszero, v))

function _extract_field_param(expr::ExprNode, prob::EVP)
    if expr isa ParamNode && haskey(prob.parameters, expr.name) &&
       !(prob.parameters[expr.name] isa Number) &&
       !(prob.parameters[expr.name] isa AbstractMatrix)
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

"""Build T-basis multiplication operator for a field parameter with scaling. Returns 1D (N_z) or full 2D (N_per_var)."""
function _field_multiply_T(param::ParamNode, scale::ComplexF64, prob::EVP, cheb_dim::Symbol,
                           ctx::Union{Nothing, DiscretizationContext}=nothing)
    domain = prob.domain
    spec = domain.coords[cheb_dim]
    N_z = spec.N
    f_vec = vec(prob.parameters[param.name])
    fourier_dim = _find_fourier_dim(domain)

    key = (param.name, cheb_dim, fourier_dim)
    if !isnothing(ctx) && haskey(ctx.field_t_cache, key)
        ctx.field_t_hits += 1
        M = ctx.field_t_cache[key]
        return scale == ComplexF64(1.0) ? M : scale * M
    end

    M = if !isnothing(fourier_dim)
        N_y = domain.coords[fourier_dim].N
        if length(f_vec) == N_y * N_z
            _build_2d_field_multiply(f_vec, domain, cheb_dim, fourier_dim, 0;
                                     coeff_tol=_ctx_coeff_tol(ctx))
        elseif length(f_vec) == N_z
            c = chebyshev_coefficients(Float64.(f_vec))
            ComplexF64.(multiplication_operator(c, N_z))
        else
            error("Field parameter :$(param.name) has length $(length(f_vec)), " *
                  "expected $N_z (1D) or $(N_y*N_z) (2D)")
        end
    else
        length(f_vec) == N_z ||
            error("Field parameter :$(param.name) has length $(length(f_vec)), expected $N_z")
        c = chebyshev_coefficients(Float64.(f_vec))
        ComplexF64.(multiplication_operator(c, N_z))
    end

    if !isnothing(ctx)
        ctx.field_t_misses += 1
        ctx.field_t_cache[key] = M
    end

    return scale == ComplexF64(1.0) ? M : scale * M
end

function _converted_field_multiply(param::ParamNode, prob::EVP, N_per_var::Int,
                                   highest_cheb_order::Int,
                                   ctx::Union{Nothing, DiscretizationContext})
    domain = prob.domain
    cheb_dim = _find_chebyshev_dim(domain)
    isnothing(cheb_dim) && return ComplexF64.(spdiagm(0 => vec(prob.parameters[param.name])))

    M_t = _field_multiply_T(param, ComplexF64(1.0), prob, cheb_dim, ctx)
    if size(M_t, 1) == N_per_var
        return highest_cheb_order == 0 ? M_t : _full_conversion(domain, highest_cheb_order, ctx) * M_t
    else
        S_1d = ComplexF64.(get_conversion_operator(domain, cheb_dim, 0, highest_cheb_order))
        return _lift_to_2d(S_1d * M_t, cheb_dim, domain)
    end
end

function _field_multiply_T_lifted(param::ParamNode, prob::EVP, N_per_var::Int,
                                  cheb_dim::Symbol,
                                  ctx::Union{Nothing, DiscretizationContext})
    M_t = _field_multiply_T(param, ComplexF64(1.0), prob, cheb_dim, ctx)
    return size(M_t, 1) == N_per_var ? M_t : _lift_to_2d(M_t, cheb_dim, prob.domain)
end

"""
    _apply_field_multiply(M_f, G, domain, cheb_dim, highest, N_per_var, ctx, field_name)

Fallback field multiplication via `S * M_f * S^{-1} * G` (T-basis operator `M_f`
converted around the operand). Used only for Chebyshev order 0; order ≥ 1 fields
are built directly as banded operators by [`_field_banded_operator`].
"""
function _apply_field_multiply(M_f, G, domain, cheb_dim, highest_cheb_order, N_per_var,
                                ctx::Union{Nothing, DiscretizationContext}=nothing,
                                field_name::Union{Symbol, Nothing}=nothing)
    spec = domain.coords[cheb_dim]
    N_z = spec.N

    if size(M_f, 1) == N_per_var
        # 2D field (varies in y and z), or order-0 1D field: S * M_f * S^{-1} * G.
        # Cache S * M_f * S^{-1} per (field_name, cheb_order) — reused across equations.
        S_M_Sinv = if !isnothing(ctx) && !isnothing(field_name)
            key = (field_name, highest_cheb_order)
            get!(ctx.s_mf_sinv_cache, key) do
                S_full     = _full_conversion(domain, highest_cheb_order, ctx)
                S_inv_full = _full_conversion_inv(domain, highest_cheb_order, ctx)
                S_full * M_f * S_inv_full
            end
        else
            S_full     = _full_conversion(domain, highest_cheb_order, ctx)
            S_inv_full = _full_conversion_inv(domain, highest_cheb_order, ctx)
            S_full * M_f * S_inv_full
        end
        return S_M_Sinv * G
    else
        # 1D field in a 2D domain. Order 0 → multiply in T basis (no conversion);
        # the order ≥ 1 banded path is handled above when coefficients are given,
        # this dense product is a defensive fallback.
        op_1d = if highest_cheb_order == 0
            M_f
        else
            S_1d = ComplexF64.(get_conversion_operator(domain, cheb_dim, 0, highest_cheb_order))
            S_inv_1d = if !isnothing(ctx)
                get!(ctx.sinv_1d_cache, highest_cheb_order) do
                    sparse(Matrix(S_1d) \ Matrix(ComplexF64(1.0) * I, N_z, N_z))
                end
            else
                sparse(Matrix(S_1d) \ Matrix(ComplexF64(1.0) * I, N_z, N_z))
            end
            S_1d * M_f * S_inv_1d
        end
        op_full = _lift_to_2d(op_1d, cheb_dim, domain)
        return op_full * G
    end
end

"""Inverse of the full conversion operator (I_y ⊗ S_z^{-1})."""
function _full_conversion_inv(domain::Domain, highest_cheb_order::Int,
                               ctx::Union{Nothing, DiscretizationContext}=nothing)
    if !isnothing(ctx)
        cached = get(ctx.s_inv_full_cache, highest_cheb_order, nothing)
        !isnothing(cached) && return cached
    end
    cheb_dim = _find_chebyshev_dim(domain)
    if isnothing(cheb_dim)
        result = _full_identity(domain)
        isnothing(ctx) || (ctx.s_inv_full_cache[highest_cheb_order] = result)
        return result
    end
    spec = domain.coords[cheb_dim]
    N_z = spec.N
    S_1d = ComplexF64.(get_conversion_operator(domain, cheb_dim, 0, highest_cheb_order))
    S_inv_1d = if !isnothing(ctx)
        get!(ctx.sinv_1d_cache, highest_cheb_order) do
            sparse(Matrix(S_1d) \ Matrix(ComplexF64(1.0) * I, N_z, N_z))
        end
    else
        sparse(Matrix(S_1d) \ Matrix(ComplexF64(1.0) * I, N_z, N_z))
    end
    result = _lift_to_2d(S_inv_1d, cheb_dim, domain)
    isnothing(ctx) || (ctx.s_inv_full_cache[highest_cheb_order] = result)
    return result
end

# ──────────────────────────────────────────────────────────────────────────────
#  Derived variable resolution
# ──────────────────────────────────────────────────────────────────────────────

"""
    _sparse_block_inverse(op, domain; bcs=[]) -> SparseMatrixCSC

Compute the inverse of an operator, with optional boundary bordering for BCs.

For block-diagonal operators (Fourier×Chebyshev), each N_z×N_z block is
inverted independently — O(N_y × N_z³) instead of O((N_y*N_z)³).

When `bcs` are provided, the last rows of each block are replaced with
BC rows before inversion (boundary bordering). This ensures the inverse
is unique for operators with null spaces (e.g., dz(dz(v)) = rhs needs BCs).
"""
function _sparse_block_inverse(op::AbstractMatrix, domain::Domain;
                                bcs::Vector{BoundaryCondition}=BoundaryCondition[])
    cheb_dim = _find_chebyshev_dim(domain)
    fourier_dim = _find_fourier_dim(domain)
    N = size(op, 1)

    if isnothing(cheb_dim) || isnothing(fourier_dim)
        # 1D (or no Fourier dim): apply boundary bordering — replace the last rows with
        # BC rows before inverting — matching the 2D block path below. Without this a
        # null-space operator (e.g. dz(dz(v)) with Dirichlet BCs) is singular and the
        # solve throws (LAPACKException), which broke the legacy augment_derived=false path.
        M = Matrix(op)
        if !isempty(bcs) && !isnothing(cheb_dim)
            spec = domain.coords[cheb_dim]
            for (i, bc) in enumerate(bcs)
                deriv_order = count_bc_deriv_order(bc.expr)
                row = chebyshev_boundary_row(bc.side, deriv_order, N;
                                             a=spec.lower, b=spec.upper)
                M[N - i + 1, :] = ComplexF64.(row)
            end
        end
        return sparse(M \ Matrix(ComplexF64(1.0) * I, N, N))
    end

    N_z = domain.coords[cheb_dim].N
    N_y = domain.coords[fourier_dim].N
    @assert N_y * N_z == N

    # Check if operator is block-diagonal (each N_z × N_z block is independent).
    # Accumulate per-block ‖·‖² directly from the sparse structure — O(nnz),
    # no densification of the full operator.
    op_sp = op isa SparseMatrixCSC{ComplexF64, Int} ? op : sparse(ComplexF64.(op))
    is_block_diag = true
    let rows = rowvals(op_sp), vals = nonzeros(op_sp), tol2 = 1e-12^2
        offnorm2 = zeros(Float64, N_y, N_y)
        for col in 1:N
            m2 = (col - 1) ÷ N_z + 1
            for idx in nzrange(op_sp, col)
                m1 = (rows[idx] - 1) ÷ N_z + 1
                m1 == m2 && continue
                offnorm2[m1, m2] += abs2(vals[idx])
            end
        end
        is_block_diag = all(<=(tol2), offnorm2)
    end

    # Build BC rows for boundary bordering (if any)
    bc_rows_1d = Vector{Float64}[]
    if !isempty(bcs) && !isnothing(cheb_dim)
        spec = domain.coords[cheb_dim]
        for bc in bcs
            deriv_order = count_bc_deriv_order(bc.expr)
            row = chebyshev_boundary_row(bc.side, deriv_order, N_z;
                                         a=spec.lower, b=spec.upper)
            push!(bc_rows_1d, row)
        end
    end

    if is_block_diag
        block = Matrix{ComplexF64}(undef, N_z, N_z)  # reused dense buffer
        rows = rowvals(op_sp); vals = nonzeros(op_sp)
        H_blocks = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, N_y)
        zero_block = spzeros(ComplexF64, N_z, N_z)
        for m in 0:N_y-1
            rs = m * N_z + 1
            fill!(block, zero(ComplexF64))
            for col in rs:rs+N_z-1
                for idx in nzrange(op_sp, col)
                    block[rows[idx] - rs + 1, col - rs + 1] = vals[idx]
                end
            end

            # Apply boundary bordering: replace last rows with BC rows
            for (i, bc_row) in enumerate(bc_rows_1d)
                row_idx = N_z - i + 1
                block[row_idx, :] = ComplexF64.(bc_row)
            end

            # One LU does both the singularity test (via det of the factors,
            # matching the old abs(det) < 1e-14 skip) and the inverse.
            F = lu!(block, check=false)
            if !issuccess(F) || abs(det(F)) < 1e-14
                H_blocks[m + 1] = zero_block  # still singular after BCs (e.g., m=0 at k=0)
                continue
            end
            H_blocks[m + 1] = sparse(inv(F))
        end
        return blockdiag(H_blocks...)
    else
        # Non-block-diagonal: fall back to dense inverse
        return sparse(Matrix(op_sp) \ Matrix(ComplexF64(1.0) * I, N, N))
    end
end

"""
Discretize an operator expression (like Dh2) applied to a dummy variable.
Returns the operator matrix (N_per_var × N_per_var) in C^(0) = T basis.
"""
function _discretize_operator(expr::ExprNode, dummy::Union{Symbol,Nothing}, prob::EVP,
                              N_per_var::Int,
                              ctx::Union{Nothing, DiscretizationContext}=nothing)
    domain = prob.domain
    if expr isa VarNode && (isnothing(dummy) || expr.name == dummy)
        return _full_identity(domain)
    elseif expr isa DerivNode
        coord = expr.coord
        spec = domain.coords[coord]
        if spec isa ChebyshevBasisSpec
            order = count_chained_derivs(expr, coord)
            inner = unwrap_chained_derivs(expr, coord)
            scale = 2.0 / (spec.upper - spec.lower)
            N_z = spec.N
            D_chain_1d = sparse(ComplexF64(1.0) * I, N_z, N_z)
            for p in 0:order-1
                D_chain_1d = ComplexF64(scale) .* differentiation_operator(p, N_z) * D_chain_1d
            end
            # D_chain maps T→C^(order). Convert back to T with S^{-1}
            # (memoized per order on the build context — dense O(N³) solve):
            S_inv_1d = if !isnothing(ctx)
                get!(ctx.sinv_1d_cache, order) do
                    S_1d = ComplexF64.(get_conversion_operator(domain, coord, 0, order))
                    sparse(Matrix(S_1d) \ Matrix(ComplexF64(1.0) * I, N_z, N_z))
                end
            else
                S_1d = ComplexF64.(get_conversion_operator(domain, coord, 0, order))
                sparse(Matrix(S_1d) \ Matrix(ComplexF64(1.0) * I, N_z, N_z))
            end
            D_T_1d = S_inv_1d * D_chain_1d  # T→T derivative operator
            D_full = _lift_to_2d(D_T_1d, coord, domain)
            return D_full * _discretize_operator(inner, dummy, prob, N_per_var, ctx)
        elseif spec isa FourierBasisSpec
            inner_mat = _discretize_operator(expr.expr, dummy, prob, N_per_var, ctx)
            D_1d = fourier_diff_operator(spec.N, spec.L, 1)
            return _lift_to_2d(D_1d, coord, domain) * inner_mat
        end
    elseif expr isa BinaryOpNode && expr.op == :+
        return _discretize_operator(expr.left, dummy, prob, N_per_var, ctx) +
               _discretize_operator(expr.right, dummy, prob, N_per_var, ctx)
    elseif expr isa BinaryOpNode && expr.op == :-
        return _discretize_operator(expr.left, dummy, prob, N_per_var, ctx) -
               _discretize_operator(expr.right, dummy, prob, N_per_var, ctx)
    elseif expr isa BinaryOpNode && expr.op == :*
        sv = _try_scalar(expr.left, prob)
        if !isnothing(sv)
            return sv .* _discretize_operator(expr.right, dummy, prob, N_per_var, ctx)
        end
        sv = _try_scalar(expr.right, prob)
        if !isnothing(sv)
            return sv .* _discretize_operator(expr.left, dummy, prob, N_per_var, ctx)
        end
        return _discretize_operator(expr.left, dummy, prob, N_per_var, ctx) *
               _discretize_operator(expr.right, dummy, prob, N_per_var, ctx)
    elseif expr isa ParamNode
        val = prob.parameters[expr.name]
        if val isa Number
            return ComplexF64(val) * _full_identity(domain)
        else
            # Field param: build T-basis multiplication operator
            cheb_dim = _find_chebyshev_dim(domain)
            isnothing(cheb_dim) && return ComplexF64.(spdiagm(0 => vec(val)))
            return _field_multiply_T_lifted(expr, prob, N_per_var, cheb_dim, ctx)
        end
    elseif expr isa ConstNode
        return ComplexF64(expr.value) * _full_identity(domain)
    elseif expr isa WavenumberNode
        # k-dependent part — handled separately
        return _full_identity(domain)
    end
    error("Cannot discretize operator expression: $(typeof(expr))")
end

"""
Store pre-discretized matrices for a derived variable term.
These are applied per-k during assembly when H(k) is computed.
"""
function _store_derived_term!(derived_caches, kt::KTerm, eq_idx::Int, eq_order::Int,
                              prob::EVP, N_per_var::Int, N_vars::Int,
                              ctx::Union{Nothing, DiscretizationContext}=nothing)
    for (dname, dvar) in prob.derived_vars
        if dname in collect_var_names(kt.expr)
            coeff_mat = _extract_coefficient_of_var(kt.expr, dname, prob, N_per_var, eq_order, ctx)

            rhs_expanded = expand_substitutions(dvar.rhs, prob.substitutions)
            rhs_lowered = lower_derivatives(rhs_expanded, prob.domain)
            rhs_distributed = distribute_products(rhs_lowered)
            rhs_additive = separate_additive_terms(rhs_distributed)

            # All pieces in T basis: coeff_T, H, rhs_T.
            # Apply S_{0→eq_order} once at the end for C^(eq_order) output.
            S_full = _full_conversion(prob.domain, eq_order, ctx)

            for rhs_term in rhs_additive
                rhs_k_powers, rhs_reduced = extract_k_powers(rhs_term)
                total_k_powers = _merge_k_powers(kt.k_powers, rhs_k_powers)

                # Build rhs operator in T basis: use _discretize_operator which
                # handles all derivatives and stays in T basis (no S conversion).
                rhs_mat = _discretize_operator(rhs_reduced, nothing, prob, N_per_var, ctx)
                real_var_idx = find_target_variable(rhs_reduced, prob.variables)

                # Chain: S * coeff_T * H * rhs_T = T → T → T → T → C^(p) ✓
                coeff_with_S = S_full * coeff_mat

                push!(derived_caches[dname].terms,
                      (eq_idx, real_var_idx, total_k_powers, coeff_with_S, rhs_mat))
            end
            return
        end
    end
end

function _replace_var(expr::ExprNode, from::Symbol, to::Symbol)
    if expr isa VarNode
        return expr.name == from ? VarNode(to) : expr
    elseif expr isa DerivNode
        return DerivNode(_replace_var(expr.expr, from, to), expr.coord)
    elseif expr isa BinaryOpNode
        return BinaryOpNode(expr.op,
                            _replace_var(expr.left, from, to),
                            _replace_var(expr.right, from, to))
    elseif expr isa UnaryOpNode
        return UnaryOpNode(expr.op, _replace_var(expr.expr, from, to))
    elseif expr isa SubstitutionNode
        return SubstitutionNode(expr.name,
                                ExprNode[_replace_var(arg, from, to) for arg in expr.args])
    end
    return expr
end

"""
Extract the coefficient operator of a derived variable from a linear term.
Works in **T basis** — no S conversion included. The caller applies S once.

This accepts differentiated operators such as `dz(B0 * v)` by replacing the
derived variable with a dummy and discretizing the whole linear operator.
"""
function _extract_coefficient_of_var(expr::ExprNode, var_name::Symbol,
                                      prob::EVP, N_per_var::Int, eq_order::Int,
                                      ctx::Union{Nothing, DiscretizationContext}=nothing)
    vars = collect_var_names(expr)
    var_name in vars || error("Expression does not contain $var_name: $expr")

    others = filter(!=(var_name), vars)
    isempty(others) ||
        error("Derived-variable term for $var_name also contains variables $others: $expr")

    dummy = :_derived_coeff_dummy
    op_expr = _replace_var(expr, var_name, dummy)
    return _discretize_operator(op_expr, dummy, prob, N_per_var, ctx)
end

# ──────────────────────────────────────────────────────────────────────────────
#  Augment EVP: promote derived variables to regular variables
# ──────────────────────────────────────────────────────────────────────────────

"""
    _augment_derived_problem(prob::EVP) -> (augmented_prob, derived_order)

Rewrite an EVP with derived variables into an equivalent plain multi-variable
EVP: each derived variable `v` (defined by `Op(v) = rhs`) becomes a regular
variable governed by the constraint equation `0 == Op(v) - rhs`, and its
`@derive_bc` boundary conditions become ordinary BCs on `v`. The returned
problem has `derived_vars` empty, so the standard `discretize` machinery
assembles it as a sparse augmented system (no operator inverse).

Returns `(augmented_prob, derived_order)`. If there are no derived variables,
returns `(prob, Symbol[])`. Does not mutate `prob`; saves/restores the global
active-problem Ref.
"""
function _augment_derived_problem(prob::EVP)
    isempty(prob.derived_vars) && return prob, Symbol[]

    derived_order = collect(keys(prob.derived_vars))

    # Save the current active prob so EVP() constructor side-effect is undone
    saved_active = _active_prob[]

    new_vars = vcat(prob.variables, derived_order)
    new_prob = EVP(prob.domain; variables=new_vars, eigenvalue=prob.eigenvalue)

    # Copy parameters and substitutions (shallow copy is fine — values are immutable)
    for (name, val) in prob.parameters
        new_prob.parameters[name] = val
    end
    for (name, sub) in prob.substitutions
        new_prob.substitutions[name] = sub
    end

    # Copy existing equations and BCs verbatim
    append!(new_prob.equations, prob.equations)
    append!(new_prob.bcs, prob.bcs)

    # For each derived variable, add a constraint equation and promote its BCs
    for dname in derived_order
        dvar = prob.derived_vars[dname]
        # Build Op(v) as a SubstitutionNode referencing the stored operator
        op_lhs = SubstitutionNode(dvar.operator_name, ExprNode[VarNode(dname)])
        # Constraint: 0 == Op(v) - rhs  →  lhs=ConstNode(0), rhs=Op(v)-rhs_expr
        constraint_rhs = BinaryOpNode(:-, op_lhs, dvar.rhs)
        push!(new_prob.equations, Equation(ConstNode(0.0), constraint_rhs))
        # Move the derive BCs into the regular BC list
        append!(new_prob.bcs, dvar.bcs)
    end

    # Restore the active problem to what it was before we created new_prob
    _active_prob[] = saved_active
    return new_prob, derived_order
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

"""
Build BC rows for the full system.

Returns `(bc_info, rhs_values)` where `bc_info` is a vector of tuples:
`(row_idx, k_powers, a_row, b_row)` — the A-side row, B-side row, and their k powers.

For **algebraic BCs** (no eigenvalue): A row = boundary evaluation, B row = zeros, k_power = 0.
For **dynamic BCs** (eigenvalue-dependent): the expression is split like an equation —
eigenvalue terms go to B, the rest to A, with k-separation on both sides.
"""
function build_bc_rows(prob::EVP, N_per_var::Int, N_total::Int)
    domain = prob.domain

    cheb_dim = _find_chebyshev_dim(domain)
    fourier_dim = _find_fourier_dim(domain)
    N_z = isnothing(cheb_dim) ? N_per_var : domain.coords[cheb_dim].N
    N_y = isnothing(fourier_dim) ? 1 : domain.coords[fourier_dim].N
    @assert N_y * N_z == N_per_var "Grid mismatch: N_y=$N_y * N_z=$N_z != N_per_var=$N_per_var"

    # Collect: (row_idx, k_powers, a_row_vec, b_row_vec)
    bc_info = Tuple{Int, KPowerKey, Vector{ComplexF64}, Vector{ComplexF64}}[]
    # Keep the RHS complex so an imaginary inhomogeneous BC (e.g. a direct
    # add_bc!(..., 1im) that bypasses @bc's Float64 coercion) is still detected
    # by the inhomogeneous-BC check rather than silently dropped to zero.
    rhs_values = ComplexF64[]
    bc_count = Dict{Symbol, Int}()

    for bc in prob.bcs
        var_names = collect_var_names(bc.expr)
        primary_var = prob.variables[1]
        for v in prob.variables
            if v in var_names
                primary_var = v
                break
            end
        end
        var_idx = findfirst(==(primary_var), prob.variables)

        count = get(bc_count, primary_var, 0) + 1
        bc_count[primary_var] = count

        var_block_start = (var_idx - 1) * N_per_var
        spec = domain.coords[bc.coord]

        if !bc.is_dynamic
            # Algebraic BC: A row = boundary eval, B row = zeros, k_power = 0
            bc_row_1d = zeros(ComplexF64, N_z)
            _build_bc_row_1d!(bc_row_1d, bc.expr, bc.side, bc.coord, prob;
                              target_var=primary_var)

            for m in 0:N_y-1
                row_idx = var_block_start + m * N_z + (N_z - count + 1)
                a_row = zeros(ComplexF64, N_total)
                for (col_var_idx, col_var) in enumerate(prob.variables)
                    col_row_1d = col_var == primary_var ? bc_row_1d : zeros(ComplexF64, N_z)
                    col_var == primary_var ||
                        _build_bc_row_1d!(col_row_1d, bc.expr, bc.side, bc.coord, prob;
                                          target_var=col_var)
                    col_start = (col_var_idx - 1) * N_per_var + m * N_z + 1
                    a_row[col_start:col_start+N_z-1] = col_row_1d
                end

                push!(bc_info, (row_idx, (), a_row, zeros(ComplexF64, N_total)))
                push!(rhs_values, ComplexF64(bc.rhs))
            end
        else
            # Dynamic BC: expression contains eigenvalue.
            # Expand substitutions and lower derivatives, then split into
            # eigenvalue-side (→ B) and remainder (→ A), with k-separation.
            expr_expanded = expand_substitutions(bc.expr, prob.substitutions)
            expr_lowered = lower_derivatives(expr_expanded, prob.domain)

            # Separate additive terms and classify as eigenvalue or not
            all_terms = separate_additive_terms(distribute_products(expr_lowered))

            # For each term: is it eigenvalue-dependent?
            # sigma * f(psi) → strip sigma, f(psi) goes to B
            # g(psi)         → goes to A
            a_terms_by_k = Dict{KPowerKey, Vector{ExprNode}}()
            b_terms_by_k = Dict{KPowerKey, Vector{ExprNode}}()

            for t in all_terms
                if _contains_eigenvalue(t, prob.eigenvalue)
                    # Strip eigenvalue from this term
                    inner = _strip_eigenvalue_from_term(t, prob.eigenvalue)
                    k_powers, reduced = extract_k_powers(inner)
                    terms_list = get!(b_terms_by_k, k_powers, ExprNode[])
                    push!(terms_list, reduced)
                else
                    k_powers, reduced = extract_k_powers(t)
                    terms_list = get!(a_terms_by_k, k_powers, ExprNode[])
                    push!(terms_list, reduced)
                end
            end

            # Build 1D boundary rows for each k-power component
            for m in 0:N_y-1
                row_idx = var_block_start + m * N_z + (N_z - count + 1)
                # Collect all k-powers from both A and B sides
                all_k_powers = union(keys(a_terms_by_k), keys(b_terms_by_k))

                for kp in all_k_powers
                    a_row = zeros(ComplexF64, N_total)
                    b_row = zeros(ComplexF64, N_total)

                    # A-side terms for this k-power
                    if haskey(a_terms_by_k, kp)
                        for (col_var_idx, col_var) in enumerate(prob.variables)
                            a_1d = zeros(ComplexF64, N_z)
                            for term in a_terms_by_k[kp]
                                _build_bc_row_1d!(a_1d, term, bc.side, bc.coord, prob;
                                                  target_var=col_var)
                            end
                            col_start = (col_var_idx - 1) * N_per_var + m * N_z + 1
                            a_row[col_start:col_start+N_z-1] = a_1d
                        end
                    end

                    # B-side terms for this k-power
                    if haskey(b_terms_by_k, kp)
                        for (col_var_idx, col_var) in enumerate(prob.variables)
                            b_1d = zeros(ComplexF64, N_z)
                            for term in b_terms_by_k[kp]
                                _build_bc_row_1d!(b_1d, term, bc.side, bc.coord, prob;
                                                  target_var=col_var)
                            end
                            col_start = (col_var_idx - 1) * N_per_var + m * N_z + 1
                            b_row[col_start:col_start+N_z-1] = b_1d
                        end
                    end

                    push!(bc_info, (row_idx, kp, a_row, b_row))
                    push!(rhs_values, ComplexF64(bc.rhs))
                end
            end
        end
    end

    return bc_info, rhs_values
end

"""
Strip the eigenvalue symbol from a multiplicative term.
`sigma * f` → `f`, `-sigma * f` → `-f`, `3 * sigma * f` → `3 * f`.
"""
function _strip_eigenvalue_from_term(expr::ExprNode, eig::Symbol)
    if expr isa EigenvalueNode
        return ConstNode(1.0)
    elseif expr isa BinaryOpNode && expr.op == :*
        if _contains_eigenvalue(expr.left, eig) && !_contains_eigenvalue(expr.right, eig)
            stripped_left = _strip_eigenvalue_from_term(expr.left, eig)
            # If stripping produced ConstNode(1.0), simplify
            if stripped_left isa ConstNode && stripped_left.value == 1.0
                return expr.right
            end
            return BinaryOpNode(:*, stripped_left, expr.right)
        elseif _contains_eigenvalue(expr.right, eig) && !_contains_eigenvalue(expr.left, eig)
            stripped_right = _strip_eigenvalue_from_term(expr.right, eig)
            if stripped_right isa ConstNode && stripped_right.value == 1.0
                return expr.left
            end
            return BinaryOpNode(:*, expr.left, stripped_right)
        else
            # Both sides contain eigenvalue — recurse
            return BinaryOpNode(:*, _strip_eigenvalue_from_term(expr.left, eig),
                                    _strip_eigenvalue_from_term(expr.right, eig))
        end
    elseif expr isa UnaryOpNode
        return UnaryOpNode(expr.op, _strip_eigenvalue_from_term(expr.expr, eig))
    else
        return expr
    end
end

"""
Build a 1D Chebyshev boundary row (length N_z) by walking the BC expression tree.
Uses ComplexF64 scale to handle `im` factors from lowered derivatives.
Field parameters are evaluated at the boundary point to become scalar multipliers.
"""
function _build_bc_row_1d!(row_vec::AbstractVector, expr::ExprNode,
                           side::Symbol, coord::Symbol, prob::EVP;
                           scale::ComplexF64=ComplexF64(1.0),
                           target_var::Union{Symbol,Nothing}=nothing)
    spec = prob.domain.coords[coord]
    N = spec.N

    if expr isa VarNode
        if target_var !== nothing && expr.name != target_var
            return
        end
        bc_row = chebyshev_boundary_row(side, 0, N; a=spec.lower, b=spec.upper)
        row_vec .+= scale .* bc_row

    elseif expr isa ConstNode
        # Bare constant (e.g., from distribute): treat as scalar scale
        # This shouldn't normally produce a row by itself — skip if no variable
        # But can appear as part of a product chain that was simplified
        return

    elseif expr isa DerivNode
        if expr.coord != coord
            error("BC expression contains derivative in direction :$(expr.coord), " *
                  "but boundary is on :$(coord).")
        end
        deriv_order = count_chained_derivs_any(expr, coord)
        inner = unwrap_chained_derivs_any(expr, coord)

        if inner isa VarNode
            if target_var !== nothing && inner.name != target_var
                return
            end
            bc_row = chebyshev_boundary_row(side, deriv_order, N; a=spec.lower, b=spec.upper)
            row_vec .+= scale .* bc_row
        else
            distributed = _distribute_deriv(inner, coord, deriv_order)
            _build_bc_row_1d!(row_vec, distributed, side, coord, prob;
                              scale=scale, target_var=target_var)
        end

    elseif expr isa BinaryOpNode && expr.op == :+
        _build_bc_row_1d!(row_vec, expr.left, side, coord, prob;
                          scale=scale, target_var=target_var)
        _build_bc_row_1d!(row_vec, expr.right, side, coord, prob;
                          scale=scale, target_var=target_var)

    elseif expr isa BinaryOpNode && expr.op == :-
        _build_bc_row_1d!(row_vec, expr.left, side, coord, prob;
                          scale=scale, target_var=target_var)
        _build_bc_row_1d!(row_vec, expr.right, side, coord, prob;
                          scale=-scale, target_var=target_var)

    elseif expr isa BinaryOpNode && expr.op == :*
        # Try to resolve either side as a scalar (const, scalar param, or field at boundary)
        lv = _try_bc_scalar(expr.left, side, coord, prob)
        rv = _try_bc_scalar(expr.right, side, coord, prob)

        if !isnothing(lv)
            _build_bc_row_1d!(row_vec, expr.right, side, coord, prob;
                              scale=scale * lv, target_var=target_var)
        elseif !isnothing(rv)
            _build_bc_row_1d!(row_vec, expr.left, side, coord, prob;
                              scale=scale * rv, target_var=target_var)
        else
            error("BC multiplication must involve a scalar or field parameter: " *
                  "got $(expr.left) * $(expr.right)")
        end

    elseif expr isa UnaryOpNode && expr.op == :-
        _build_bc_row_1d!(row_vec, expr.expr, side, coord, prob;
                          scale=-scale, target_var=target_var)

    else
        error("Unsupported expression in BC: $(typeof(expr))")
    end
end

"""
Try to evaluate a BC expression node as a scalar (complex).
Handles: ConstNode, scalar ParamNode, field ParamNode (evaluated at boundary), negation, products of scalars.
"""
function _try_bc_scalar(expr::ExprNode, side::Symbol, coord::Symbol, prob::EVP)::Union{ComplexF64, Nothing}
    if expr isa ConstNode
        return ComplexF64(expr.value)
    elseif expr isa ParamNode && haskey(prob.parameters, expr.name)
        val = prob.parameters[expr.name]
        if val isa Number
            return ComplexF64(val)
        else
            # Field parameter: evaluate at the boundary point
            spec = prob.domain.coords[coord]
            x_ref = chebyshev_points(spec.N, spec.lower, spec.upper)
            f_vals = Float64.(vec(val))
            boundary_val = side == :left ? f_vals[end] : f_vals[1]  # CGL: first point is right (x=b)
            return ComplexF64(boundary_val)
        end
    elseif expr isa UnaryOpNode && expr.op == :-
        inner = _try_bc_scalar(expr.expr, side, coord, prob)
        return isnothing(inner) ? nothing : -inner
    elseif expr isa BinaryOpNode && expr.op == :*
        l = _try_bc_scalar(expr.left, side, coord, prob)
        r = _try_bc_scalar(expr.right, side, coord, prob)
        return (isnothing(l) || isnothing(r)) ? nothing : l * r
    end
    return nothing
end


# Note: The old _build_bc_row! function has been replaced by _build_bc_row_1d!
# which operates on 1D Chebyshev rows and is replicated across Fourier modes
# in build_bc_rows above.

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
    _discretize_n_total(prob; augment_derived=true) -> Int

Post-augmentation system size `N_total = total_grid_size(domain) * N_vars`, computed
WITHOUT building any operator. With `augment_derived=true`, derived variables are
rewritten into extra unknown blocks, so this augments first to count them — matching
`discretize`'s internal `N_total` exactly. Used by the distributed path to compute the
PETSc row split before calling `discretize(prob; row_range=...)`.
"""
function _discretize_n_total(prob::EVP; augment_derived::Bool=true)
    if augment_derived && !isempty(prob.derived_vars)
        prob, _ = _augment_derived_problem(prob)
    end
    return total_grid_size(prob.domain) * length(prob.variables)
end

"""
    discretize(prob::EVP; augment_derived=true) -> DiscretizationCache

Validate the problem, expand substitutions, lower derivatives, separate by k-power,
and build sparse matrix components. Returns a cache for fast wavenumber assembly.

Derived variables (`@derive`) are **augmented** into the system by default: each
becomes a regular unknown governed by its defining equation `Op(v) = rhs`, which
keeps the assembled operator sparse (no dense `Op⁻¹`) and handles operators whose
inverse the legacy path cannot form. Pass `augment_derived=false` to use the legacy
path that eliminates derived variables via an explicit operator inverse (a smaller
but denser system). The augmented form is a singular-`B` descriptor system; `solve`
filters the resulting infinite/spurious modes, but shift-target the physical region.
"""
function discretize(prob::EVP; augment_derived::Bool=true,
                    row_range::Union{Nothing,Tuple{Int,Int}}=nothing,
                    coeff_tol::Float64=0.0)
    (0.0 <= coeff_tol < 1.0) || throw(ArgumentError("coeff_tol=$(coeff_tol) must be in [0, 1)"))
    validate_problem(prob)

    # Augmented (descriptor) form: rewrite derived variables into regular
    # variables + constraint equations so the assembled operator stays sparse
    # (no Op⁻¹). Record the derived block order for reconstruction.
    derived_var_order = Symbol[]
    if augment_derived && !isempty(prob.derived_vars)
        prob, derived_var_order = _augment_derived_problem(prob)
    end

    N_per_var = total_grid_size(prob.domain)
    N_vars = length(prob.variables)
    N_total = N_per_var * N_vars

    # Owned-row slice. nothing → full build (rstart=0, nrows=N_total): every branch
    # below reduces to the original behavior. Set → build only [rstart,rend) (0-based).
    rstart, rend = row_range === nothing ? (0, N_total) : row_range
    (0 <= rstart <= rend <= N_total) ||
        throw(ArgumentError("row_range=$(row_range) out of [0,$N_total]"))
    nrows = rend - rstart
    restricted = row_range !== nothing

    A_buffers = Dict{KPowerKey, _COOBuf}()
    B_buffers = Dict{KPowerKey, _COOBuf}()
    ctx = DiscretizationContext(coeff_tol)

    # Compute per-equation highest Chebyshev derivative order.
    # Each equation lives in its own C^(p) basis where p is the max z-derivative
    # in that equation. This avoids over-converting variables that appear in
    # low-order equations (e.g., a constraint with no z-derivatives).
    eq_cheb_orders = Int[]
    eq_lowered_rhs = ExprNode[]
    eq_lowered_lhs = ExprNode[]

    for eq in prob.equations
        rhs_exp = expand_substitutions(eq.rhs, prob.substitutions)
        lhs_exp = expand_substitutions(eq.lhs, prob.substitutions)
        rhs_low = lower_derivatives(rhs_exp, prob.domain)
        lhs_low = lower_derivatives(lhs_exp, prob.domain)
        push!(eq_lowered_rhs, rhs_low)
        push!(eq_lowered_lhs, lhs_low)

        order = 0
        for dim in prob.domain.resolved_dims
            if prob.domain.coords[dim] isa ChebyshevBasisSpec
                order = max(order, max_deriv_order(rhs_low, dim))
                order = max(order, max_deriv_order(lhs_low, dim))
            end
        end
        push!(eq_cheb_orders, order)
    end

    # Pre-compute derived variable operator components.
    # The operator (e.g., Dh2 = dx² + dy²) is split into k-independent part (dy²)
    # and k² coefficient (-1 from dx²). H(k) = (op_k0 + k² * coeff * I)^{-1}
    # is recomputed per-wavenumber during assemble.
    derived_caches = Dict{Symbol, DerivedVarCache}()

    for (dname, dvar) in prob.derived_vars
        op_expr = expand_substitutions(
            SubstitutionNode(dvar.operator_name, [VarNode(:_dummy_)]),
            prob.substitutions
        )
        op_expr_lowered = lower_derivatives(op_expr, prob.domain)

        # Separate operator into k-independent and k-dependent parts
        op_terms = separate_by_k_power(distribute_products(op_expr_lowered))

        op_k0 = spzeros(ComplexF64, N_per_var, N_per_var)
        op_k2_coeff = ComplexF64(0.0)
        op_k_components = Dict{KPowerKey, SparseMatrixCSC{ComplexF64, Int}}()

        for kt in op_terms
            mat = _discretize_operator(kt.expr, :_dummy_, prob, N_per_var, ctx)
            if isempty(kt.k_powers)
                op_k0 = op_k0 + mat
            else
                _add_component!(op_k_components, kt.k_powers, mat)
            end

            if kt.k_power == 2
                # The k² term is typically a scalar times identity (from dx(dx(v)) → -k²*v)
                # Extract the scalar coefficient
                diag_val = mat[1, 1]
                op_k2_coeff += diag_val
            end
        end

        derived_caches[dname] = DerivedVarCache(op_k0, op_k2_coeff, op_k_components, dvar.bcs,
            Tuple{Int, Int, KPowerKey, SparseMatrixCSC{ComplexF64, Int}, SparseMatrixCSC{ComplexF64, Int}}[])
    end

    for (eq_idx, eq) in enumerate(prob.equations)
        # Skip equations whose row-block does not intersect the owned slice. Equation
        # eq_idx writes only into block row eq_idx (row_off = (eq_idx-1)*N_per_var), so
        # its (empty-for-us) contribution can be dropped entirely — the build-time win.
        if restricted
            bs = (eq_idx - 1) * N_per_var
            be = eq_idx * N_per_var
            max(bs, rstart) < min(be, rend) || continue
        end

        rhs_lowered = eq_lowered_rhs[eq_idx]
        lhs_lowered = eq_lowered_lhs[eq_idx]
        eq_order = eq_cheb_orders[eq_idx]

        # RHS: separate by k-power, discretize using THIS equation's order
        rhs_terms = separate_by_k_power(rhs_lowered)

        for kt in rhs_terms
            # Check if this term targets a derived variable
            target_vars = collect_var_names(kt.expr)
            derived_targets = filter(v -> haskey(prob.derived_vars, v), target_vars)

            if !isempty(derived_targets)
                # Store pre-discretized matrices for per-k H assembly. Derived caches
                # stay FULL; assemble_rows row-restricts them via _place_in_block_rows,
                # so storing only owned-equation terms is consistent (non-owned rows
                # contribute nothing to our slice anyway).
                _store_derived_term!(derived_caches, kt, eq_idx, eq_order,
                                    prob, N_per_var, N_vars, ctx)
                continue
            end

            mat = discretize_expr(kt.expr, prob, N_per_var, eq_order, ctx)
            var_idx = find_target_variable(kt.expr, prob.variables)
            if restricted
                _scatter_block_rows!(A_buffers, kt.k_powers, mat, eq_idx, var_idx,
                                     N_per_var, rstart, rend)
            else
                _scatter_block!(A_buffers, kt.k_powers, mat, eq_idx, var_idx, N_per_var)
            end
        end

        # LHS: eigenvalue side → B matrix (also needs k-separation)
        # Check if this is a constraint equation (no eigenvalue on LHS)
        has_eigenvalue = _contains_eigenvalue(lhs_lowered, prob.eigenvalue)

        if has_eigenvalue
            lhs_inner = strip_eigenvalue(lhs_lowered)
            lhs_var_idx = find_target_variable(lhs_inner, prob.variables)
            lhs_terms = separate_by_k_power(lhs_inner)

            for kt in lhs_terms
                mat = discretize_expr(kt.expr, prob, N_per_var, eq_order, ctx)
                if restricted
                    _scatter_block_rows!(B_buffers, kt.k_powers, mat, eq_idx, lhs_var_idx,
                                         N_per_var, rstart, rend)
                else
                    _scatter_block!(B_buffers, kt.k_powers, mat, eq_idx, lhs_var_idx, N_per_var)
                end
            end
        end
        # If no eigenvalue on LHS → constraint equation, B rows stay zero
    end

    # In restricted mode: ensure ALL k-power keys from every equation are present in
    # the buffers (possibly as empty entries), even for equations whose rows were
    # entirely skipped above. This guarantees the restricted cache has the same key
    # set as the full build — required by restrict_cache_rows and assemble_rows.
    # Cheap: walks already-lowered ASTs via separate_by_k_power, no discretize_expr calls.
    if restricted
        for eq_idx2 in 1:length(prob.equations)
            for kt in separate_by_k_power(eq_lowered_rhs[eq_idx2])
                target_vars = collect_var_names(kt.expr)
                isempty(filter(v -> haskey(prob.derived_vars, v), target_vars)) || continue
                get!(A_buffers, kt.k_powers, (Int[], Int[], ComplexF64[]))
            end
            lhs_low2 = eq_lowered_lhs[eq_idx2]
            if _contains_eigenvalue(lhs_low2, prob.eigenvalue)
                for kt in separate_by_k_power(strip_eigenvalue(lhs_low2))
                    get!(B_buffers, kt.k_powers, (Int[], Int[], ComplexF64[]))
                end
            end
        end
    end

    A_components = _finalize_components(A_buffers, nrows, N_total)
    B_components = _finalize_components(B_buffers, nrows, N_total)

    # Apply BCs
    bc_info, rhs_values = build_bc_rows(prob, N_per_var, N_total)

    # Check for inhomogeneous BCs
    for (i, rhs) in enumerate(rhs_values)
        if abs(rhs) > 0.0
            error("Inhomogeneous boundary condition (rhs = $rhs) is not supported " *
                  "for eigenvalue problems. Use homogeneous BCs (rhs == 0).")
        end
    end

    # Collect owned BC rows (local indices) for zeroing
    owned_bc_rows = Set{Int}()
    for (row_idx, _, _, _) in bc_info
        lr = row_idx - rstart
        (1 <= lr <= nrows) && push!(owned_bc_rows, lr)
    end

    # First, zero out owned BC rows in ALL k-power components (both A and B) —
    # one O(nnz) pass per component instead of a sparse row-assignment per row.
    if !isempty(owned_bc_rows)
        for p in keys(A_components)
            _zero_sparse_rows!(A_components[p], owned_bc_rows)
        end
        for p in keys(B_components)
            _zero_sparse_rows!(B_components[p], owned_bc_rows)
        end
    end

    # Then write owned BC rows into the correct k-power components: gather all BC
    # entries per key as COO triplets and add each key's sparse matrix once
    # (element-wise sparse `+=` is an O(nnz) insertion per entry).
    bc_bufA = Dict{KPowerKey, _COOBuf}()
    bc_bufB = Dict{KPowerKey, _COOBuf}()
    for (row_idx, kp, a_row, b_row) in bc_info
        # Register the BC's k-power key in BOTH A and B regardless of row ownership, so a
        # restricted cache has the same key set as the full build (mirrors restrict_cache_rows,
        # which keeps every key as a sliced/zero block). Dynamic BCs can carry non-() keys that
        # no equation term produces; those must still appear (empty) on ranks that don't own the
        # BC row.
        if !haskey(A_components, kp)
            A_components[kp] = spzeros(ComplexF64, nrows, N_total)
        end
        if !haskey(B_components, kp)
            B_components[kp] = spzeros(ComplexF64, nrows, N_total)
        end
        lr = row_idx - rstart
        (1 <= lr <= nrows) || continue
        Ia, Ja, Va = get!(() -> (Int[], Int[], ComplexF64[]), bc_bufA, kp)
        for (j, v) in enumerate(a_row)
            v != 0.0 && (push!(Ia, lr); push!(Ja, j); push!(Va, ComplexF64(v)))
        end
        Ib, Jb, Vb = get!(() -> (Int[], Int[], ComplexF64[]), bc_bufB, kp)
        for (j, v) in enumerate(b_row)
            v != 0.0 && (push!(Ib, lr); push!(Jb, j); push!(Vb, ComplexF64(v)))
        end
    end
    for (kp, (Ia, Ja, Va)) in bc_bufA
        isempty(Va) && continue
        A_components[kp] = A_components[kp] + sparse(Ia, Ja, Va, nrows, N_total)
    end
    for (kp, (Ib, Jb, Vb)) in bc_bufB
        isempty(Vb) && continue
        B_components[kp] = B_components[kp] + sparse(Ib, Jb, Vb, nrows, N_total)
    end

    return DiscretizationCache(A_components, B_components, derived_caches,
                               N_total, N_per_var, N_vars, prob.domain,
                               derived_var_order, row_range, ReconstructMemo())
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
    _k_values(cache, k) -> Dict{Symbol,Float64}

Build the wavenumber dict for a single scalar `k`: `:_total_k => k` when there are
no FourierTransformed directions, else `Symbol(:k_, dim) => k` for each. Shared by
`assemble` and the row-wise `assemble_rows` so they agree exactly.
"""
function _k_values(cache::DiscretizationCache, k::Float64)
    k_vals = Dict{Symbol, Float64}()
    if isempty(cache.domain.transformed_dims)
        k_vals[:_total_k] = k
    else
        for dim in cache.domain.transformed_dims
            k_vals[Symbol(:k_, dim)] = k
        end
    end
    return k_vals
end

function assemble(cache::DiscretizationCache, k::Float64)
    return _assemble(cache, _k_values(cache, k))
end

# Append the scaled triplets of every k-power component to (I, J, V). The final
# `sparse(I, J, V, ...)` sums duplicates — identical to the old repeated sparse `+`,
# without allocating a fresh full-size accumulator per component.
function _append_component_triplets!(I::Vector{Int}, J::Vector{Int}, V::Vector{ComplexF64},
                                     components::Dict{KPowerKey, SparseMatrixCSC{ComplexF64, Int}},
                                     k_vals::Dict{Symbol, Float64})
    total = 0
    for (kp, M) in components
        _k_coeff(kp, k_vals) == 0.0 && continue
        total += nnz(M)
    end
    sizehint!(I, length(I) + total)
    sizehint!(J, length(J) + total)
    sizehint!(V, length(V) + total)
    for (kp, M) in components
        c = _k_coeff(kp, k_vals)
        c == 0.0 && continue
        rows = rowvals(M); vals = nonzeros(M)
        for col in 1:size(M, 2)
            for idx in nzrange(M, col)
                push!(I, rows[idx])
                push!(J, col)
                push!(V, c == 1.0 ? vals[idx] : c * vals[idx])
            end
        end
    end
    return nothing
end

function _assemble(cache::DiscretizationCache, k_vals::Dict{Symbol, Float64})
    cache.row_range === nothing ||
        throw(ArgumentError("assemble needs a full (unrestricted) cache; got " *
            "row_range=$(cache.row_range). Its components are row-sliced — " *
            "use assemble_rows(cache, k, rstart, rend) instead."))
    N = cache.N_total
    N_per_var = cache.N_per_var

    AI = Int[]; AJ = Int[]; AV = ComplexF64[]
    _append_component_triplets!(AI, AJ, AV, cache.A_components, k_vals)

    BI = Int[]; BJ = Int[]; BV = ComplexF64[]
    _append_component_triplets!(BI, BJ, BV, cache.B_components, k_vals)
    B = sparse(BI, BJ, BV, N, N)

    # Apply derived variable terms with k-dependent H(k), scattered straight into
    # the COO buffers (no full-size place_in_block temporary per term).
    for (dname, dc) in cache.derived_caches
        isempty(dc.terms) && continue

        # Build H(k) = (op_k0 + k² * coeff * I)^{-1}
        op_k = dc.op_k0
        for (kp, mat) in dc.op_k_components
            coeff = _k_coeff(kp, k_vals)
            coeff == 0.0 && continue
            op_k = op_k + coeff * mat
        end
        H_k = _sparse_block_inverse(op_k, cache.domain; bcs=dc.bcs)

        for (eq_idx, var_idx, total_kp, coeff_mat, rhs_mat) in dc.terms
            w = _k_coeff(total_kp, k_vals)
            w == 0.0 && continue
            combined = coeff_mat * H_k * rhs_mat
            row_off = (eq_idx - 1) * N_per_var
            col_off = (var_idx - 1) * N_per_var
            rows = rowvals(combined); vals = nonzeros(combined)
            for col in 1:size(combined, 2)
                for idx in nzrange(combined, col)
                    push!(AI, rows[idx] + row_off)
                    push!(AJ, col + col_off)
                    push!(AV, w * vals[idx])
                end
            end
        end
    end

    A = sparse(AI, AJ, AV, N, N)
    return A, B
end

"""
    assemble(cache; k_x=0.0, k_y=0.0, ...) -> (A, B)

Assemble for multiple FourierTransformed directions using keyword wavenumbers.
Each keyword name must match a FourierTransformed coordinate name.

```julia
# Domain with two FourierTransformed directions:
domain = Domain(x=FourierTransformed(), y=FourierTransformed(), z=Chebyshev(N=30, lower=0, upper=1))
cache = discretize(prob)
A, B = assemble(cache; k_x=1.0, k_y=0.5)
```
"""
function assemble(cache::DiscretizationCache; kwargs...)
    k_vals = _normalize_k_values(cache.domain, kwargs)

    # For single FourierTransformed direction, delegate to the simple version
    transformed = cache.domain.transformed_dims
    if length(transformed) == 1 && length(k_vals) <= 1
        k_name = Symbol(:k_, transformed[1])
        k_val = get(k_vals, k_name, 0.0)
        return assemble(cache, k_val)
    end

    return _assemble(cache, k_vals)
end

"""
    _assembled_density(cache) -> Float64

Estimate the fill fraction (nnz / N²) of the assembled operator from the cached
k-components — an upper bound on the per-wavenumber fill. Any derived cache forces
the dense estimate (`1.0`). Retained as an assembly-banding diagnostic for tests;
the SLEPc solve path does not consume it.
"""
function _assembled_density(cache::DiscretizationCache)
    isempty(cache.derived_caches) || return 1.0
    N = cache.N_total
    N == 0 && return 1.0
    pat = spzeros(ComplexF64, N, N)
    for (_, M) in cache.A_components
        pat += M
    end
    return nnz(pat) / (N * N)
end
