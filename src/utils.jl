using LinearAlgebra
using Printf
using Dierckx #: Spline1D, derivative, evaluate
#using BasicInterpolators: BicubicInterpolator

# # print the eigenvalues
# function print_evals(λs, n)
#     @printf "%i largest eigenvalues: \n" n
#     for p in n:-1:1
#         if imag(λs[p]) >= 0
#             @printf "%i: %1.4e+%1.4eim\n" p real(λs[p]) imag(λs[p])
#         end
#         if imag(λs[p]) < 0
#             @printf "%i: %1.4e%1.4eim\n" p real(λs[p]) imag(λs[p])
#         end
#     end
# end

# Custom struct for nicely formatted eigenvalue output
struct EigenvalueDisplay
    λ::Complex
    idx::Int
end

# Custom display for EigenvalueDisplay
function Base.show(io::IO, ::MIME"text/plain", ev::EigenvalueDisplay)
    r, i = real(ev.λ), imag(ev.λ)
    if iszero(i)
        @printf(io, "%3d │ % .6e          ", ev.idx, r)
    else
        sign_str = i ≥ 0 ? "+" : ""
        @printf(io, "%3d │ % .6e %s%.6eim", ev.idx, r, sign_str, i)
    end
end

"""
    print_evals(λs::AbstractVector{<:Number}; n::Int=length(λs), by::Symbol=:abs)

Pretty-print the top `n` eigenvalues from the list `λs`, optionally sorted by `:real`, `:imag`, or `:abs`.
"""
function print_evals(λs::Vector{<:Complex})
    n = length(λs) 
    λs_sorted = λs #sort(λs, by=abs, rev=true)
    println("Top $n eigenvalues (sorted):")
    println("Idx │ Real Part     Imag Part")
    println("────┼──────────────────────────────")
    for p in n:-1:1
        show(stdout, "text/plain", EigenvalueDisplay(λs_sorted[p], p))
        println()  # ensure newline after each
    end
end


"""
    λs_sorted, χ_sorted = sort_evals(λs, χ, "R"; sorting="lm")
    Sort the eigenvalues `λs` and corresponding eigenvectors `χ` based on the specified criterion 
    (`"M"` for magnitude, `"I"` for imaginary part, or `"R"` for real part).
    The `sorting` argument determines the order: `"lm"` for descending order.
"""
function sort_evals(λs::AbstractVector, χ::AbstractMatrix, which::String; sorting::String="lm")
    @assert which in ("M", "I", "R")
    by_func = Dict(
        "M" => abs,
        "I" => imag,
        "R" => real
    )[which]

    idx = sortperm(λs, by=by_func, rev=(sorting == "lm"))
    return λs[idx], χ[:, idx]
end



"""
    sort_evals_(λ, Χ, by::Symbol; rev::Bool)

Sort eigenvalues `λ` and corresponding eigenvectors `Χ` by:
- `:R` → real part
- `:I` → imaginary part
- `:M` → magnitude (abs)

Set `rev=true` for descending (default), `false` for ascending.
"""
function sort_evals_(λ::Vector, Χ::Matrix, by::Symbol; rev::Bool=true)
    sortfun = by == :R ? real : by == :I ? imag : abs
    idx = sortperm(λ, by=sortfun, rev=rev)
    return λ[idx], Χ[:, idx]
end


function remove_evals(λs, χ, lower, higher, which)
    
    @assert which ∈ ["M", "I", "R"]

    if which == "I" # imaginary part
        arg = findall( (lower .≤ imag(λs)) .& (imag(λs) .≤ higher) )
    end

    if which == "R" # real part
        arg = findall( (lower .≤ real(λs)) .& (real(λs) .≤ higher) )
    end
    
    if which == "M" # absolute magnitude
        arg = findall( (lower .≤ abs.(λs)) .& (abs.(λs) .≤ higher) )
    end
    
    χ  = χ[:,arg]
    λs = λs[arg]

    return λs, χ

    #return nothing
end

# function remove_spurious(λₛ, X)
#     #p = findall(x->x>=abs(item), abs.(real(λₛ)))  
#     deleteat!(λₛ, 1)
#     X₁ = X[:, setdiff(1:end, 1)]
#     return λₛ, X₁
# end

function inverse_Lap_hor(∇ₕ²)
    Qm, Rm = qr(Matrix(∇ₕ²))
    Qm     = sparse(Qm)
    Qᵀ     = transpose(Qm)
    H      = Rm \ Matrix(Qᵀ)
    return H
end

# function inverse_Lap_hor(∇ₕ²::SparseMatrixCSC)
#     F = qr(∇ₕ²)
#     Qm = Matrix(F.Q)
#     H  = F.R \ transpose(Qm)
#     return sparse(H)
# end

struct InverseLaplace{T}
    Qᵀ::SparseMatrixCSC{T,Int}
    R::Matrix{T}
end

function InverseLaplace(∇ₕ²::AbstractMatrix{T}) where T
    F    = qr(Matrix(∇ₕ²))
    Q    = sparse(Matrix(F.Q))  # force sparse Q
    return InverseLaplace{T}(Q', Matrix(F.R))
end

"""
    H = InverseLaplace(∇ₕ²::AbstractMatrix{T}) where T<:AbstractFloat

    # Suppose ∇ₕ² is your horizontal Laplacian matrix
    ∇ₕ² = your_matrix_here
    H = InverseLaplace(∇ₕ²)

    # Apply the inverse to a vector x
    x = rand(size(∇ₕ², 1))
    u = H(x)  # equivalent to H * x
    """
@inline function (H::InverseLaplace)(x::AbstractVector{T}) where T
    return H.R \ (H.Qᵀ * x)
end


# function inverse_Lap_hor(∇ₕ²::AbstractMatrix{T}) where T<:AbstractFloat
#     F = qr(∇ₕ²)                     # QR factorization
#     Q = Matrix(F.Q)                 # full Q (dense, orthogonal)
#     R = F.R                         # upper triangular
#     return inv(R) * Q'              # A⁻¹ = R⁻¹ * Qᵀ
# end

function ∇f(f::AbstractVector{T}, x::AbstractVector{T}) where T<:AbstractFloat
    @assert length(f) == length(x)
    @assert length(x) ≥ 9 "∇f requires at least 9 points (uses 8th-order central stencil)"
    dx = x[2:end] .- x[1:end-1]
    @assert std(dx) ≤ 1.0e-6 "x must be uniformly spaced"
    Δx = dx[1]
    N = length(x)

    ∂f_∂x = Array{T}(undef, N)

    c₄₊ = (-25//12, 4, -3, 4//3, -1//4)
    c₄₋ = reverse(tuple((-c for c in c₄₊)...))
    c₈  = (1//280, -4//105, 1//5, -4//5, 0.0, 4//5, -1//5, 4//105, -1//280)

    @inbounds for k in 1:4
        ∂f_∂x[k] = c₄₊[1]*f[k] + c₄₊[2]*f[k+1] + c₄₊[3]*f[k+2] + c₄₊[4]*f[k+3] + c₄₊[5]*f[k+4]
    end

    @inbounds for k in 5:N-4
        ∂f_∂x[k]  = c₈[1]*f[k-4] + c₈[2]*f[k-3] + c₈[3]*f[k-2] + c₈[4]*f[k-1]
        ∂f_∂x[k] += c₈[5]*f[k]
        ∂f_∂x[k] += c₈[6]*f[k+1] + c₈[7]*f[k+2] + c₈[8]*f[k+3] + c₈[9]*f[k+4]
    end

    @inbounds for k in N-3:N
        ∂f_∂x[k] = c₄₋[1]*f[k] + c₄₋[2]*f[k-1] + c₄₋[3]*f[k-2] + c₄₋[4]*f[k-3] + c₄₋[5]*f[k-4]
    end

    return ∂f_∂x ./ Δx
end

# using SparseArrays, LinearAlgebra
# export field_to_spdiagm, spdiag_to_field, diagm

# ────────────────────────────────────────────────────────────────────────────
#  Original helpers (unchanged)                                              │
# ────────────────────────────────────────────────────────────────────────────
"""
    field_to_spdiagm(U; k=0, order=:col, dims=nothing, scale=identity, pad=:error)

See previous message for full docstring.
"""
function field_to_spdiagm(U::AbstractMatrix;
                          k::Integer = 0,
                          order::Symbol = :col,
                          dims::Union{Nothing,Tuple{Int,Int}} = nothing,
                          scale = identity,
                          pad::Symbol = :error)

    v = order === :row ? reshape(U', :) : vec(U)
    v = scale.(v)
    m, n = isnothing(dims) ? (length(v), length(v)) : dims
    diag_len = min(m, n) - abs(k)
    diag_len > 0 || error("Diagonal k=$k does not fit into $m×$n matrix")

    # fit vector to diagonal length
    if length(v) < diag_len
        v = vcat(v, zeros(eltype(v), diag_len - length(v)))
    elseif length(v) > diag_len
        pad === :trim  && (v = v[1:diag_len])
        pad === :zero  && (v = v[1:diag_len] .= v[diag_len+1:diag_len])
        pad === :wrap  && (v = v[mod1.(1:diag_len, length(v))])
        pad === :error && error("Vector length $(length(v)) exceeds diagonal length $diag_len")
    end

    return spdiagm(m, n, k => v)
end

"""
    spdiag_to_field(S, m, n; k=0, order=:col)

Inverse operation of `field_to_spdiagm`.
"""
function spdiag_to_field(S::SparseMatrixCSC, m::Int, n::Int;
                         k::Integer = 0, order::Symbol = :col)
    d = diag(S, k)
    length(d) < m*n && (d = vcat(d, zeros(eltype(d), m*n - length(d))))
    return order === :row ? reshape(d, n, m)' : reshape(d, m, n)
end


"""
    DiagM(U::AbstractMatrix; k=0, order=:col, sparse=true,
          dims=nothing, scale=identity, pad=:error)

*If the first argument is a 2‑D array*, build a diagonal matrix from it.
Keyword interface is identical to `field_to_spdiagm`.

* `sparse=true`  → returns `SparseMatrixCSC` (uses `spdiagm`).  
* `sparse=false` → returns dense `Matrix`, allocating zeros for the rest.
"""
function DiagM(U::AbstractMatrix;
               k::Integer = 0,
               order::Symbol = :col,
               sparse::Bool = true,
               dims::Union{Nothing,Tuple{Int,Int}} = nothing,
               scale = identity,
               pad::Symbol = :error)

    if sparse
        return field_to_spdiagm(U; k=k, order=order, dims=dims,
                                   scale=scale, pad=pad)
    else
        # build dense version by filling zeros matrix then dropping diagonal in
        m, n = isnothing(dims) ? (length(vec(U)), length(vec(U))) : dims
        S = zeros(eltype(U), m, n)
        v = order === :row ? reshape(U', :) : vec(U)
        v = scale.(v)
        diag_len = min(m, n) - abs(k)
        v = length(v) ≥ diag_len ? v[1:diag_len] : vcat(v, zeros(eltype(v), diag_len - length(v)))

        # place on the k‑th diagonal
        r0, c0 = k ≥ 0 ? (1, 1+k) : (1-k, 1)
        @inbounds for i = 1:diag_len
            S[r0+i-1, c0+i-1] = v[i]
        end
        return S
    end
end

