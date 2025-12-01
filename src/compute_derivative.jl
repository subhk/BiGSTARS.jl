# ────────────────────────────────────────────────────────────────────────────────
# Legacy alias map (ASCII → Unicode / new names)
# ────────────────────────────────────────────────────────────────────────────────

const _ALIASES = Dict{Symbol,Symbol}(
    :∂ʸU₀   => :∂ʸU,     :∂ʸB₀   => :∂ʸB,
    :∂ʸʸU₀  => :∂ʸʸU,    :∂ʸʸB₀  => :∂ʸʸB,
    :∂ᶻU₀   => :∂ᶻU,     :∂ᶻB₀   => :∂ᶻB,
    :∂ᶻᶻU₀ => :∂ᶻᶻU,   :∂ᶻᶻB₀ => :∂ᶻᶻB,
    :∂ʸᶻU₀ => :∂ʸᶻU,    :∂ʸᶻB₀ => :∂ʸᶻB,
)

# ────────────────────────────────────────────────────────────────────────────────
# Lazy container (stores original fields + derivative cache)
# ────────────────────────────────────────────────────────────────────────────────

"""
    Derivatives{T}(U, B, y; Dᶻ, D²ᶻ, gridtype = :Mixed)

A **lazy** wrapper that owns the background fields `U` and `B` and computes
first‑/second‑order derivatives on first access.  The results are cached in a
`Dict` so repeated property access is free.

Supported `gridtype` values
- `:Fourier`   – only y‑direction (Fourier) derivatives are exposed
- `:Chebyshev` – only z‑direction (Chebyshev) derivatives are exposed
- `:Mixed`     – Fourier × Chebyshev grid (default)
- `:All`       – expose every derivative regardless of cost (still lazy)
"""
mutable struct Derivatives{T}
    U        :: AbstractMatrix{T}
    B        :: AbstractMatrix{T}
    y        :: AbstractVector{T}
    Dᶻ       :: Union{AbstractMatrix{T}, Nothing}
    D²ᶻ      :: Union{AbstractMatrix{T}, Nothing}
    gridtype :: Symbol
    cache    :: Dict{Symbol,Any}
end

function Derivatives(U::AbstractMatrix{T}, B::AbstractMatrix{T}, y::AbstractVector{T};
                     Dᶻ::Union{AbstractMatrix{T}, Nothing}=nothing,
                     D²ᶻ::Union{AbstractMatrix{T}, Nothing}=nothing,
                     gridtype::Symbol=:Mixed) where {T}

    if gridtype in (:Chebyshev, :Mixed, :All)
        @assert Dᶻ  !== nothing "Chebyshev derivatives requested but `Dᶻ` is missing"
        @assert D²ᶻ !== nothing "Chebyshev derivatives requested but `D²ᶻ` is missing"
    end

    Derivatives{T}(U, B, y, Dᶻ, D²ᶻ, gridtype, Dict{Symbol,Any}())
end

# ────────────────────────────────────────────────────────────────────────────────
# User‑facing constructors
# ────────────────────────────────────────────────────────────────────────────────

compute_derivatives(U::AbstractMatrix{T}, 
                    B::AbstractMatrix{T}, 
                    y::AbstractVector{T};
                    Dᶻ::Union{AbstractMatrix{T}, Nothing}=nothing,
                    D²ᶻ::Union{AbstractMatrix{T}, Nothing}=nothing,
                    gridtype::Symbol=:Mixed) where {T} =
    Derivatives(U, B, y; Dᶻ=Dᶻ, D²ᶻ=D²ᶻ, gridtype=gridtype)

# Legacy tuple‑return API for older BiGSTARS code
function compute_derivatives_legacy(U::AbstractMatrix{T}, 
                                    B::AbstractMatrix{T}, 
                                    y::AbstractVector{T};
                                    Dᶻ::Union{AbstractMatrix{T}, Nothing}=nothing,
                                    D²ᶻ::Union{AbstractMatrix{T}, Nothing}=nothing,
                                    gridtype::Symbol=:All) where {T}
    D = compute_derivatives(U, B, y; Dᶻ=Dᶻ, D²ᶻ=D²ᶻ, gridtype=gridtype)
    return D.∂ʸU, D.∂ʸB, D.∂ᶻU, D.∂ᶻB, D.∂ʸʸU, D.∂ᶻᶻU, D.∂ʸᶻU
end

# ────────────────────────────────────────────────────────────────────────────────
# Helper kernels (BLAS‑friendly eager computations)
# ────────────────────────────────────────────────────────────────────────────────

_fourier(U, B, y, order) = begin
    dim = (size(U,1) == length(y)) ? 1 : 2
    (gradient(U, y; dims=dim, order=order),
     gradient(B, y; dims=dim, order=order))
end

_first_fourier  = (U,B,y) -> _fourier(U,B,y,1)
_second_fourier = (U,B,y) -> _fourier(U,B,y,2)

function _first_cheb(U, B, y, Dᶻ)
    if size(U,1) == length(y)
        return U * Dᶻ, B * Dᶻ  # multiply on right (faster contiguous rows)
    else
        return Dᶻ * U, Dᶻ * B
    end
end

function _second_cheb(U, B, y, D²ᶻ)
    if size(U,1) == length(y)
        return U * D²ᶻ, B * D²ᶻ
    else
        return D²ᶻ * U, D²ᶻ * B
    end
end

_cross(FyU, FyB, y, Dᶻ) = size(FyU,1) == length(y) ? (FyU * Dᶻ, FyB * Dᶻ) : (Dᶻ * FyU, Dᶻ * FyB)

# ────────────────────────────────────────────────────────────────────────────────
# Lazy property access with caching and alias support
# ────────────────────────────────────────────────────────────────────────────────

function Base.getproperty(D::Derivatives, s::Symbol)
    # map legacy alias → canonical symbol
    s = get(_ALIASES, s, s)

    # direct fields
    if s in (:U, :B, :y, :Dᶻ, :D²ᶻ, :gridtype, :cache)
        return getfield(D, s)
    end

    # cached?
    if haskey(D.cache, s)
        return D.cache[s]
    end

    # compute required block lazily
    if s in (:∂ʸU, :∂ʸB) && D.gridtype in (:Fourier, :Mixed, :All)
        if !haskey(D.cache, :∂ʸU)
            D.cache[:∂ʸU], D.cache[:∂ʸB] = _first_fourier(D.U, D.B, D.y)
        end

    elseif s in (:∂ʸʸU, :∂ʸʸB) && D.gridtype in (:Fourier, :Mixed, :All)
        if !haskey(D.cache, :∂ʸʸU)
            D.cache[:∂ʸʸU], D.cache[:∂ʸʸB] = _second_fourier(D.U, D.B, D.y)
        end

    elseif s in (:∂ᶻU, :∂ᶻB) && D.gridtype in (:Chebyshev, :Mixed, :All)
        if !haskey(D.cache, :∂ᶻU)
            D.cache[:∂ᶻU], D.cache[:∂ᶻB] = _first_cheb(D.U, D.B, D.y, D.Dᶻ)
        end

    elseif s in (:∂ᶻᶻU, :∂ᶻᶻB) && D.gridtype in (:Chebyshev, :Mixed, :All)
        if !haskey(D.cache, :∂ᶻᶻU)
            D.cache[:∂ᶻᶻU], D.cache[:∂ᶻᶻB] = _second_cheb(D.U, D.B, D.y, D.D²ᶻ)
        end

    elseif s in (:∂ʸᶻU, :∂ʸᶻB) && D.gridtype in (:Mixed, :All)
        _ = getproperty(D, :∂ʸU)  # ensure Fourier block ready
        D.cache[:∂ʸᶻU], D.cache[:∂ʸᶻB] = _cross(D.cache[:∂ʸU], D.cache[:∂ʸB], D.y, D.Dᶻ)

    else
        error("Property `$s` not available for gridtype $(D.gridtype)")
    end

    return D.cache[s]
end

# ────────────────────────────────────────────────────────────────────────────────
# Eager helper – compute selected derivative families now
# ────────────────────────────────────────────────────────────────────────────────

"""
    precompute!(D; which = :All) -> Derivatives

Force computation of derivative blocks and return the same `Derivatives`
object.  Valid `which` keys:
* `:Fourier`   – y‑direction (∂ʸ*, ∂ʸʸ*)
* `:Chebyshev` – z‑direction (∂ᶻ*, ∂ᶻᶻ*)
* `:Cross`     – cross derivative (∂ʸᶻ*) – requires mixed grid
* `:All`       – every family applicable to `D.gridtype`

The function is idempotent: requesting the same block twice does no extra work.
"""
function precompute!(D::Derivatives; which::Symbol = :All)
    which ∈ (:Fourier, :Chebyshev, :Cross, :All) ||
        error("Invalid value for `which`: $which")

    if which in (:Fourier, :All) && D.gridtype in (:Fourier, :Mixed, :All)
        _ = D.∂ʸU    # first derivative
        _ = D.∂ʸʸU   # second derivative
    end

    if which in (:Chebyshev, :All) && D.gridtype in (:Chebyshev, :Mixed, :All)
        _ = D.∂ᶻU    # first derivative
        _ = D.∂ᶻᶻU   # second derivative
    end

    if which in (:Cross, :All) && D.gridtype in (:Mixed, :All)
        _ = D.∂ʸᶻU  # triggers cross derivative
    end

    return D
end
