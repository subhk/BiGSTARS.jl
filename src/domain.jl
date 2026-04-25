#══════════════════════════════════════════════════════════════════════════════#
#  Domain: coordinate system and spectral operator construction               #
#══════════════════════════════════════════════════════════════════════════════#

# ──────────────────────────────────────────────────────────────────────────────
#  Coordinate specification types
# ──────────────────────────────────────────────────────────────────────────────

"""Marker type for a Fourier-transformed (wavenumber) direction."""
struct FourierTransformed end

"""Specification for a resolved Fourier direction."""
struct FourierBasisSpec
    N::Int
    L::Float64
    lower::Float64
    upper::Float64
end

"""Specification for a resolved Chebyshev direction."""
struct ChebyshevBasisSpec
    N::Int
    lower::Float64
    upper::Float64
end

# User-facing constructors — multiple calling conventions:
#   Fourier(N=64, L=2pi)          keyword (L = domain length)
#   Fourier(N=64, [0, 2pi])       positional domain
#   Chebyshev(N=30, [0, 1])       positional domain
#   Chebyshev(N=30, lower=0, upper=1)  keyword (backward compat)

Fourier(; N::Int, L::Float64) = FourierBasisSpec(N, L, 0.0, L)
Fourier(N::Int, domain::AbstractVector) = FourierBasisSpec(N, domain[2] - domain[1], Float64(domain[1]), Float64(domain[2]))
Chebyshev(; N::Int, lower::Float64, upper::Float64) = ChebyshevBasisSpec(N, lower, upper)
Chebyshev(N::Int, domain::AbstractVector) = ChebyshevBasisSpec(N, Float64(domain[1]), Float64(domain[2]))

const CoordSpec = Union{FourierTransformed, FourierBasisSpec, ChebyshevBasisSpec}

# ──────────────────────────────────────────────────────────────────────────────
#  Domain type
# ──────────────────────────────────────────────────────────────────────────────

"""
    Domain(; x=FourierTransformed(), y=Fourier(N=60, L=1.0), z=Chebyshev(N=30, lower=0, upper=1))

A multi-dimensional domain with named coordinates. Each coordinate is one of:
- `FourierTransformed()` — wavenumber parameter direction
- `Fourier(N=..., L=...)` — resolved periodic direction
- `Chebyshev(N=..., lower=..., upper=...)` — resolved non-periodic direction
"""
struct Domain
    coords::Dict{Symbol, CoordSpec}
    coord_order::Vector{Symbol}
    resolved_dims::Vector{Symbol}
    transformed_dims::Vector{Symbol}

    function Domain(; kwargs...)
        coords = Dict{Symbol, CoordSpec}()
        coord_order = Symbol[]
        resolved = Symbol[]
        transformed = Symbol[]

        for (name, spec) in kwargs
            coords[name] = spec
            push!(coord_order, name)
            if spec isa FourierTransformed
                push!(transformed, name)
            else
                push!(resolved, name)
            end
        end

        new(coords, coord_order, resolved, transformed)
    end
end

function Base.show(io::IO, d::Domain)
    print(io, "Domain(")
    for (i, name) in enumerate(d.coord_order)
        spec = d.coords[name]
        if spec isa FourierTransformed
            print(io, "$name=FourierTransformed()")
        elseif spec isa FourierBasisSpec
            print(io, "$name=Fourier(N=$(spec.N), [$(spec.lower),$(spec.upper)])")
        elseif spec isa ChebyshevBasisSpec
            print(io, "$name=Chebyshev(N=$(spec.N), [$(spec.lower),$(spec.upper)])")
        end
        i < length(d.coord_order) && print(io, ", ")
    end
    print(io, ")")
end

# ──────────────────────────────────────────────────────────────────────────────
#  Grid points
# ──────────────────────────────────────────────────────────────────────────────

"""
    gridpoints(domain, dims...) -> Vector or Tuple

Return physical-space grid points for the requested resolved dimensions.
Returns a single vector for one dim, a tuple for multiple.
"""
function _gridpoints_one(domain::Domain, dim::Symbol)
    spec = domain.coords[dim]
    if spec isa FourierBasisSpec
        return fourier_points(spec.N, spec.L) .+ spec.lower
    elseif spec isa ChebyshevBasisSpec
        return chebyshev_points(spec.N, spec.lower, spec.upper)
    else
        error("Cannot get grid points for FourierTransformed direction :$dim")
    end
end

function gridpoints(domain::Domain, dim::Symbol)
    return _gridpoints_one(domain, dim)
end

function gridpoints(domain::Domain, dims::Symbol...)
    return map(dim -> _gridpoints_one(domain, dim), dims)
end

"""
    meshgrid(domain, dim1, dim2) -> (Matrix, Matrix)

Return 2D meshgrid arrays for two resolved dimensions.
Convention: dim2 (typically z) varies along rows, dim1 (typically y) along columns.
Result matrices are size (N_dim2, N_dim1) — z-fastest, matching state vector ordering.

```julia
Y, Z = meshgrid(domain, :y, :z)
U = @. Z - 0.5 + 0.1 * sin(2π * Y)  # 2D field, N_z × N_y
prob[:U] = vec(U)  # flatten for DSL
```
"""
function meshgrid(domain::Domain, dim1::Symbol, dim2::Symbol)
    pts1 = gridpoints(domain, dim1)
    pts2 = gridpoints(domain, dim2)
    Y = [p1 for p2 in pts2, p1 in pts1]  # N2 × N1
    Z = [p2 for p2 in pts2, p1 in pts1]  # N2 × N1
    return Y, Z
end

# ──────────────────────────────────────────────────────────────────────────────
#  Spectral operators from domain
# ──────────────────────────────────────────────────────────────────────────────

"""
    get_diff_operator(domain, dim, order) -> SparseMatrixCSC

Get the 1D spectral differentiation operator for a resolved dimension.
Includes domain scaling for Chebyshev directions.
"""
function get_diff_operator(domain::Domain, dim::Symbol, order::Int)
    @assert order >= 0 "Derivative order must be non-negative"
    spec = domain.coords[dim]
    if spec isa FourierBasisSpec
        return fourier_diff_operator(spec.N, spec.L, order)
    elseif spec isa ChebyshevBasisSpec
        N = spec.N
        order == 0 && return sparse(1.0I, N, N)
        scale = 2.0 / (spec.upper - spec.lower)
        # Chain: D_{order-1} * ... * D_1 * D_0, each scaled
        D = scale * differentiation_operator(0, N)
        for p in 1:order-1
            D = (scale * differentiation_operator(p, N)) * D
        end
        return D
    else
        error("Cannot get differentiation operator for FourierTransformed direction :$dim")
    end
end

"""
    get_conversion_operator(domain, dim, from_order, to_order) -> SparseMatrixCSC

Get the conversion operator chain S that converts from C^(from_order) to C^(to_order).
Only applicable to Chebyshev directions.
"""
function get_conversion_operator(domain::Domain, dim::Symbol, from_order::Int, to_order::Int)
    spec = domain.coords[dim]
    spec isa ChebyshevBasisSpec || error("Conversion operators only for Chebyshev directions")
    N = spec.N
    if from_order >= to_order
        return sparse(1.0I, N, N)
    end
    S = conversion_operator(from_order, N)
    for p in from_order+1:to_order-1
        S = conversion_operator(p, N) * S
    end
    return S
end

"""
    total_grid_size(domain) -> Int

Compute the total number of grid points across all resolved dimensions.
"""
function total_grid_size(domain::Domain)
    n = 1
    for dim in domain.resolved_dims
        spec = domain.coords[dim]
        if spec isa FourierBasisSpec
            n *= spec.N
        elseif spec isa ChebyshevBasisSpec
            n *= spec.N
        end
    end
    return n
end

"""
    get_N(domain, dim) -> Int

Get the number of grid points for a resolved dimension.
"""
function get_N(domain::Domain, dim::Symbol)
    spec = domain.coords[dim]
    if spec isa FourierBasisSpec
        return spec.N
    elseif spec isa ChebyshevBasisSpec
        return spec.N
    else
        error("FourierTransformed direction :$dim has no grid points")
    end
end
