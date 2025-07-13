import Base: show

const SparseMat{T} = SparseMatrixCSC{T, Int}

"""
@inline sparse kron product for memory efficiency.
"""
@inline function kron_s(A, B)
    return sparse(kron(A, B))
end

"""
Container for reusable identity matrices in Kronecker products.
"""
struct OperatorCache{T}
    Iʸ::SparseMat{T}
    Iᶻ::SparseMat{T}
end

function OperatorCache(Ny::Int, Nz::Int, T::Type{<:Real}=Float64)
    Iʸ = sparse(LinearAlgebra.I, Ny, Ny)
    Iᶻ = sparse(LinearAlgebra.I, Nz, Nz)
    return OperatorCache{T}(Iʸ, Iᶻ)
end


"Container for the linear operators used in the problem"
struct Problem{Tg<:AbstractFloat}
    # Fourier in y
    Dʸ     :: SparseMat{Tg}
    D²ʸ    :: SparseMat{Tg}
    D⁴ʸ    :: SparseMat{Tg}

    # Chebyshev in z (Neumann BC)
    Dᶻᴺ    :: SparseMat{Tg}
    D²ᶻᴺ   :: SparseMat{Tg}
    D⁴ᶻᴺ   :: SparseMat{Tg}

    # Chebyshev in z (Dirichlet BC)
    Dᶻᴰ    :: SparseMat{Tg}
    D²ᶻᴰ   :: SparseMat{Tg}
    D⁴ᶻᴰ   :: SparseMat{Tg}

    # Mixed
    Dʸ²ᶻᴰ  :: SparseMat{Tg}
    D²ʸ²ᶻᴰ :: SparseMat{Tg}
end



"""
    struct EmptyParams <: AbstractParams

A placeholder struct for parameters.
"""
struct EmptyParams <: AbstractParams
    "A placeholder for parameters, used when no parameters are needed."
end


function Problem{T<:Real}(params=EmptyParams, grid=AbstractGrid{T}, cache::OperatorCache{T}) where T

    # Kronecker products: Dirichlet
    Dᶻᴰ   = kron_s(cache.Iʸ, grid.Dᶻᴰ )
    D²ᶻᴰ  = kron_s(cache.Iʸ, grid.D²ᶻᴰ)
    D⁴ᶻᴰ  = kron_s(cache.Iʸ, grid.D⁴ᶻᴰ)

    # Kronecker products: Neumann
    Dᶻᴺ   = kron_s(cache.Iʸ, grid.Dᶻᴺ )
    D²ᶻᴺ  = kron_s(cache.Iʸ, grid.D²ᶻᴺ)
    D⁴ᶻᴺ  = kron_s(cache.Iʸ, grid.D⁴ᶻᴺ)

    # Mixed derivatives
    Dʸ²ᶻᴰ   = kron_s(grid.D²ʸ, grid.Dᶻᴰ)
    D²ʸ²ᶻᴰ  = kron_s(grid.D²ʸ, grid.D²ᶻᴰ)

    # For Fourier differentiation matrix
    Dʸ      = kron_s(grid.Dʸ,  cache.Iᶻ)
    D²ʸ     = kron_s(grid.D²ʸ, cache.Iᶻ)
    D⁴ʸ     = kron_s(grid.D⁴ʸ, cache.Iᶻ)

    return Problem{T}(Dʸ, D²ʸ, D⁴ʸ,
                    Dᶻᴰ, D²ᶻᴰ, D⁴ᶻᴰ,
                    Dᶻᴺ, D²ᶻᴺ,
                    Dʸ²ᶻᴰ, D²ʸ²ᶻᴰ)
end


struct TwoDGrid{T<:AbstractFloat, Ty, Tm} <: AbstractGrid{T, Ty, Tm}
    "Number of grid points in the y-direction"  
    Ny::Int
    "Number of grid points in the z-direction"
    Nz::Int
    
    "domain extent in ``y``"
    L :: T
    "domain extent in ``z``"
    H :: T

    "range with ``y``-grid-points"
    y :: Ty
    "range with ``z``-grid-points"
    z :: Ty

    "Fourier differentiation matrices"
    Dʸ  :: Tm
    D²ʸ :: Tm
    D⁴ʸ :: Tm

    "Chebyshev differentiation matrices"
    Dᶻ  :: Tm
    D²ᶻ :: Tm
    D⁴ᶻ :: Tm

    Dᶻᴰ  :: Tm
    D²ᶻᴰ :: Tm
    D⁴ᶻᴰ :: Tm

    Dᶻᴺ  :: Tm
    D²ᶻᴺ :: Tm
end


function TwoDGrid(Ny::Int, L::Real, Nz::Int, H::Real;
                  T::Type{<:Real}=Float64,
                  apply_bcs::Bool=false,
                  params::AbstractParams=EmptyParams())

    @assert Ny > 0 && Nz > 0, "Ny and Nz must be positive integers."

    # setup Fourier differentiation matrices  
    # Fourier in y-direction: y ∈ [0, L)
    y,  Dʸ  = FourierDiff(Ny, 1)
    _,  D²ʸ = FourierDiff(Ny, 2)
    _,  D⁴ʸ = FourierDiff(Ny, 4)

    # Transform the domain and derivative operators from [0, 2π) → [0, L)
    scale = (2π / L)
    y    .= (L / 2π) .* y
    Dʸ   .*= scale
    D²ʸ  .*= scale^2
    D⁴ʸ  .*= scale^4

    # Chebyshev in the z-direction
    z,  Dᶻ  = chebdif(Nz, 1)
    _,  D²ᶻ = chebdif(Nz, 2)
    _,  D³ᶻ = chebdif(Nz, 3)
    _,  D⁴ᶻ = chebdif(Nz, 4)

    # Transform the domain and derivative operators from [-1, 1] → [0, H]
    z, Dᶻ, D²ᶻ  = chebder_transform(z,  Dᶻ, 
                                        D²ᶻ, 
                                        zerotoL_transform, 
                                        H)

    _, _, D⁴ᶻ  = chebder_transform_ho(z, Dᶻ, 
                                        D²ᶻ, 
                                        D³ᶻ, 
                                        D⁴ᶻ, 
                                        zerotoL_transform_ho, 
                                        H)

    #### Create the grid object
        grid = TwoDGrid{T, typeof(y), typeof(Dʸ)}(
            Ny, Nz, L, H, y, z,
            Dʸ, D²ʸ, D⁴ʸ,
            Dᶻ, D²ᶻ, D⁴ᶻ,
            Dᶻᴰ, D²ᶻᴰ, D⁴ᶻᴰ,
            Dᶻᴺ, D²ᶻᴺ
        )

        #### Apply BCs if requested
        if apply_bcs
            setBCs!(grid, params, :dirichlet)
            setBCs!(grid, params, :neumann)
        end

    return grid
end


function show(io::IO, p::Params{T}) where T
    print(io,
        "Eigen Solver Configuration \n",
        "  ├────────────────────── Float Type: $T \n",
        "  ├─────────────── Domain Size (L, H): ", (p.L, p.H), "\n",
        "  ├───────────── Resolution (Ny, Nz): ", (p.Ny, p.Nz), "\n",
        "  ├──── Boundary Conditions (w, ζ, b): ", (p.w_bc, p.ζ_bc, p.b_bc), "\n",
        "  └────────────── Eigenvalue Solver: ", p.eig_solver, "\n"
    )
end

# function show(io::IO, p::Params{T}) where T
#     print(io, """
# Eigen Solver Configuration
#   ┌────────────────────────────────────────────
#   │ Float Type                 : $T
#   │ Domain Size (L × H)        : ($(p.L), $(p.H))
#   │ Resolution (Ny × Nz)       : ($(p.Ny), $(p.Nz))
#   │ Boundary Conditions (w, ζ, b): ($(p.w_bc), $(p.ζ_bc), $(p.b_bc))
#   │ Eigenvalue Solver          : $(p.eig_solver)
#   └────────────────────────────────────────────
# """)
# end

