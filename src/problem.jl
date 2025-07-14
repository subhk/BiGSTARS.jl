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
struct OperatorI{T}
    Iʸ::SparseMat{T}
    Iᶻ::SparseMat{T}
end

function OperatorI(params::AbstractParams, T::Type{<:Real}=Float64)
    Iʸ = sparse(LinearAlgebra.I, params.Ny, params.Ny)
    Iᶻ = sparse(LinearAlgebra.I, params.Nz, params.Nz)
    return OperatorI{T}(Iʸ, Iᶻ)
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

    # Chebyshev in z (Dirichlet BC)
    Dᶻᴰ    :: SparseMat{Tg}
    D²ᶻᴰ   :: SparseMat{Tg}
    D⁴ᶻᴰ   :: SparseMat{Tg}

    # Mixed
    Dʸᶻᴰ   :: SparseMat{Tg}
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


function Problem(grid::AbstractGrid{T, Ty, Tm}, 
                cache::OperatorI{T}, 
                params::AbstractParams) where {T<:Real, Ty<:AbstractVector, Tm<:AbstractMatrix}

    # Kronecker products: Dirichlet
    Dᶻᴰ     = kron_s(cache.Iʸ, Matrix(grid.Dᶻᴰ) )
    D²ᶻᴰ    = kron_s(cache.Iʸ, Matrix(grid.D²ᶻᴰ))
    D⁴ᶻᴰ    = kron_s(cache.Iʸ, Matrix(grid.D⁴ᶻᴰ))

    # Kronecker products: Neumann
    Dᶻᴺ     = kron_s(cache.Iʸ, Matrix(grid.Dᶻᴺ) )
    D²ᶻᴺ    = kron_s(cache.Iʸ, Matrix(grid.D²ᶻᴺ))

    # Mixed derivatives
    Dʸᶻᴰ    = kron_s(Matrix(grid.Dʸ ),  Matrix(grid.Dᶻᴰ ))
    Dʸ²ᶻᴰ   = kron_s(Matrix(grid.Dʸ ),  Matrix(grid.D²ᶻᴰ))
    D²ʸ²ᶻᴰ  = kron_s(Matrix(grid.D²ʸ),  Matrix(grid.D²ᶻᴰ))

    # For Fourier differentiation matrix
    Dʸ      = kron_s(Matrix(grid.Dʸ ), cache.Iᶻ)
    D²ʸ     = kron_s(Matrix(grid.D²ʸ), cache.Iᶻ)
    D⁴ʸ     = kron_s(Matrix(grid.D⁴ʸ), cache.Iᶻ)

    #### Create the grid object
    prob = Problem{T}(Dʸ, D²ʸ, D⁴ʸ,
                    Dᶻᴺ, D²ᶻᴺ,
                    Dᶻᴰ, D²ᶻᴰ, D⁴ᶻᴰ,
                    Dʸᶻᴰ, Dʸ²ᶻᴰ, D²ʸ²ᶻᴰ
    )

    return prob
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

    # Fourier differentiation matrices (sparse)
    Dʸ  :: SparseMatrixCSC{T, Int}
    D²ʸ :: SparseMatrixCSC{T, Int}
    D⁴ʸ :: SparseMatrixCSC{T, Int}

    # Chebyshev matrices (sparse)
    Dᶻ  :: SparseMatrixCSC{T, Int}
    D²ᶻ :: SparseMatrixCSC{T, Int}
    D⁴ᶻ :: SparseMatrixCSC{T, Int}

    Dᶻᴰ  :: SparseMatrixCSC{T, Int}
    D²ᶻᴰ :: SparseMatrixCSC{T, Int}
    D⁴ᶻᴰ :: SparseMatrixCSC{T, Int}

    Dᶻᴺ  :: SparseMatrixCSC{T, Int}
    D²ᶻᴺ :: SparseMatrixCSC{T, Int}
end


function TwoDGrid(params::AbstractParams) 

    Ny = params.Ny
    Nz = params.Nz
    L  = params.L
    H  = params.H

    # setup Fourier differentiation matrices  
    # Fourier in y-direction: y ∈ [0, L)
    y1, Dʸ  = FourierDiff(Ny, 1)
    _,  D²ʸ = FourierDiff(Ny, 2)
    _,  D⁴ʸ = FourierDiff(Ny, 4)

    # Transform the domain and derivative operators from [0, 2π) → [0, L)
    y   = L/2π * y1
    Dʸ  = (2π/L)^1 * Dʸ
    D²ʸ = (2π/L)^2 * D²ʸ
    D⁴ʸ = (2π/L)^4 * D⁴ʸ

    # Chebyshev in the z-direction
    z1, D1z = chebdif(Nz, 1)
    _,  D2z = chebdif(Nz, 2)
    _,  D3z = chebdif(Nz, 3)
    _,  D4z = chebdif(Nz, 4)

    # Transform the domain and derivative operators from [-1, 1] → [0, H]
    z, Dᶻ, D²ᶻ  = chebder_transform(z1, D1z, 
                                        D2z, 
                                        zerotoL_transform, 
                                        H)

    _, _, D⁴ᶻ  = chebder_transform_ho(z1, D1z, 
                                        D2z, 
                                        D3z, 
                                        D4z, 
                                        zerotoL_transform_ho, 
                                        H)
    T = eltype(Dʸ)

   # Convert to mutable matrices to allow BCs
    Dᶻᴰ  = Matrix(deepcopy(Dᶻ) )
    D²ᶻᴰ = Matrix(deepcopy(D²ᶻ))
    D⁴ᶻᴰ = Matrix(deepcopy(D⁴ᶻ))

    Dᶻᴺ  = Matrix(deepcopy(Dᶻ) )
    D²ᶻᴺ = Matrix(deepcopy(D²ᶻ))

    #### Create the grid object
    grid = TwoDGrid{T, typeof(y), typeof(Dʸ)}(
        Ny, Nz, L, H, y, z,
        Dʸ, D²ʸ, D⁴ʸ,
        Dᶻ, D²ᶻ, D⁴ᶻ,
        Dᶻᴰ, D²ᶻᴰ, D⁴ᶻᴰ,
        Dᶻᴺ, D²ᶻᴺ
    )

    #### Apply BCs 
    setBCs!(grid, params, :dirichlet)
    setBCs!(grid, params, :neumann)

    return grid
end



function show(io::IO, params::AbstractParams)
    T = typeof(params.L)  # infer float type from a field
    print(io,
        "Eigen Solver Configuration \n",
        "  ├────────────────────── Float Type: $T \n",
        "  ├─────────────── Domain Size (L, H): ", (params.L, params.H), "\n",
        "  ├───────────── Resolution (Ny, Nz): ", (params.Ny, params.Nz), "\n",
        "  ├──── Boundary Conditions (w, ζ, b): ", (params.w_bc, params.ζ_bc, params.b_bc), "\n",
        "  └────────────── Eigenvalue Solver: ", params.eig_solver, "\n"
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

