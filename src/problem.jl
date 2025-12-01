import Base: show

const SparseMat{T} = SparseMatrixCSC{T, Int}

"""
@inline sparse kron product for memory efficiency.
"""
@inline function kron_s(A, B)
    # kron(sparse, sparse) already returns a sparse matrix; avoid re-wrapping
    if issparse(A) && issparse(B)
        return kron(A, B)
    end
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

    # Chebychev with any BC
    Dᶻ     :: SparseMat{Tg}
    D²ᶻ    :: SparseMat{Tg}
    D⁴ᶻ    :: SparseMat{Tg}

    # Chebyshev in z (Neumann BC)
    Dᶻᴺ    :: SparseMat{Tg}
    D²ᶻᴺ   :: SparseMat{Tg}
    D⁴ᶻᴺ   :: SparseMat{Tg}

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
                cache::OperatorI{T}) where {T<:Real, Ty<:AbstractVector, Tm<:AbstractMatrix}

    # For Fourier differentiation matrix
    Dʸ      = kron_s(grid.Dʸ,  cache.Iᶻ)
    D²ʸ     = kron_s(grid.D²ʸ, cache.Iᶻ)
    D⁴ʸ     = kron_s(grid.D⁴ʸ, cache.Iᶻ)

    # Kronecker products: no BC
    Dᶻ      = kron_s(cache.Iʸ, grid.Dᶻ )
    D²ᶻ     = kron_s(cache.Iʸ, grid.D²ᶻ)
    D⁴ᶻ     = kron_s(cache.Iʸ, grid.D⁴ᶻ)

    # Kronecker products: Dirichlet
    Dᶻᴰ     = kron_s(cache.Iʸ, grid.Dᶻᴰ )
    D²ᶻᴰ    = kron_s(cache.Iʸ, grid.D²ᶻᴰ)
    D⁴ᶻᴰ    = kron_s(cache.Iʸ, grid.D⁴ᶻᴰ)

    # Kronecker products: Neumann
    Dᶻᴺ     = kron_s(cache.Iʸ, grid.Dᶻᴺ)
    D²ᶻᴺ    = kron_s(cache.Iʸ, grid.D²ᶻᴺ)
    D⁴ᶻᴺ    = kron_s(cache.Iʸ, grid.D⁴ᶻᴺ)

    # Mixed derivatives
    Dʸᶻᴰ    = kron_s(grid.Dʸ,  grid.Dᶻᴰ)
    Dʸ²ᶻᴰ   = kron_s(grid.Dʸ,  grid.D²ᶻᴰ)
    D²ʸ²ᶻᴰ  = kron_s(grid.D²ʸ,  grid.D²ᶻᴰ)

    #### Create the grid object
    prob = Problem{T}(Dʸ, D²ʸ, D⁴ʸ,
                    Dᶻ, D²ᶻ, D⁴ᶻ,
                    Dᶻᴺ, D²ᶻᴺ, D⁴ᶻᴺ,
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
    D³ʸ :: SparseMatrixCSC{T, Int}
    D⁴ʸ :: SparseMatrixCSC{T, Int}

    # Chebyshev matrices (sparse)
    Dᶻ  :: SparseMatrixCSC{T, Int}
    D²ᶻ :: SparseMatrixCSC{T, Int}
    D³ᶻ :: SparseMatrixCSC{T, Int}
    D⁴ᶻ :: SparseMatrixCSC{T, Int}

    Dᶻᴰ  :: SparseMatrixCSC{T, Int}
    D²ᶻᴰ :: SparseMatrixCSC{T, Int}
    D³ᶻᴰ :: SparseMatrixCSC{T, Int}
    D⁴ᶻᴰ :: SparseMatrixCSC{T, Int}

    Dᶻᴺ  :: SparseMatrixCSC{T, Int}
    D²ᶻᴺ :: SparseMatrixCSC{T, Int}
    D³ᶻᴺ :: SparseMatrixCSC{T, Int}
    D⁴ᶻᴺ :: SparseMatrixCSC{T, Int}
end


function TwoDGrid(params::AbstractParams) 

    Ny = params.Ny
    Nz = params.Nz
    L  = params.L
    H  = params.H

    # setup Fourier differentiation matrices  
    fd  = FourierDiffn(Ny; L = L) 
    y   = fd.x
    Dʸ  = fd.D₁
    D²ʸ = fd.D₂
    D³ʸ = fd.D₃
    D⁴ʸ = fd.D₄     


    # Chebyshev in the z-direction
    cd  = ChebyshevDiffn(Nz, [0.0, H], 4)
    z   = cd.x
    Dᶻ  = cd.D₁
    D²ᶻ = cd.D₂
    D³ᶻ = cd.D₃
    D⁴ᶻ = cd.D₄

    T = eltype(Dʸ)

   # Convert to mutable matrices to allow BCs (shallow copy is sufficient)
    Dᶻᴰ  = copy(Dᶻ)
    D²ᶻᴰ = copy(D²ᶻ)
    D³ᶻᴰ = copy(D⁴ᶻ)
    D⁴ᶻᴰ = copy(D⁴ᶻ)

    Dᶻᴺ  = copy(Dᶻ)
    D²ᶻᴺ = copy(D²ᶻ)
    D³ᶻᴺ = copy(D⁴ᶻ)
    D⁴ᶻᴺ = copy(D⁴ᶻ)

    #### Create the grid object
    grid = TwoDGrid{T, typeof(y), typeof(Dʸ)}(
        Ny, Nz, L, H, y, z,
        Dʸ, D²ʸ, D³ʸ, D⁴ʸ,
        Dᶻ, D²ᶻ, D³ᶻ, D⁴ᶻ,
        Dᶻᴰ, D²ᶻᴰ, D³ᶻᴰ, D⁴ᶻᴰ,
        Dᶻᴺ, D²ᶻᴺ, D³ᶻᴺ, D⁴ᶻᴺ
    )

    #### Apply BCs 
    bc_handler = BoundaryConditionHandler(params.Nz)
    bc_handler(grid, :dirichlet)  
    bc_handler(grid, :neumann)  

    return grid
end




