```julia
"""
This code finds critical Rayleigh number for rotating Rayleigh Benrad Convection (rRBC)
where the domain is periodic in y-direction.
The code is benchmarked against Chandrashekar's theoretical results.
# Hydrodynamic and hydromagnetic stability by S. Chandrasekhar, 1961 (page no-95)
parameter: Ek (Ekman number) = 10⁻⁴
eigenvalue: critical modified Rayleigh number (Raᶜ) = 189.7
"""
# load required packages
using LazyGrids
using LinearAlgebra
using Printf
using StaticArrays
using SparseArrays
using SparseMatrixDicts
using FillArrays
using SpecialFunctions
using Parameters
using Test
using BenchmarkTools

using ArnoldiMethod: partialschur, partialeigen, LR, LI, LM, SR

@with_kw mutable struct TwoDimGrid{Ny, Nz} 
    y = @SVector zeros(Float64, Ny)
    z = @SVector zeros(Float64, Nz)
end

@with_kw mutable struct ChebMarix{Ny, Nz} 
    𝒟ʸ::Array{Float64,  2}   = SparseMatrixCSC(Zeros(Ny, Ny))
    𝒟²ʸ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Ny, Ny))
    𝒟⁴ʸ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Ny, Ny))

    𝒟ᶻ::Array{Float64,  2}   = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟²ᶻ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟⁴ᶻ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Nz, Nz))

    𝒟ᶻᴺ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟²ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟⁴ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))

    𝒟ᶻᴰ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟²ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟⁴ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))
end

@with_kw mutable struct Operator{N}
"""
    `subperscript with N' means Operator with Neumann boundary condition 
        after kronker product
    `subperscript with D' means Operator with Dirchilet boundary condition
        after kronker product
""" 

    𝒟ʸ::Array{Float64,  2}   = SparseMatrixCSC(Zeros(N, N))
    𝒟²ʸ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(N, N))
    𝒟⁴ʸ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(N, N))

    𝒟ᶻ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))

    𝒟ᶻᴺ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟⁴ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))

    𝒟ᶻᴰ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟ʸᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟⁴ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))

    𝒟ʸ²ᶻᴰ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟²ʸ²ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
end

# Construct the derivative operator
function Construct_DerivativeOperator!(diffMatrix, grid, params)
    N = params.Ny * params.Nz

    # ------------- setup differentiation matrices  -------------------
    # Fourier in y-direction: y ∈ [0, L)
    y1, diffMatrix.𝒟ʸ  = FourierDiff(params.Ny, 1)
    _,  diffMatrix.𝒟²ʸ = FourierDiff(params.Ny, 2)
    _,  diffMatrix.𝒟⁴ʸ = FourierDiff(params.Ny, 4)

    # Transform the domain and derivative operators from [0, 2π) → [0, L)
    grid.y         = params.L/2π  * y1
    diffMatrix.𝒟ʸ  = (2π/params.L)^1 * diffMatrix.𝒟ʸ
    diffMatrix.𝒟²ʸ = (2π/params.L)^2 * diffMatrix.𝒟²ʸ
    diffMatrix.𝒟⁴ʸ = (2π/params.L)^4 * diffMatrix.𝒟⁴ʸ

    #@assert maximum(grid.y) ≈ params.L && minimum(grid.y) ≈ 0.0

    # Chebyshev in the z-direction
    z, diffMatrix.𝒟ᶻ  = cheb(params.Nz-1)
    grid.z = z
    diffMatrix.𝒟²ᶻ = diffMatrix.𝒟ᶻ  * diffMatrix.𝒟ᶻ
    diffMatrix.𝒟⁴ᶻ = diffMatrix.𝒟²ᶻ * diffMatrix.𝒟²ᶻ


    # z1, D1z = chebdif(params.Nz, 1)
    # _,  D2z = chebdif(params.Nz, 2)
    # _,  D3z = chebdif(params.Nz, 3)
    # _,  D4z = chebdif(params.Nz, 4)
    # # Transform the domain and derivative operators from [-1, 1] → [0, H]
    # grid.z, diffMatrix.𝒟ᶻ, diffMatrix.𝒟²ᶻ  = chebder_transform(z1,  D1z, 
    #                                                                 D2z, 
    #                                                                 zerotoL_transform, 
    #                                                                 params.H)
    # _, _, diffMatrix.𝒟⁴ᶻ = chebder_transform_ho(z1, D1z, 
    #                                                 D2z, 
    #                                                 D3z, 
    #                                                 D4z, 
    #                                                 zerotoL_transform_ho, 
    #                                                 params.H)
    
    #@printf "size of Chebyshev matrix: %d × %d \n" size(diffMatrix.𝒟ᶻ)[1]  size(diffMatrix.𝒟ᶻ)[2]

    @assert maximum(grid.z) ≈ params.H && minimum(grid.z) ≈ 0.0

    return nothing
end

function ImplementBCs_cheb!(Op, diffMatrix, params)
    Iʸ = sparse(Matrix(1.0I, params.Ny, params.Ny)) #Eye{Float64}(params.Ny)
    Iᶻ = sparse(Matrix(1.0I, params.Nz, params.Nz)) #Eye{Float64}(params.Nz)

    # Cheb matrix with Dirichilet boundary condition
    diffMatrix.𝒟ᶻᴰ  = deepcopy( diffMatrix.𝒟ᶻ  )
    diffMatrix.𝒟²ᶻᴰ = deepcopy( diffMatrix.𝒟²ᶻ )
    diffMatrix.𝒟⁴ᶻᴰ = deepcopy( diffMatrix.𝒟⁴ᶻ )

    # Cheb matrix with Neumann boundary condition
    diffMatrix.𝒟ᶻᴺ  = deepcopy( diffMatrix.𝒟ᶻ  )
    diffMatrix.𝒟²ᶻᴺ = deepcopy( diffMatrix.𝒟²ᶻ )

    setBCs!(diffMatrix, params, "dirchilet")
    setBCs!(diffMatrix, params, "neumann"  )
    
    kron!( Op.𝒟ᶻᴰ  ,  Iʸ , diffMatrix.𝒟ᶻᴰ  )
    kron!( Op.𝒟²ᶻᴰ ,  Iʸ , diffMatrix.𝒟²ᶻᴰ )
    kron!( Op.𝒟⁴ᶻᴰ ,  Iʸ , diffMatrix.𝒟⁴ᶻᴰ )

    kron!( Op.𝒟ᶻᴺ  ,  Iʸ , diffMatrix.𝒟ᶻᴺ )
    kron!( Op.𝒟²ᶻᴺ ,  Iʸ , diffMatrix.𝒟²ᶻᴺ)

    kron!( Op.𝒟ʸ   ,  diffMatrix.𝒟ʸ  ,  Iᶻ ) 
    kron!( Op.𝒟²ʸ  ,  diffMatrix.𝒟²ʸ ,  Iᶻ )
    kron!( Op.𝒟⁴ʸ  ,  diffMatrix.𝒟⁴ʸ ,  Iᶻ ) 

    kron!( Op.𝒟ʸ²ᶻᴰ  ,  diffMatrix.𝒟ʸ  ,  diffMatrix.𝒟²ᶻᴰ )
    kron!( Op.𝒟²ʸ²ᶻᴰ ,  diffMatrix.𝒟²ʸ ,  diffMatrix.𝒟²ᶻᴰ )

    return nothing
end

function construct_matrices(Op, params)
    N  = params.Ny * params.Nz
    I⁰ = sparse(Matrix(1.0I, N, N)) #Eye{Float64}(N)
    s₁ = size(I⁰, 1); s₂ = size(I⁰, 2)

    # allocating memory for the LHS and RHS matrices
    𝓛₁ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))
    𝓛₂ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))
    𝓛₃ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))

    ℳ₁ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))
    ℳ₂ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))
    ℳ₃ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))

    @printf "Start constructing matrices \n"
    # -------------------- construct matrix  ------------------------
    # lhs of the matrix (size := 3 × 3)
    # eigenvectors: [uᶻ ωᶻ b]ᵀ

    ∇ₕ² = SparseMatrixCSC(Zeros(N, N))
    ∇ₕ² = (1.0 * Op.𝒟²ʸ - 1.0 * params.kₓ^2 * I⁰)

    D⁴ = (1.0 * Op.𝒟⁴ʸ + 1.0 * Op.𝒟⁴ᶻᴰ + 2.0 * Op.𝒟²ʸ²ᶻᴰ 
        + 1.0 * params.kₓ^4 * I⁰ 
        - 2.0 * params.kₓ^2 * Op.𝒟²ʸ 
        - 2.0 * params.kₓ^2 * Op.𝒟²ᶻᴰ)
        
    D²  = 1.0 * Op.𝒟²ᶻᴰ + 1.0 * Op.𝒟²ʸ - 1.0 * params.kₓ^2 * I⁰
    Dₙ² = 1.0 * Op.𝒟²ᶻᴺ + 1.0 * Op.𝒟²ʸ - 1.0 * params.kₓ^2 * I⁰   

    #* 1. uᶻ equation
    𝓛₁[:,    1:1s₂] =  1.0 * params.E * D⁴ 
    𝓛₁[:,1s₂+1:2s₂] = -1.0 * Op.𝒟ᶻᴺ
    𝓛₁[:,2s₂+1:3s₂] =  0.0 * I⁰ 

    #* 2. ωᶻ equation 
    𝓛₂[:,    1:1s₂] = 1.0 * Op.𝒟ᶻᴰ
    𝓛₂[:,1s₂+1:2s₂] = 1.0 * params.E * Dₙ²
    𝓛₂[:,2s₂+1:3s₂] = 0.0 * I⁰        

    #* 3. b equation 
    𝓛₃[:,    1:1s₂] = 1.0 * I⁰ 
    𝓛₃[:,1s₂+1:2s₂] = 0.0 * I⁰
    𝓛₃[:,2s₂+1:3s₂] = 1.0 * D²     

    𝓛 = ([𝓛₁; 𝓛₂; 𝓛₃]);

##############
    ℳ₁[:,2s₂+1:3s₂] = -1.0 * ∇ₕ²

    ℳ = ([ℳ₁; ℳ₂; ℳ₃])

    @printf "Done constructing matrices \n"

    return 𝓛, ℳ
end


# @with_kw mutable struct Params{T1<:Real} @deftype T1
#     L::T1        = 1.0        # horizontal domain size
#     H::T1        = 1.0          # vertical domain size
#     kₓ::T1       = 0.0          # x-wavenumber
#     E::T1        = 1.0e-4       # Ekman number 
#     Ny::Int64   = 48          # no. of y-grid points
#     Nz::Int64   = 24           # no. of z-grid points
#     #method::String    = "shift_invert"
#     method::String    = "krylov"
#     #method::String   = "arnoldi"
# end

@with_kw mutable struct Params{T<:Real} @deftype T
    L::T        = 2π        # horizontal domain size
    H::T        = 1.0          # vertical domain size
    Γ::T        = 0.1         # front strength Γ ≡ M²/f² = λ/H = 1/ε → ε = 1/Γ
    ε::T        = 0.1         # aspect ratio ε ≡ H/L
    kₓ::T       = 0.0          # x-wavenumber
    E::T        = 1.0e-4       # Ekman number 
    Ny::Int64   = 180          # no. of y-grid points
    Nz::Int64   = 20           # no. of z-grid points
    #method::String    = "shift_invert"
    #method::String    = "krylov"
    method::String   = "arnoldi"
end

function EigSolver(Op, params, σ₀)

    printstyled("kₓ: $(params.kₓ) \n"; color=:blue)

    𝓛, ℳ = construct_matrices(Op,  params)
    
    N = params.Ny * params.Nz 
    MatrixSize = 3N
    @assert size(𝓛, 1)  == MatrixSize && 
            size(𝓛, 2)  == MatrixSize &&
            size(ℳ, 1)  == MatrixSize &&
            size(ℳ, 2)  == MatrixSize "matrix size does not match!"

    if params.method == "shift_invert"
        printstyled("Eigensolver using Arpack eigs with shift and invert method ...\n"; 
                    color=:red)

        λₛ, Χ = EigSolver_shift_invert_arpack( 𝓛, ℳ, σ₀=σ₀, maxiter=40, which=:LM)
        
        #@printf "found eigenvalue (at first): %f + im %f \n" λₛ[1].re λₛ[1].im

    elseif params.method == "krylov"
        printstyled("KrylovKit Method ... \n"; color=:red)

        # look for the largest magnitude of eigenvalue (:LM)
         λₛ, Χ = EigSolver_shift_invert_krylov( 𝓛, ℳ, σ₀=σ₀, maxiter=40, which=:LM)
        
        #@printf "found eigenvalue (at first): %f + im %f \n" λₛ[1].re λₛ[1].im

    elseif params.method == "arnoldi"

        printstyled("Arnoldi: based on Implicitly Restarted Arnoldi Method ... \n"; 
                        color=:red)

        # decomp, history = partialschur(construct_linear_map(𝓛, ℳ), 
        #                             nev=20, 
        #                             tol=0.0, 
        #                             restarts=50000, 
        #                             which=LM())

        # println(history)

        # λₛ⁻¹, Χ = partialeigen(decomp)
        # λₛ = @. 1.0 / λₛ⁻¹

        λₛ, Χ = EigSolver_shift_invert_arnoldi( 𝓛, ℳ, 
                                            σ₀=0.0, 
                                            maxiter=50000, 
                                            which=LM())

        λₛ, Χ = remove_evals(λₛ, Χ, 10.0, 1.0e15, "R")
        λₛ, Χ = sort_evals(λₛ, Χ, "R", "")

    end
    # ======================================================================

    @printf "||𝓛Χ - λₛℳΧ||₂: %f \n" norm(𝓛 * Χ[:,1] - λₛ[1] * ℳ * Χ[:,1])
    
    #print_evals(λₛ, length(λₛ))
    @printf "largest growth rate : %1.4e%+1.4eim\n" real(λₛ[1]) imag(λₛ[1])

    # 𝓛 = nothing
    # ℳ = nothing

    #return nothing #
    return λₛ[1] #, Χ[:,1]
end

function solve_rRBC(kₓ::Float64)
    params      = Params{Float64}(kₓ=0.5)
    grid        = TwoDimGrid{params.Ny,  params.Nz}()
    diffMatrix  = ChebMarix{ params.Ny,  params.Nz}()
    Op          = Operator{params.Ny * params.Nz}()
    Construct_DerivativeOperator!(diffMatrix, grid, params)
    ImplementBCs_cheb!(Op, diffMatrix, params)
    
    σ₀   = 0.0
    params.kₓ = kₓ
    
    λₛ = EigSolver(Op, params, σ₀)

    # Theoretical results from Chandrashekar (1961)
    λₛₜ = 189.7 
    @printf "Analytical solution of Stone (1971): %1.4e \n" λₛₜ 

    return abs(real(λₛ) - λₛₜ)/λₛₜ < 1e-4
    
end

#end #module
# ========== end of the module ==========================


if abspath(PROGRAM_FILE) == @__FILE__
    solve_rRBC(0.0)
end