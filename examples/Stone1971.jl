"""
Stability of a 2D front based on Stone (1971)
"""
## load required packages
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

using JLD2
using ModelingToolkit
using NonlinearSolve

using BiGSTARS
using BiGSTARS : FourierDiff, cheb_coord_transform_ho, 

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

@with_kw mutable struct MeanFlow{N} 
    B₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
    U₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))

  ∇ʸU₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
  ∇ᶻU₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
  ∇ʸB₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
  ∇ᶻB₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))

  ∇ʸʸU₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
  ∇ᶻᶻU₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
  ∇ʸᶻU₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
end


# function Construct_DerivativeOperator!(diffMatrix, grid, params)
#     N = params.Ny * params.Nz

#     y1, diffMatrix.𝒟ʸ  = FourierDiff(params.Ny, 1)
#     y2, d2y = FourierDiff(params.Ny, 2)
#     diffMatrix.𝒟²ʸ = d2y
#     # y4, d4y = FourierDiff(params.Ny, 4)
#     # diffMatrix.𝒟⁴ʸ = d4y

#     # Transform the domain and derivative operators from [0, 2π) → [0, L)
#     grid.y         = params.L/2π  * y1
#     diffMatrix.𝒟ʸ  = (2π/params.L)^1 * diffMatrix.𝒟ʸ
#     diffMatrix.𝒟²ʸ = (2π/params.L)^2 * diffMatrix.𝒟²ʸ
#     diffMatrix.𝒟⁴ʸ = (2π/params.L)^4 * diffMatrix.𝒟⁴ʸ

#     z1,  D1z = chebdif(params.Nz, 1)
#     z2,  D2z = chebdif(params.Nz, 2)
#     z3,  D3z = chebdif(params.Nz, 3)
#     z4,  D4z = chebdif(params.Nz, 4)

#     ## Transform the domain and derivative operators from [-1, 1] → [0, H]
#     grid.z, diffMatrix.𝒟ᶻ, diffMatrix.𝒟²ᶻ  = chebder_transform(z1,  D1z, 
#                                                                     D2z, 
#                                                                     zerotoL_transform, 
#                                                                     params.H)
#     p1, q1, diffMatrix.𝒟⁴ᶻ = chebder_transform_ho(z1, D1z, 
#                                                     D2z, 
#                                                     D3z, 
#                                                     D4z, 
#                                                     zerotoL_transform_ho, 
#                                                     params.H)

#     return nothing
# end


# function ImplementBCs_cheb!(Op, diffMatrix, params)
#     Iʸ = sparse(Matrix(1.0I, params.Ny, params.Ny)) 
#     Iᶻ = sparse(Matrix(1.0I, params.Nz, params.Nz)) 

#     # Cheb matrix with Dirichilet boundary condition
#     @. diffMatrix.𝒟ᶻᴰ  = diffMatrix.𝒟ᶻ 
#     @. diffMatrix.𝒟²ᶻᴰ = diffMatrix.𝒟²ᶻ 
#     @. diffMatrix.𝒟⁴ᶻᴰ = diffMatrix.𝒟⁴ᶻ 

#     # Cheb matrix with Neumann boundary condition
#     @. diffMatrix.𝒟ᶻᴺ  = diffMatrix.𝒟ᶻ  
#     @. diffMatrix.𝒟²ᶻᴺ = diffMatrix.𝒟²ᶻ 

#     n = params.Nz
#     for iter ∈ 1:n-1
#         diffMatrix.𝒟⁴ᶻᴰ[1,iter+1] = (diffMatrix.𝒟⁴ᶻᴰ[1,iter+1] + 
#                                 -1.0 * diffMatrix.𝒟⁴ᶻᴰ[1,1] * diffMatrix.𝒟²ᶻᴰ[1,iter+1])

#           diffMatrix.𝒟⁴ᶻᴰ[n,iter] = (diffMatrix.𝒟⁴ᶻᴰ[n,iter] + 
#                                 -1.0 * diffMatrix.𝒟⁴ᶻᴰ[n,n] * diffMatrix.𝒟²ᶻᴰ[n,iter])
#     end

#     diffMatrix.𝒟ᶻᴰ[1,1]  = 0.0
#     diffMatrix.𝒟ᶻᴰ[n,n]  = 0.0

#     diffMatrix.𝒟²ᶻᴰ[1,1] = 0.0
#     diffMatrix.𝒟²ᶻᴰ[n,n] = 0.0   

#     diffMatrix.𝒟⁴ᶻᴰ[1,1] = 0.0
#     diffMatrix.𝒟⁴ᶻᴰ[n,n] = 0.0  

#     # Neumann boundary condition
#     @. diffMatrix.𝒟ᶻᴺ  = diffMatrix.𝒟ᶻ 
#     @. diffMatrix.𝒟²ᶻᴺ = diffMatrix.𝒟²ᶻ
#     for iter ∈ 1:n-1
#         diffMatrix.𝒟²ᶻᴺ[1,iter+1] = (diffMatrix.𝒟²ᶻᴺ[1,iter+1] + 
#                                 -1.0 * diffMatrix.𝒟²ᶻᴺ[1,1] * diffMatrix.𝒟ᶻᴺ[1,iter+1]/diffMatrix.𝒟ᶻᴺ[1,1])

#         diffMatrix.𝒟²ᶻᴺ[n,iter]   = (diffMatrix.𝒟²ᶻᴺ[n,iter] + 
#                                 -1.0 * diffMatrix.𝒟²ᶻᴺ[n,n] * diffMatrix.𝒟ᶻᴺ[n,iter]/diffMatrix.𝒟ᶻᴺ[n,n])
#     end

#     diffMatrix.𝒟²ᶻᴺ[1,1] = 0.0
#     diffMatrix.𝒟²ᶻᴺ[n,n] = 0.0

#     @. diffMatrix.𝒟ᶻᴺ[1,1:end] = 0.0
#     @. diffMatrix.𝒟ᶻᴺ[n,1:end] = 0.0

#     #setBCs!(diffMatrix, params, "dirchilet")
#     #setBCs!(diffMatrix, params, "neumann"  )
    
#     kron!( Op.𝒟ᶻᴰ  ,  Iʸ , diffMatrix.𝒟ᶻᴰ  )
#     kron!( Op.𝒟²ᶻᴰ ,  Iʸ , diffMatrix.𝒟²ᶻᴰ )
#     kron!( Op.𝒟⁴ᶻᴰ ,  Iʸ , diffMatrix.𝒟⁴ᶻᴰ )

#     kron!( Op.𝒟ᶻᴺ  ,  Iʸ , diffMatrix.𝒟ᶻᴺ )
#     kron!( Op.𝒟²ᶻᴺ ,  Iʸ , diffMatrix.𝒟²ᶻᴺ)

#     kron!( Op.𝒟ʸ   ,  diffMatrix.𝒟ʸ  ,  Iᶻ ) 
#     kron!( Op.𝒟²ʸ  ,  diffMatrix.𝒟²ʸ ,  Iᶻ )
#     kron!( Op.𝒟⁴ʸ  ,  diffMatrix.𝒟⁴ʸ ,  Iᶻ ) 

#     kron!( Op.𝒟ʸᶻᴰ   ,  diffMatrix.𝒟ʸ  ,  diffMatrix.𝒟ᶻᴰ  )
#     kron!( Op.𝒟ʸ²ᶻᴰ  ,  diffMatrix.𝒟ʸ  ,  diffMatrix.𝒟²ᶻᴰ )
#     kron!( Op.𝒟²ʸ²ᶻᴰ ,  diffMatrix.𝒟²ʸ ,  diffMatrix.𝒟²ᶻᴰ )

#     return nothing
# end


# function BasicState!(diffMatrix, mf, grid, params)
#     Y, Z = ndgrid(grid.y, grid.z)
#     Y    = transpose(Y)
#     Z    = transpose(Z)

#     # imposed buoyancy profile
#     B₀   = @. 1.0/params.Γ * Z - Y  
#     ∂ʸB₀ = - 1.0 .* ones(size(Y))  
#     ∂ᶻB₀ = 1.0/params.Γ .* ones(size(Y))  

#     U₀      = @. 1.0 * Z - 0.5params.H
#     ∂ᶻU₀    = ones( size(Y)) 
#     ∂ʸU₀    = zeros(size(Y)) 

#     ∂ʸʸU₀   = zeros(size(Y)) 
#     ∂ʸᶻU₀   = zeros(size(Y))
#     ∂ᶻᶻU₀   = zeros(size(Y))

#       B₀  = B₀[:];
#       U₀  = U₀[:];
#     ∂ʸB₀  = ∂ʸB₀[:]; 
#     ∂ᶻB₀  = ∂ᶻB₀[:];

#     ∂ᶻU₀  = ∂ᶻU₀[:];
#     ∂ʸU₀  = ∂ʸU₀[:];
    
#     ∂ʸʸU₀ = ∂ʸʸU₀[:];
#     ∂ʸᶻU₀ = ∂ʸᶻU₀[:];
#     ∂ᶻᶻU₀ = ∂ᶻᶻU₀[:];  

#     mf.B₀[diagind(mf.B₀)] = B₀
#     mf.U₀[diagind(mf.U₀)] = U₀

#     mf.∇ᶻU₀[diagind(mf.∇ᶻU₀)] = ∂ᶻU₀
#     mf.∇ʸU₀[diagind(mf.∇ʸU₀)] = ∂ʸU₀

#     mf.∇ʸB₀[diagind(mf.∇ʸB₀)] = ∂ʸB₀
#     mf.∇ᶻB₀[diagind(mf.∇ᶻB₀)] = ∂ᶻB₀

#     mf.∇ʸʸU₀[diagind(mf.∇ʸʸU₀)] = ∂ʸʸU₀;
#     mf.∇ᶻᶻU₀[diagind(mf.∇ᶻᶻU₀)] = ∂ᶻᶻU₀;
#     mf.∇ʸᶻU₀[diagind(mf.∇ʸᶻU₀)] = ∂ʸᶻU₀;

#     return nothing
# end

function construct_matrices(Op, mf, grid, params)
    Y, Z = ndgrid(grid.y, grid.z)
    Y    = transpose(Y)
    Z    = transpose(Z)

    ## basic state
    B₀   = @. 1.0/params.Γ * Z - Y  
    ∂ʸB₀ = - 1.0 .* ones(size(Y))  
    ∂ᶻB₀ = 1.0/params.Γ .* ones(size(Y))  

    U₀      = @. 1.0 * Z - 0.5params.H
    ∂ᶻU₀    = ones( size(Y)) 
    ∂ʸU₀    = zeros(size(Y)) 

    ∂ʸʸU₀   = zeros(size(Y)) 
    ∂ʸᶻU₀   = zeros(size(Y))
    ∂ᶻᶻU₀   = zeros(size(Y))

      B₀  = B₀[:];
      U₀  = U₀[:];
    ∂ʸB₀  = ∂ʸB₀[:]; 
    ∂ᶻB₀  = ∂ᶻB₀[:];

    ∂ᶻU₀  = ∂ᶻU₀[:];
    ∂ʸU₀  = ∂ʸU₀[:];
    
    ∂ʸʸU₀ = ∂ʸʸU₀[:];
    ∂ʸᶻU₀ = ∂ʸᶻU₀[:];
    ∂ᶻᶻU₀ = ∂ᶻᶻU₀[:];  

    mf.B₀[diagind(mf.B₀)] = B₀
    mf.U₀[diagind(mf.U₀)] = U₀

    mf.∇ᶻU₀[diagind(mf.∇ᶻU₀)] = ∂ᶻU₀
    mf.∇ʸU₀[diagind(mf.∇ʸU₀)] = ∂ʸU₀

    mf.∇ʸB₀[diagind(mf.∇ʸB₀)] = ∂ʸB₀
    mf.∇ᶻB₀[diagind(mf.∇ᶻB₀)] = ∂ᶻB₀

    mf.∇ʸʸU₀[diagind(mf.∇ʸʸU₀)] = ∂ʸʸU₀;
    mf.∇ᶻᶻU₀[diagind(mf.∇ᶻᶻU₀)] = ∂ᶻᶻU₀;
    mf.∇ʸᶻU₀[diagind(mf.∇ʸᶻU₀)] = ∂ʸᶻU₀;

    N  = params.Ny * params.Nz
    I⁰ = sparse(Matrix(1.0I, N, N)) #Eye{Float64}(N)
    s₁ = size(I⁰, 1); s₂ = size(I⁰, 2)

    ## allocating memory for the LHS and RHS matrices
    𝓛₁ = SparseMatrixCSC(Zeros{ComplexF64}(s₁, 3s₂))
    𝓛₂ = SparseMatrixCSC(Zeros{ComplexF64}(s₁, 3s₂))
    𝓛₃ = SparseMatrixCSC(Zeros{ComplexF64}(s₁, 3s₂))

    ℳ₁ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))
    ℳ₂ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))
    ℳ₃ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))

    ∇ₕ² = SparseMatrixCSC(Zeros(N, N))
    H   = SparseMatrixCSC(Zeros(N, N))

    ∇ₕ² = (1.0 * Op.𝒟²ʸ - 1.0 * params.kₓ^2 * I⁰)


    H = inverse_Lap_hor(∇ₕ²)
    @assert norm(∇ₕ² * H - I⁰) ≤ 1.0e-4 "difference in L2-norm should be small"


    D⁴  = (1.0 * Op.𝒟⁴ʸ 
        + 1.0/params.ε^4 * Op.𝒟⁴ᶻᴰ 
        + 1.0params.kₓ^4 * I⁰ 
        - 2.0params.kₓ^2 * Op.𝒟²ʸ 
        - 2.0/params.ε^2 * params.kₓ^2 * Op.𝒟²ᶻᴰ
        + 2.0/params.ε^2 * Op.𝒟²ʸ²ᶻᴰ)
        
    D²  = (1.0/params.ε^2 * Op.𝒟²ᶻᴰ + 1.0 * ∇ₕ²)
    Dₙ² = (1.0/params.ε^2 * Op.𝒟²ᶻᴺ + 1.0 * ∇ₕ²)

    ## 1. uᶻ (vertical velocity)  equation (bcs: uᶻ = ∂ᶻᶻuᶻ = 0 @ z = 0, 1)
    𝓛₁[:,    1:1s₂] = (-1.0params.E * D⁴ 
                    + 1.0im * params.kₓ * mf.U₀ * D²) * params.ε^2
    𝓛₁[:,1s₂+1:2s₂] = 1.0 * Op.𝒟ᶻᴺ 
    𝓛₁[:,2s₂+1:3s₂] = -1.0 * ∇ₕ²

    ## 2. ωᶻ (vertical vorticity) equation (bcs: ∂ᶻωᶻ = 0 @ z = 0, 1)
    𝓛₂[:,    1:1s₂] = - 1.0 * mf.∇ᶻU₀ * Op.𝒟ʸ - 1.0 * Op.𝒟ᶻᴰ
    𝓛₂[:,1s₂+1:2s₂] = (1.0im * params.kₓ * mf.U₀ * I⁰
                    - 1.0params.E * Dₙ²)
    𝓛₂[:,2s₂+1:3s₂] = 0.0 * I⁰        

    ## 3. b (buoyancy) equation (bcs: b = 0 @ z = 0, 1)
    𝓛₃[:,    1:1s₂] = (1.0 * mf.∇ᶻB₀ * I⁰
                    - 1.0 * mf.∇ʸB₀ * H * Op.𝒟ʸᶻᴰ) 
    𝓛₃[:,1s₂+1:2s₂] = 1.0im * params.kₓ * mf.∇ʸB₀ * H * I⁰
    𝓛₃[:,2s₂+1:3s₂] = (-1.0params.E * Dₙ² 
                    + 1.0im * params.kₓ * mf.U₀ * I⁰) 

    𝓛 = ([𝓛₁; 𝓛₂; 𝓛₃]);

    
    cnst = -1.0 
    ℳ₁[:,    1:1s₂] = 1.0cnst * params.ε^2 * D²;
    ℳ₂[:,1s₂+1:2s₂] = 1.0cnst * I⁰;
    ℳ₃[:,2s₂+1:3s₂] = 1.0cnst * I⁰;
    ℳ = ([ℳ₁; ℳ₂; ℳ₃])
    
    return 𝓛, ℳ
end


# Parameters:
@with_kw mutable struct Params{T<:Real} @deftype T
    L::T        = 1.0        # horizontal domain size
    H::T        = 1.0        # vertical domain size
    Γ::T        = 0.1        # front strength Γ ≡ M²/f² = λ/H = 1/ε → ε = 1/Γ
    ε::T        = 0.1        # aspect ratio ε ≡ H/L
    kₓ::T       = 0.0        # x-wavenumber
    E::T        = 1.0e-9     # Ekman number 
    Ny::Int64   = 48         # no. of y-grid points
    Nz::Int64   = 24         # no. of z-grid points
    method::String = "krylov"
end


function EigSolver(Op, mf, grid, params, σ₀)

    𝓛, ℳ = construct_matrices(Op, mf, grid, params)
    
    N = params.Ny * params.Nz 
    MatrixSize = 3N
    @assert size(𝓛, 1)  == MatrixSize && 
            size(𝓛, 2)  == MatrixSize &&
            size(ℳ, 1)  == MatrixSize &&
            size(ℳ, 2)  == MatrixSize "matrix size does not match!"

    if params.method == "shift_invert"
        λₛ = EigSolver_shift_invert( 𝓛, ℳ, σ₀=σ₀)

    elseif params.method == "krylov"

        λₛ, Χ = EigSolver_shift_invert_krylov( 𝓛, ℳ, σ₀=σ₀, maxiter=40, which=:LR)
        

    elseif params.method == "arnoldi"

        λₛ, Χ = EigSolver_shift_invert_arnoldi( 𝓛, ℳ, σ₀=σ₀, maxiter=40, which=:LR)
    end
    # ======================================================================
    @assert length(λₛ) > 0 "No eigenvalue(s) found!"

    @printf "||𝓛Χ - λₛℳΧ||₂: %f \n" norm(𝓛 * Χ[:,1] - λₛ[1] * ℳ * Χ[:,1])
    
    @printf "largest growth rate : %1.4e%+1.4eim\n" real(λₛ[1]) imag(λₛ[1])

    return λₛ[1] #, Χ[:,1]
end


function solve_Stone1971(kₓ::Float64=0.0)
    params      = Params{Float64}(kₓ=0.5)
    grid        = TwoDimGrid{params.Ny,  params.Nz}()
    diffMatrix  = ChebMarix{ params.Ny,  params.Nz}()
    Op          = Operator{params.Ny * params.Nz}()
    mf          = MeanFlow{params.Ny * params.Nz}()

    Construct_DerivativeOperator!(diffMatrix, grid, params)
    ImplementBCs_cheb!(Op, diffMatrix, params)

    σ₀   = 0.01
    params.kₓ = kₓ
    
    λₛ = EigSolver(Op, mf, grid, params, σ₀)

    ## Analytical solution of Stone (1971) for the growth rate
    cnst = 1.0 + 1.0/params.Γ + 5.0*params.ε^2 * params.kₓ^2/42.0 
    λₛₜ = 1.0/(2.0*√3.0) * (params.kₓ - 2.0/15.0 * params.kₓ^3 * cnst)

    return abs(λₛ.re - λₛₜ) < 1e-3

end

solve_Stone1971(0.1)

println("Example runs OK")

