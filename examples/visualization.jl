# ## Load required packages
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
using Parameters: @with_kw

using BiGSTARS
using BiGSTARS: AbstractParams
using BiGSTARS: Problem, OperatorI, TwoDGrid

# ## Parameters
@with_kw mutable struct Params{T} <: AbstractParams
    L::T                = 1.0           # horizontal domain size
    H::T                = 1.0           # vertical domain size
    Ri::T               = 1.0           # the Richardson number 
    ε::T                = 0.1           # aspect ratio ε ≡ H/L
    k::T                = 0.1           # along-front wavenumber
    E::T                = 1.0e-8        # the Ekman number 
    Ny::Int64           = 24            # no. of y-grid points
    Nz::Int64           = 20            # no. of z-grid points
end
nothing #hide


"""
    calculation of u and v from w and ζ
    
    ```math
    -∇ₕ²\\hat{u} = ik∂ᶻ\\hat{w} + ∂ʸ\\hat{ζ}
    -∇ₕ²\\hat{v} =  ∂ʸᶻ\\hat{w} - ik\\hat{ζ} 
"""
function cal_u_v(X, params, grid, prob)
    Ny = params.Ny
    Nz = params.Nz
    N  = Ny * Nz

    u = zeros(ComplexF64, N)
    v = zeros(ComplexF64, N)

    w  = X[1:1N,1]
    ζ  = X[1N+1:2N,1]
    
    wʳ = real(w)    # real part of w
    wⁱ = imag(w)    # imaginary part of w
    
    ζʳ = real(ζ)
    ζⁱ = imag(ζ) 
    
    I⁰ = sparse(Matrix(1.0I, N, N)) 
    s₁ = size(I⁰, 1); 
    s₂ = size(I⁰, 2)

    ∇ₕ² = SparseMatrixCSC(Zeros(N, N))
    ∇ₕ² = (1.0 * prob.D²ʸ - 1.0 * params.kₓ^2 * I⁰)

    # Setup inverse operator (see utils.jl)
    H   = InverseLaplace(∇ₕ²)

    tmp1 = prob.Dᶻᴰ * wʳ; 
    tmp2 = prob.Dᶻᴰ * wⁱ
    ∂zw  = @. tmp1 + 1.0im * tmp2

    tmp1 = Op.𝒟ʸ * ζʳ; tmp2 = Op.𝒟ʸ * ζⁱ
    ∂yζ  = @. tmp1 + 1.0im * tmp2

    tmp1 = Op.𝒟ʸᶻᴰ * wʳ; tmp2 = Op.𝒟ʸᶻᴰ * wⁱ
    ∂yzw = @. tmp1 + 1.0im * tmp2

    ## calculating `u`
    tmp1 = 1.0im * kₓ * ∂zw + ∂yζ
    u = -1.0 * H(tmp1)

    ## calculating `v` 
    tmp1 = 1.0 * ∂yzw - 1.0im * kₓ * ζ
    v = -1.0 * H(tmp1)

    return u, v
end


function plot_eigfun()

    ## here we are doing it for Stone (1971)
    filename = "stone_ms_eigenval.jld2.jld2"
	file = jldopen(filename, "r"); 
    k   = file["k"];
	λ   = file["λ"];
	X   = file["X"];
	close(file)


    ## calling all the problem parameters 
    params = Params{Float64}()

    ## Construct grid and derivative operators
    grid  = TwoDGrid(params)

    ## Construct the necesary operator
    ops  = OperatorI(params)
    prob = Problem(grid, ops)








end