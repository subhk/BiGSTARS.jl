# ### Linear stability analysis of baroclinic instability of a 2D front based on Stone (1971)
#
# ## Introduction
# Baroclinic instability (BCI) arises when a rotating, stratified fluid has tilted density surfaces, 
# enabling eddies to tap available potential energy and convert it to kinetic energy.
# Stone (1971) investigated non-geostrophic effects on BCI using Eady’s framework. 
# He found that as the $Ri$ decreases, the wavelength of the most unstable mode increases 
# while the growth rate diminishes relative to predictions from the quasigeostrophic (QG) approximation.
# ## Basic state
# The basic state is given by
# ```math
# \begin{align}
#     B(y, z) &= Ri z - y, \\
#     U(y, z) &= z - {1}/{2},
# \end{align}
# ```
# where ``Ri`` is the Richardson number. We aim to analyze the stability of the 
# above basic state against small perturbations. The perturbation variables are
# defined as
# ```math
# \begin{align}
#     \mathbf{u}(x, y, z, t) &= (u, v, \epsilon w)(x, y, z, t), \\
#     p(x, y, z, t) &= p(x, y, z, t), \\
#     b(x, y, z, t) &= b(x, y, z, t),
# \end{align}
# ```
# where ``\epsilon`` is the aspect ratio, ``\mathbf{u}`` is the velocity perturbation, 
# ``p`` is the pressure perturbation, and ``b`` is the buoyancy perturbation.
#
# ## Governing equations
# The resulting nondimensional, linearized Boussinesq equations of motion 
# under the ``f``-plane approximation are given by
# ```math
# \begin{align}
#     \frac{D \mathbf{u}}{Dt}
#     + \Big(v \frac{\partial U}{\partial y} + w \frac{\partial U}{\partial z} \Big) \hat{x}
#     + \hat{z} \times \mathbf{u} &=
#     -\nabla p + \frac{1}{\epsilon} b \hat{z} + E \nabla^2 \mathbf{u}, 
# \\
#     \frac{Db}{Dt}
#     +  v \frac{\partial B}{\partial y} + w \frac{\partial B}{\partial z} &= \frac{E}{Pr} \nabla^2 b, 
# \\
#     \nabla \cdot \mathbf{u} &= 0,
# \end{align}
# ```
# where 
# ```math
# \begin{align}
#   D/Dt \equiv \partial/\partial t + U (\partial/\partial x)
# \end{align}
# ```
# is the material derivative. The operators:
# ```math
# \nabla \equiv (\partial/\partial x, \partial/\partial y, (1/\epsilon) \partial/\partial z),
# ```
# ```math
# \nabla^2 \equiv \partial^2/\partial x^2 + \partial^2/\partial y^2 + (1/\epsilon^2) \partial^2/ \partial z^2,
# ```
# ```math
#   \nabla_h^2 \equiv \partial^2 /\partial x^2 + \partial^2/\partial y^2.
# ```
#
# To eliminate pressure, following [teed2010rapidly@citet, we apply the operator 
# $\hat{z} \cdot \nabla \times \nabla \times$  and $\hat{z} \cdot \nabla \times$ 
# to the above momentum equation. This procedure yields governing equations of 
# three perturbation variables, the vertical velocity $w$, the vertical vorticity $\zeta \, (=\hat{z} \cdot \nabla \times \mathbf{u})$, 
# and the buoyancy $b$ 
# ```math
# \begin{align}
#     \frac{D}{Dt}\nabla^2 {w} 
#     + \frac{1}{\epsilon^2} \frac{\partial \zeta}{\partial z} 
#     &= \frac{1}{\epsilon^2} \nabla_h^2 b + E \nabla^4 w,
# \\
#     \frac{D \zeta}{Dt}
#     - \frac{\partial U}{\partial z}\frac{\partial w}{\partial y}
#     - \frac{\partial w}{\partial z} &= E \nabla^2 \zeta, 
# \\
#     \frac{Db}{Dt}
#     + v \frac{\partial B}{\partial y} + 
#     w \frac{\partial B}{\partial z}
#     &= \frac{E}{Pr} \nabla^2 b,
# \end{align}
# ```
# where 
# ```math
#   \nabla_h^2 \equiv \partial^2 /\partial x^2 + \partial^2/\partial y^2.
# ```
# The benefit of using the above sets of equations is that it enables us to 
# examine the instability at an along-front wavenumber ``k \to 0``. 
#
#
# ## Normal mode solutions
# Next we consider normal-mode perturbation solutions in the form of 
# ```math
# \begin{align}
#     [w, \zeta, b](x,y,z,t) = \mathfrak{R}\big([\tilde{w}, \tilde{\zeta}, \tilde{b}](y, z)  e^{i kx + \sigma t}\big),
# \end{align}
# ```
# where the symbol $\mathfrak{R}$ denotes the real part and a variable with `tilde' denotes an eigenfunction. The variable 
# $\sigma=\sigma_r + i \sigma_i$. The real part represents the growth rate, and the imaginary part 
# shows the frequency of the  perturbation. 
#
# Finally following systems of differential equations are obtained,
# ```math
# \begin{align}
#     (i k U - E \mathcal{D}^2) \mathcal{D}^2 \tilde{w}
#     + \epsilon^{-2} \partial_z \tilde{\zeta}
#     - \epsilon^{-2} \mathcal{D}_h^2 \tilde{b} &= -\sigma \mathcal{D}^2 \tilde{w},
# \\
#     - \partial_z U \partial_y \tilde{w}
#     - \partial_z \tilde{w}
#     + \left(ik U - E \mathcal{D}^2 \right) \tilde{\zeta} &= -\sigma \tilde{\zeta},
# \\
#     \partial_z B \tilde{w} + \partial_y B  \tilde{v} + 
#     \left[ik U - E \mathcal{D}^2 \right] \tilde{b} &= -\sigma \tilde{b}, 
# \end{align}
# ```
# where 
# ```math
#  \mathcal{D}^4  = (\mathcal{D}^2 )^2 = \big(\partial_y^2 + (1/\epsilon^2)\partial_z^2 - k^2\big)^2, 
# ```
# and
# ```math
#  \mathcal{D}_h^2 = (\partial_y^2 - k^2).
# ```
# 
# ## Boundary conditions
# We choose periodic boundary conditions in the ``y``-direction and 
# free-slip, rigid lid, with zero buoyancy flux in the ``z`` direction, i.e., 
# ```math
# \begin{align}
#   \tilde{w} = \partial_{zz} \tilde{w} = 
#   \partial_z \tilde{\zeta} = \partial_z \tilde{b} = 0, 
#   \,\,\,\,\,\,\, \text{at} \,\,\, {z}=0, 1.
# \end{align}
# ```
# ## Generalized eigenvalue problem
# The above sets of equations with the boundary conditions can be expressed as a 
# standard generalized eigenvalue problem,
# ```math
# \begin{align}
#  AX= λBX,
# \end{align}
# ```
# where $\lambda$ is the eigenvalue, and $X$ is the eigenvector. The matrices    
# $A$ and $B$ are given by
# ```math
# \begin{align}
#     A &= \begin{bmatrix}
#         \epsilon^2(i k U \mathcal{D}^2 -E \mathcal{D}^4)   
#          & \mathcal{D}_z  & -\mathcal{D}_h^2 
#   \\  
#         -\partial_z U \mathcal{D}_y - \mathcal{D}_z  
#           & i k U - E \mathcal{D}^2 & 0 
#  \\ 
#       \partial_z B -  \partial_y B H \mathcal{D}_{yz}   
#       &  k \partial_y B H  & ikU - E \mathcal{D}^2 
#     \end{bmatrix}, 
# \,\,\,\,\,\,\,
#     B &= \begin{bmatrix}  
#         \epsilon^2 \mathcal{D}^2 & 0 & 0 \\   
#         0 & I & 0 \\
#         0 & 0 & I
#     \end{bmatrix},
# \end{align}
# ```
# where $I$ is the identity matrix and $H$ is the inverse of the horizontal Laplacian $(\mathcal{D}_h^2)^{-1}$.
#
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
using ModelingToolkit
using NonlinearSolve

using BiGSTARS


# ### Define the parameters
@with_kw mutable struct Params{T<:Real} @deftype T
    L::T        = 1.0        # horizontal domain size
    H::T        = 1.0        # vertical domain size
    Ri::T       = 0.1       # the Richardson number
    ε::T        = 0.1        # aspect ratio ε ≡ H/L
    k::T        = 0.0        # x-wavenumber
    E::T        = 1.0e-9     # Ekman number 
    Ny::Int64   = 48         # no. of y-grid points
    Nz::Int64   = 24         # no. of z-grid points
    method::String = "krylov"
end
nothing #hide


# ### Define the grid and derivative operators
@with_kw mutable struct TwoDimGrid{Ny, Nz} 
    y = @SVector zeros(Float64, Ny)
    z = @SVector zeros(Float64, Nz)
end
nothing #hide

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
nothing #hide


@with_kw mutable struct Operator{N}
    ## `subperscript N' means Operator with Neumann boundary condition  
    ## `subperscript D' means Operator with Dirchilet boundary condition 
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
nothing #hide

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
nothing #hide


# ### Constructing the derivative operators
function construct_matrices(Op, mf, grid, params)
    Y, Z = ndgrid(grid.y, grid.z)
    Y    = transpose(Y)
    Z    = transpose(Z)

    ## basic state
    B₀   = @. 1.0params.Ri * Z - Y  
    ∂ʸB₀ = - 1.0 .* ones(size(Y))  
    ∂ᶻB₀ = 1.0params.Ri .* ones(size(Y))  

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

    ∇ₕ² = (1.0 * Op.𝒟²ʸ - 1.0 * params.k^2 * I⁰)


    H = inverse_Lap_hor(∇ₕ²)
    @assert norm(∇ₕ² * H - I⁰) ≤ 1.0e-4 "difference in L2-norm should be small"


    D⁴  = (1.0 * Op.𝒟⁴ʸ 
        + 1.0/params.ε^4 * Op.𝒟⁴ᶻᴰ 
        + 1.0 * params.k^4 * I⁰ 
        - 2.0 * params.k^2 * Op.𝒟²ʸ 
        - 2.0/params.ε^2 * params.k^2 * Op.𝒟²ᶻᴰ
        + 2.0/params.ε^2 * Op.𝒟²ʸ²ᶻᴰ)
        
    D²  = (1.0/params.ε^2 * Op.𝒟²ᶻᴰ + 1.0 * ∇ₕ²)
    Dₙ² = (1.0/params.ε^2 * Op.𝒟²ᶻᴺ + 1.0 * ∇ₕ²)

    ## 1. uᶻ (vertical velocity)  equation (bcs: uᶻ = ∂ᶻᶻuᶻ = 0 @ z = 0, 1)
    𝓛₁[:,    1:1s₂] = (-1.0 * params.E * D⁴ 
                    + 1.0im * params.k * mf.U₀ * D²) * params.ε^2
    𝓛₁[:,1s₂+1:2s₂] = 1.0 * Op.𝒟ᶻᴺ 
    𝓛₁[:,2s₂+1:3s₂] = -1.0 * ∇ₕ²

    ## 2. ωᶻ (vertical vorticity) equation (bcs: ∂ᶻωᶻ = 0 @ z = 0, 1)
    𝓛₂[:,    1:1s₂] = - 1.0 * mf.∇ᶻU₀ * Op.𝒟ʸ - 1.0 * Op.𝒟ᶻᴰ
    𝓛₂[:,1s₂+1:2s₂] = (1.0im * params.k * mf.U₀ * I⁰ - 1.0 * params.E * Dₙ²)
    𝓛₂[:,2s₂+1:3s₂] = 0.0 * I⁰

    ## 3. b (buoyancy) equation (bcs: b = 0 @ z = 0, 1)
    𝓛₃[:,    1:1s₂] = (1.0 * mf.∇ᶻB₀ * I⁰
                    - 1.0 * mf.∇ʸB₀ * H * Op.𝒟ʸᶻᴰ) 
    𝓛₃[:,1s₂+1:2s₂] = 1.0im * params.k * mf.∇ʸB₀ * H * I⁰
    𝓛₃[:,2s₂+1:3s₂] = (-1.0 * params.E * Dₙ² 
                    + 1.0im * params.k * mf.U₀ * I⁰) 

    𝓛 = ([𝓛₁; 𝓛₂; 𝓛₃]);

    
    cnst = -1.0 
    ℳ₁[:,    1:1s₂] = 1.0cnst * params.ε^2 * D²;
    ℳ₂[:,1s₂+1:2s₂] = 1.0cnst * I⁰;
    ℳ₃[:,2s₂+1:3s₂] = 1.0cnst * I⁰;
    ℳ = ([ℳ₁; ℳ₂; ℳ₃])
    
    return 𝓛, ℳ
end
nothing #hide


# ### Define the eigenvalue solver
function EigSolver(Op, mf, grid, params, σ₀)

    𝓛, ℳ = construct_matrices(Op, mf, grid, params)

    if params.method == "shift_invert"
        λₛ = EigSolver_shift_invert( 𝓛, ℳ, σ₀=σ₀)

    elseif params.method == "krylov"

        λₛ, Χ = EigSolver_shift_invert_krylov( 𝓛, ℳ, σ₀=σ₀, maxiter=40, which=:LR)
        
    elseif params.method == "arnoldi"

        λₛ, Χ = EigSolver_shift_invert_arnoldi( 𝓛, ℳ, σ₀=σ₀, maxiter=40, which=:LR)
    end
    ## ======================================================================
    @assert length(λₛ) > 0 "No eigenvalue(s) found!"

    @printf "||𝓛Χ - λₛℳΧ||₂: %f \n" norm(𝓛 * Χ[:,1] - λₛ[1] * ℳ * Χ[:,1])
    
    @printf "largest growth rate : %1.4e%+1.4eim\n" real(λₛ[1]) imag(λₛ[1])

    return λₛ[1] #, Χ[:,1]
end
nothing #hide

# ### Solving the Stone problem
function solve_Stone1971(k::Float64=0.0)
    params      = Params{Float64}(k=0.5)
    grid        = TwoDimGrid{params.Ny,  params.Nz}()
    diffMatrix  = ChebMarix{ params.Ny,  params.Nz}()
    Op          = Operator{params.Ny * params.Nz}()
    mf          = MeanFlow{params.Ny * params.Nz}()

    Construct_DerivativeOperator!(diffMatrix, grid, params)
    ImplementBCs_cheb!(Op, diffMatrix, params)

    σ₀   = 0.02 # initial guess for the growth rate
    params.k = k
    
    λₛ = EigSolver(Op, mf, grid, params, σ₀)

    ## Analytical solution of Stone (1971) for the growth rate
    cnst = 1.0 + 1.0 * params.Ri + 5.0 * params.ε^2 * params.k^2 / 42.0
    λₛₜ = 1.0 / (2.0 * √3.0) * (params.k - 2.0 / 15.0 * params.k^3 * cnst)

    @printf "Analytical solution of Stone (1971) for the growth rate: %f \n" λₛₜ

    return abs(λₛ.re - λₛₜ) < 1e-3

end
nothing #hide

# ## Result
solve_Stone1971(0.1) # growth rate is at k=0.1  
nothing #hide
