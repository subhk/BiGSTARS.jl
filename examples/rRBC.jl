# ### Finding critical Rayleigh number for rotating Rayleigh-Benard Convection 
#
# ## Introduction
# This code finds critical Rayleigh number for the onset of convection for rotating Rayleigh Benrad Convection (rRBC)
# where the domain is periodic in y-direction.
# The code is benchmarked against Chandrashekar's theoretical results.
# Hydrodynamic and hydromagnetic stability by S. Chandrasekhar, 1961 (page no-95).
#
# Parameter: 
#
# Ekman number $E = 10⁻⁴$
#
# Eigenvalue: critical modified Rayleigh number $Ra_c = 189.7$
#
# In this module, we do a linear stability analysis of a 2D rotating Rayleigh-Bernard case where the domain 
# is periodic in the ``y``-direction, in the ``x``-direction is of infinite extent and vertically bounded. 
#
# ## Basic state
# The background temperature profile $\overline{\theta}$ is given by 
# ```math
# \overline{\theta} = 1 - z.
# ```
# 
# ## Governing equations
# The non-dimensional form of the equations governing the perturbation is given by 
# ```math
#     \frac{E}{Pr} \frac{\partial \mathbf{u}}{\partial t} 
#     + \hat{z} \times \mathbf{u} =
#     -\nabla p + Ra \theta \hat{z} + E \nabla^2 \mathbf{u},
# ```
# ```math
#     \frac{\partial \theta}{\partial t} 
#     = \mathbf{u} \cdot \hat{z} + \nabla^2 \theta,
# ```
# ```math
#     \nabla \cdot \mathbf{u} = 0,
# ```
# where $E=\nu/(fH^2)$ is the Ekman number and $Ra = g\alpha \Delta T/(f \kappa)$, 
# $\Delta T$ is the temperature difference between the bottom and the top walls) is the modified Rayleigh number.
# By applying the operators $(\nabla \times \nabla \times)$ and $(\nabla \times)$ and 
# taking the $z$-component of the equations and assuming wave-like perturbations, 
# we obtained the equations for vertical velocity $w$, vertical vorticity $\zeta$ and temperature $\theta$,
# ```math
# \begin{align}
#     E \mathcal{D}^4 w - \partial_z \zeta &= -Ra \mathcal{D}_h^2 \theta,
# \\
#     E \mathcal{D}^2 \zeta + \partial_z w &= 0,
# \\
#     \mathcal{D}^2 b + w &= 0.
# \end{align}
# ```
#
# ## Normal mode solutions
# Next we consider normal-mode perturbation solutions in the form of (we seek stationary solutions at the marginal state, i.e., $\sigma = 0$),
# ```math
# \begin{align}
#  [w, \zeta, \theta](x,y,z,t) = \mathfrak{R}\big([\tilde{w}, \tilde{\zeta}, \tilde{\theta}](y, z)  e^{i kx + \sigma t}\big),
# \end{align}
# ```
# where the symbol $\mathfrak{R}$ denotes the real part and a variable with `tilde' denotes an eigenfunction. 
# Finally following systems of differential equations are obtained,
# ```math
# \begin{align}
#     E \mathcal{D}^4  \tilde{w} - \partial_z \tilde{\zeta} &= - Ra \mathcal{D}_h^2 \tilde{\theta},
# \\
#     E \mathcal{D}^2 \tilde{\zeta} + \partial_z \tilde{w} &= 0,
# \\
#     \mathcal{D}^2 \tilde{\theta} + \tilde{w} &= 0, 
# \end{align}
# ```
# where 
# ```math
# \begin{align}
# \mathcal{D}^4  = (\mathcal{D}^2 )^2 = \big(\partial_y^2 + \partial_z^2 - k^2\big)^2, 
# \,\,\,\, \text{and} \,\, \mathcal{D}_h^2 = (\partial_y^2 - k^2).
# \end{align}
# ```
# The eigenfunctions ``\tilde{u}``, ``\tilde{v}`` are related to ``\tilde{w}``, ``\tilde{\zeta}`` by the relations 
# ```math
# \begin{align}
#     -\mathcal{D}_h^2 \tilde{u} &= i k \partial_{z} \tilde{w} + \partial_y \tilde{\zeta},
# \\   
#     -\mathcal{D}_h^2 \tilde{v} &= \partial_{yz} \tilde{w} -  i k \tilde{\zeta}.
# \end{align}
# ```
#
# ## Boundary conditions
# We choose periodic boundary conditions in the ``y``-direction and free-slip, rigid lid, with zero buoyancy flux in the ``z`` direction, i.e., 
# ```math
# \begin{align}
#     \tilde{w} = \partial_{zz} \tilde{w} = 
#     \partial_z \tilde{\zeta} = \partial_z \tilde{b} = 0, 
#     \,\,\,\,\,\,\, \text{at} \,\,\, {z}=0, 1.
# \end{align}
# ```
#
# ## Generalized eigenvalue problem
# The above sets of equations with the boundary conditions can be expressed as a 
# standard generalized eigenvalue problem,
# ```math
# \begin{align}
#  AX= λBX, 
# \end{align}
# ```  
# where $\lambda=Ra$ is the eigenvalue, and $X$ is the eigenvector, The matrices    
# $A$ and $B$ are given by
# ```math
# \begin{align}
#     A &= \begin{bmatrix}
#         E \mathcal{D}^4 & -\mathcal{D}_z & 0 \\
#         \mathcal{D}_z & E \mathcal{D}^2 & 0 \\ 
#         I & 0 & \mathcal{D}^2
#     \end{bmatrix},
# \,\,\,\,\,\,\,
#     B &= \begin{bmatrix}
#         0 & 0 & -\mathcal{D}_h^2 \\
#         0 & 0 & 0 \\    
#         0 & 0 & 0 
#     \end{bmatrix}.
# \end{align}
# ```
# where $I$ is the identity matrix.
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
using BiGSTARS: AbstractParams
using BiGSTARS: Problem, OperatorI, TwoDGrid, Params

# ### Define the parameters
@with_kw mutable struct Params{T<:Real} @deftype T
    L::T        = 2π          # horizontal domain size
    H::T        = 1.0         # vertical domain size
    k::T        = 0.0         # x-wavenumber
    E::T        = 1.0e-4      # Ekman number 
    Ny::Int64   = 280         # no. of y-grid points
    Nz::Int64   = 18          # no. of z-grid points
    method::String   = "arnoldi"
end
nothing #hide
params = Params{Float64}()

# ### Construct grid and derivative operators
grid  = TwoDGrid(params)


# ### Define the basic state
function basic_state(grid, params)
    
    Y, Z = ndgrid(grid.y, grid.z)
    Y    = transpose(Y)
    Z    = transpose(Z)

    ## Define the basic state
    B₀   = @. 1.0 * Z - 1.0    # temperature
    U₀   = @. 1.0 * Z - 0.5 * params.H   # along-front velocity

    ## Calculate all the necessary derivatives
    derivs = compute_derivatives(U₀, B₀, y, grid.Dᶻ, grid.D²ᶻ, :All)

    bs = initialize_basic_state_from_fields(B₀, U₀)

    initialize_basic_state!(bs, deriv.∂ʸB₀, deriv.∂ᶻB₀, 
                                deriv.∂ʸU₀, deriv.∂ᶻU₀, 
                                deriv.∂ʸʸU₀, deriv.∂ᶻᶻU₀, 
                                deriv.∂ʸᶻU₀)

    return bs
end


function construct_matrices(Op, params)
    N  = params.Ny * params.Nz
    I⁰ = sparse(Matrix(1.0I, N, N)) #Eye{Float64}(N)
    s₁ = size(I⁰, 1); s₂ = size(I⁰, 2)

    ## allocating memory for the LHS and RHS matrices
    𝓛₁ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))
    𝓛₂ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))
    𝓛₃ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))

    ℳ₁ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))
    ℳ₂ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))
    ℳ₃ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))

    @printf "Start constructing matrices \n"
    ## -------------------- construct matrix  ------------------------
    ## lhs of the matrix (size := 3 × 3)
    ## eigenvectors: [uᶻ ωᶻ θ]ᵀ

    ∇ₕ² = SparseMatrixCSC(Zeros(N, N))
    ∇ₕ² = (1.0 * Op.𝒟²ʸ - 1.0 * params.k^2 * I⁰)

    D⁴ = (1.0 * Op.𝒟⁴ʸ + 1.0 * Op.𝒟⁴ᶻᴰ + 2.0 * Op.𝒟²ʸ²ᶻᴰ 
        + 1.0 * params.k^4 * I⁰ 
        - 2.0 * params.k^2 * Op.𝒟²ʸ 
        - 2.0 * params.k^2 * Op.𝒟²ᶻᴰ)

    D²  = 1.0 * Op.𝒟²ᶻᴰ + 1.0 * Op.𝒟²ʸ - 1.0 * params.k^2 * I⁰
    Dₙ² = 1.0 * Op.𝒟²ᶻᴺ + 1.0 * Op.𝒟²ʸ - 1.0 * params.k^2 * I⁰

    ## 1. uᶻ (vertical velocity) equation
    𝓛₁[:,    1:1s₂] =  1.0 * params.E * D⁴ 
    𝓛₁[:,1s₂+1:2s₂] = -1.0 * Op.𝒟ᶻᴺ
    𝓛₁[:,2s₂+1:3s₂] =  0.0 * I⁰ 

    ## 2. ωᶻ (vertical vorticity) equation 
    𝓛₂[:,    1:1s₂] = 1.0 * Op.𝒟ᶻᴰ
    𝓛₂[:,1s₂+1:2s₂] = 1.0 * params.E * Dₙ²
    𝓛₂[:,2s₂+1:3s₂] = 0.0 * I⁰        

    ## 3. θ (temperature) equation 
    𝓛₃[:,    1:1s₂] = 1.0 * I⁰ 
    𝓛₃[:,1s₂+1:2s₂] = 0.0 * I⁰
    𝓛₃[:,2s₂+1:3s₂] = 1.0 * D²     

    𝓛 = ([𝓛₁; 𝓛₂; 𝓛₃]);


    ℳ₁[:,2s₂+1:3s₂] = -1.0 * ∇ₕ²

    ℳ = ([ℳ₁; ℳ₂; ℳ₃])

    return 𝓛, ℳ
end
nothing #hide

# ### Define the eigenvalue solver
function EigSolver(Op, params, σ₀)

    𝓛, ℳ = construct_matrices(Op,  params)
    
    N = params.Ny * params.Nz 
    MatrixSize = 3N
    @assert size(𝓛, 1)  == MatrixSize && 
            size(𝓛, 2)  == MatrixSize &&
            size(ℳ, 1)  == MatrixSize &&
            size(ℳ, 2)  == MatrixSize "matrix size does not match!"

    if params.method == "shift_invert"

        λₛ, Χ = EigSolver_shift_invert_arpack( 𝓛, ℳ, σ₀=σ₀, maxiter=40, which=:LM)

    elseif params.method == "krylov"

         λₛ, Χ = EigSolver_shift_invert_krylov( 𝓛, ℳ, σ₀=σ₀, maxiter=40, which=:LM)

    elseif params.method == "arnoldi"

        λₛ, Χ = EigSolver_shift_invert_arnoldi( 𝓛, ℳ, 
                                            σ₀=0.0, 
                                            maxiter=50000, 
                                            which=LM())

        λₛ, Χ = remove_evals(λₛ, Χ, 10.0, 1.0e15, "R")
        λₛ, Χ = sort_evals(λₛ, Χ, "R", "")

    end

    @printf "Obtained critical Ra: %f \n" real(λₛ[1]) 

    return λₛ[1] #, Χ[:,1]
end
nothing #hide

# ### solving the rRBC problem
function solve_rRBC(k::Float64)
    params      = Params{Float64}(k=0.5)
    grid        = TwoDimGrid{params.Ny,  params.Nz}()
    diffMatrix  = ChebMarix{ params.Ny,  params.Nz}()
    Op          = Operator{params.Ny * params.Nz}()
    Construct_DerivativeOperator!(diffMatrix, grid, params)
    ImplementBCs_cheb!(Op, diffMatrix, params)
    
    σ₀   = 0.0
    params.k = k
    
    λₛ = EigSolver(Op, params, σ₀)

    ## Theoretical results from Chandrashekar (1961)
    λₛₜ = 189.7 

    @printf "Analytical solution of critical Ra: %f \n" λₛₜ

    return abs(real(λₛ) - λₛₜ)/λₛₜ < 1e-4
    
end
nothing #hide


# ## Result
solve_rRBC(0.0) # Critical Rayleigh number is at k=0.0
nothing #hide