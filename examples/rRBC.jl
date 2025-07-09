# ## This code finds critical Rayleigh number for rotating Rayleigh Benrad Convection (rRBC)
# ## where the domain is periodic in y-direction.
# ## The code is benchmarked against Chandrashekar's theoretical results.
# ## Hydrodynamic and hydromagnetic stability by S. Chandrasekhar, 1961 (page no-95)
# ## parameter: Ek (Ekman number) = 10⁻⁴
# ## eigenvalue: critical modified Rayleigh number (Raᶜ) = 189.7

# ## Problem setup
# In this module, we do a linear stability analysis of a 2D rotating Rayleigh-Bernard case where the domain is periodic in the ``y``-direction, in the ``x``-direction is of infinite extent and vertically bounded. The reason to choose this simple case is because we can find an analytical solution for this case. The background temperature profile is given by 
# ```math
# \overline{\theta} = 1 - z.
# ```
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
# where ```E=\nu/(fH^2)``` is the Ekman number and ```Ra = g\alpha \Delta T/(f \kappa)```, ```\Delta T``` is the temperature difference between the bottom and the top walls) is the modified Rayleigh number.
# By applying the operators ```(\nabla \times \nabla \times)``` and ```(\nabla \times)``` and taking the ```z```-component of the equations and assuming wave-like perturbations as done previously, we obtained the equations for vertical velocity ```w^```, vertical vorticity ```\zeta``` and temperature ```\theta```,
# ```math
# \begin{align}
#     E \mathcal{D}^4 w - \partial_z \zeta &= -Ra \mathcal{D}_h^2 \theta,
# \\
#     E \mathcal{D}^2 \zeta + \partial_z w &= 0,
# \\
#     \mathcal{D}^2 b + w &= 0.
# \end{align}
# ```
# The boundary conditions are: 
# ```math
# \begin{align}
#     w = \partial_z^2 w = \partial_z \zeta = \theta = 0
#     \,\,\,\,\,\ \text{at} \,\,\, z=0,1
# \end{align}
# ```
# The boundary conditions are implemented in 
# ```@docs
# BiGSTARS.setBCs!
# ```

# ## Normal mode 
# Next we consider normal-mode perturbation solutions in the form of (we seek stationary solutions at the marginal state, i.e., ```\sigma = 0```),
# ```math
# \begin{align}
#     [w, \zeta, \theta](x,y,z,t) =
# \mathfrak{R}\big([\tilde{w}, \, \tilde{\zeta}, \, \tilde{\theta}](y, z) \, e^{i k x + \sigma t}\big),
# \end{align}
# ```
# where the symbol ``\mathfrak{R}`` denotes the real part and a variable with `tilde' denotes an eigenfunction. 
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
# \mathcal{D}^4  = (\mathcal{D}^2 )^2 = \big(\partial_y^2 + \partial_z^2 - k^2\big)^2, \,\,\,\, \text{and} \,\, \mathcal{D}_h^2 = (\partial_y^2 - k^2).
# ```
# The eigenfunctions ``\tilde{u}``, ``\tilde{v}`` are related to ``\tilde{w}``, ``\tilde{\zeta}`` by the relations 
# ```math
# \begin{align}
#     -\mathcal{D}_h^2 \tilde{u} &= i k \partial_{z} \tilde{w} + \partial_y \tilde{\zeta},
# \\   
#     -\mathcal{D}_h^2 \tilde{v} &= \partial_{yz} \tilde{w} -  i k \tilde{\zeta}.
# \end{align}
# ```
# We choose periodic boundary conditions in the ``y``-direction and free-slip, rigid lid, with zero buoyancy flux in the ``z`` direction, i.e., 
# ```math
# \begin{align}
#     \tilde{w} = \partial_{zz} \tilde{w} = 
#     \partial_z \tilde{\zeta} = \partial_z \tilde{b} = 0, 
#     \,\,\,\,\,\,\, \text{at} \,\,\, {z}=0, 1.
# \end{align}
# ```
# The above sets of equations with the boundary conditions can be expressed as a standard generalized eigenvalue problem,
# ```math
# \begin{align}
#     \mathsfit{A} \mathsf{X}= \lambda \mathsfit{B} \mathsf{X},   
# \end{align}
# ```
# where ```\lambda=Ra``` is the eigenvalue. 

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

using ArnoldiMethod: partialschur, partialeigen, LR, LI, LM, SR

# ## Let's begin
using BiGSTARS

# ## Define the grid and derivative operators
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

# ## `subperscript with N' means Operator with Neumann boundary condition 
# ##        after kronker product
# ##    `subperscript with D' means Operator with Dirchilet boundary condition
# ##        after kronker product
@with_kw mutable struct Operator{N}
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
    ∇ₕ² = (1.0 * Op.𝒟²ʸ - 1.0 * params.kₓ^2 * I⁰)

    D⁴ = (1.0 * Op.𝒟⁴ʸ + 1.0 * Op.𝒟⁴ᶻᴰ + 2.0 * Op.𝒟²ʸ²ᶻᴰ 
        + 1.0 * params.kₓ^4 * I⁰ 
        - 2.0 * params.kₓ^2 * Op.𝒟²ʸ 
        - 2.0 * params.kₓ^2 * Op.𝒟²ᶻᴰ)
        
    D²  = 1.0 * Op.𝒟²ᶻᴰ + 1.0 * Op.𝒟²ʸ - 1.0 * params.kₓ^2 * I⁰
    Dₙ² = 1.0 * Op.𝒟²ᶻᴺ + 1.0 * Op.𝒟²ʸ - 1.0 * params.kₓ^2 * I⁰   

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

# ## Define the parameters
@with_kw mutable struct Params{T<:Real} @deftype T
    L::T        = 2π          # horizontal domain size
    H::T        = 1.0         # vertical domain size
    Γ::T        = 0.1         # front strength Γ ≡ M²/f² = λ/H = 1/ε → ε = 1/Γ
    ε::T        = 0.1         # aspect ratio ε ≡ H/L
    kₓ::T       = 0.0         # x-wavenumber
    E::T        = 1.0e-4      # Ekman number 
    Ny::Int64   = 180         # no. of y-grid points
    Nz::Int64   = 20          # no. of z-grid points
    method::String   = "arnoldi"
end
nothing #hide

# ## Define the eigenvalue solver
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

    return λₛ[1] #, Χ[:,1]
end
nothing #hide

# ## solving the rRBC problem
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

    ## Theoretical results from Chandrashekar (1961)
    λₛₜ = 189.7 

    return abs(real(λₛ) - λₛₜ)/λₛₜ < 1e-4
    
end
nothing #hide

solve_rRBC(0.0)
nothing #hide