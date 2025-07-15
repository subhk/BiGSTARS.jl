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
@with_kw mutable struct Params{T} <: AbstractParams
    L::T                = 2π            # horizontal domain size
    H::T                = 1.0           # vertical   domain size
    E::T                = 1.0e-4        # inverse of Reynolds number 
    k::T                = 0.0           # x-wavenumber
    Ny::Int64           = 120           # no. of y-grid points
    Nz::Int64           = 30            # no. of Chebyshev points
    w_bc::String        = "rigid_lid"   # boundary condition for vertical velocity
    ζ_bc::String        = "free_slip"   # boundary condition for vertical vorticity
    b_bc::String        = "fixed"        # boundary condition for temperature
    eig_solver::String  = "arnoldi"      # eigenvalue solver
end
nothing #hide

# ### Define the basic state
function basic_state(grid, params)
    
    Y, Z = ndgrid(grid.y, grid.z)
    Y    = transpose(Y)
    Z    = transpose(Z)

    ## Define the basic state
    B₀   = @. Z - params.H  # temperature
    U₀   = @. 0.0 * Z       # velocity

    ## Calculate all the necessary derivatives
    deriv = compute_derivatives(U₀, B₀, grid.y, grid.Dᶻ, grid.D²ᶻ, :All)

    bs = initialize_basic_state_from_fields(B₀, U₀)

    initialize_basic_state!(
            bs,
            deriv.∂ʸB₀,  deriv.∂ᶻB₀, 
            deriv.∂ʸU₀,  deriv.∂ᶻU₀,
            deriv.∂ʸʸU₀, deriv.∂ᶻᶻU₀, deriv.∂ʸᶻU₀,
            deriv.∂ʸʸB₀, deriv.∂ᶻᶻB₀, deriv.∂ʸᶻB₀
        )

    return bs
end

# ### Constructing Generalized EVP
function generalized_EigValProb(prob, grid, params)

    bs = basic_state(grid, params)

    N  = params.Ny * params.Nz
    I⁰ = sparse(Matrix(1.0I, N, N)) 
    s₁ = size(I⁰, 1); 
    s₂ = size(I⁰, 2);

    ## the horizontal Laplacian operator
    ∇ₕ² = SparseMatrixCSC(Zeros(N, N))
    ∇ₕ² = (1.0 * prob.D²ʸ - 1.0 * params.k^2 * I⁰)

    ## inverse of the horizontal Laplacian operator
    H = inverse_Lap_hor(∇ₕ²)

    ## Construct the 4th order derivative
    D⁴  = (1.0 * prob.D⁴ʸ 
        + 1.0 * prob.D⁴ᶻᴰ 
        + 1.0 * params.k^4 * I⁰ 
        - 2.0 * params.k^2 * prob.D²ʸ 
        - 2.0 * params.k^2 * prob.D²ᶻᴰ
        + 2.0 * prob.D²ʸ²ᶻᴰ)
        
    ## Construct the 2nd order derivative
    D²  = (1.0 * prob.D²ᶻᴰ  + 1.0 * ∇ₕ²)
    Dₙ² = (1.0  * prob.D²ᶻᴺ + 1.0 * ∇ₕ²)

    ## Construct the matrix `A`
    ## ──────────────────────────────────────────────────────────────────────────────
    ## 1) Now define your 3×3 block-rows in a NamedTuple of 3-tuples
    ## ──────────────────────────────────────────────────────────────────────────────
    ## Construct the matrix `A`
    Ablocks = (
        w = (  # w-equation: ED⁴ -Dᶻ zero
                sparse(params.E * D⁴),
                sparse(-prob.Dᶻᴺ),
                spzeros(Float64, s₁, s₂)
        ),
        ζ = (  # ζ-equation: Dᶻ ED² zero
                sparse(prob.Dᶻᴰ),
                sparse(params.E * Dₙ²),
                spzeros(Float64, s₁, s₂)
        ),
        b = (  # b-equation: I zero D²
                sparse(I⁰),
                spzeros(Float64, s₁, s₂),
                sparse(D²)
        )
    )

    ## Construct the matrix `B`
    Bblocks = (
        w = (  # w-equation: zero, zero -∇ₕ²
                spzeros(Float64, s₁, s₂),
                spzeros(Float64, s₁, s₂),
                sparse(-∇ₕ²)
        ),
        ζ = (  # ζ-equation: zero, zero, zero
                spzeros(Float64, s₁, s₂),
                spzeros(Float64, s₁, s₂),
                spzeros(Float64, s₁, s₂)
        ),
        b = (  # b-equation: zero, zero, zero
                spzeros(Float64, s₁, s₂),
                spzeros(Float64, s₁, s₂),
                spzeros(Float64, s₁, s₂)
        )
    )

    ## ──────────────────────────────────────────────────────────────────────────────
    ## 2) Assemble in beautiful line
    ## ──────────────────────────────────────────────────────────────────────────────
    gevp = GEVPMatrices(Ablocks, Bblocks)


    ## ──────────────────────────────────────────────────────────────────────────────
    ## 3) And now you have exactly:
    ##    gevp.A, gevp.B                    → full sparse matrices
    ##    gevp.As.w, gevp.As.ζ, gevp.As.b   → each block-row view
    ##    gevp.Bs.w, gevp.Bs.ζ, gevp.Bs.b
    ## ──────────────────────────────────────────────────────────────────────────────

    return gevp.A, gevp.B
end
nothing #hide


# ### Define the eigenvalue solver
function EigSolver(prob, grid, params, σ₀)

    A, B = generalized_EigValProb(prob, grid, params)

    if params.eig_solver == "arpack"

        λ, Χ = solve_shift_invert_arpack(A, B; 
                                        σ₀=σ₀, 
                                        which=:LM, 
                                        sortby=:R, 
                                        nev = 10,
                                        maxiter=100)

    elseif params.eig_solver == "krylov"

        λ, Χ = solve_shift_invert_krylov(A, B; 
                                        σ₀=σ₀, 
                                        which=:LM, 
                                        sortby=:R, 
                                        maxiter=100)

    elseif params.eig_solver == "arnoldi"

        λ, Χ = solve_shift_invert_arnoldi(A, B; 
                                        σ₀=σ₀, 
                                        which=:LM, 
                                        sortby=:R,
                                        nev = 10, 
                                        maxiter=100)
    end
    ## ======================================================================
    @assert length(λ) > 0 "No eigenvalue(s) found!"

    @printf "||AΧ - λBΧ||₂: %f \n" norm(A * Χ[:,1] - λ[1] * B * Χ[:,1])

    ## looking for min Ra 
    λ, Χ = remove_evals(λ, Χ, 10.0, 1.0e15, "R")
    λ, Χ = sort_evals_(λ, Χ,  :R, rev=false)

    print_evals(complex.(λ))

    return λ[1], Χ[:,1]
end
nothing #hide

# ### solving the rRBC problem
function solve_rRBC(k::Float64)

    ## Calling problem parameters
    params = Params{Float64}()

    ## Construct grid and derivative operators
    grid  = TwoDGrid(params)

    ## Construct the necesary operator
    ops  = OperatorI(params)
    prob = Problem(grid, ops, params)

    ## update the wavenumber
    params.k = k

    ## initial guess for the growth rate
    σ₀   = 0.0 

    λ, Χ = EigSolver(prob, grid, params, σ₀)

    ## Theoretical results from Chandrashekar (1961)
    λₜ = 189.7 
    @printf "Analytical solution of critical Ra: %1.4e \n" λₜ 

    return abs(real(λ) - λₜ)/λₜ < 1e-4
end
nothing #hide


# ### Result
solve_rRBC(0.0) # Critical Rayleigh number is at k=0.0
nothing #hide