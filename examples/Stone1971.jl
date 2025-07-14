# ### Linear stability analysis of baroclinic instability of a 2D front based on Stone (1971)
#
# ## Introduction
# Baroclinic instability (BCI) arises when a rotating, stratified fluid has tilted density surfaces, 
# enabling eddies to tap available potential energy and convert it to kinetic energy.
# Stone (1971) investigated non-hydrostatic effects on BCI using Eady’s framework. 
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
# standard generalized eigenvalue problem (GEVP),
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
using BiGSTARS: AbstractParams
using BiGSTARS: Problem, OperatorI, TwoDGrid, Params

# # Define abstract type first
# abstract type AbstractParams end

# ### Define the parameters
@with_kw struct Params{T} <: AbstractParams
    L::T                = 1.0          # horizontal domain size
    H::T                = 1.0          # vertical domain size
    Ri::T               = 1.0          # the Richardson number 
    ε::T                = 0.1          # aspect ratio ε ≡ H/L
    k::T                = 0.1          # along-front wavenumber
    E::T                = 1.0e-9       # the Ekman number 
    Ny::Int64           = 48           # no. of y-grid points
    Nz::Int64           = 24           # no. of z-grid points
    w_bc::String        = "rigid_lid"  # boundary condition for vertical velocity
    ζ_bc::String        = "free_slip"  # boundary condition for vertical vorticity
    b_bc::String        = "zero_flux"   # boundary condition for buoyancy
    eig_solver::String  = "krylov"     # eigenvalue solver
end
nothing #hide
params = Params{Float64}()

# ### Construct grid and derivative operators
grid  = TwoDGrid(params)

# ### Define the basic state
function basic_state(params, grid)
    ## Define the basic state
    B₀   = @. params.Ri * grid.z - grid.y     # buoyancy
    U₀   = @. 1.0 * grid.z - 0.5 * params.H   # along-front velocity

    ## Calculate all the necessary derivatives
    derivs = compute_derivatives(U₀, B₀, y, grid.Dᶻ, grid.D²ᶻ, :All)

    bs = initialize_basic_state_from_fields(B₀, U₀)

    initialize_basic_state!(bs, deriv.∂ʸB₀, deriv.∂ᶻB₀, 
                                deriv.∂ʸU₀, deriv.∂ᶻU₀, 
                                deriv.∂ʸʸU₀, deriv.∂ᶻᶻU₀, 
                                deriv.∂ʸᶻU₀)

    return bs
end

# ### Construct the necesary operator
ops  = OperatorI(params)
prob = Problem(grid, ops, params)

# ### Constructing GEVP
function generalized_EigValProb(Op, mf, grid, params)

    bs = basic_state(params, grid)

    N  = params.Ny * params.Nz
    I⁰ = sparse(Matrix(1.0I, N, N)) #Eye{Float64}(N)
    s₁ = size(I⁰, 1); s₂ = size(I⁰, 2)

    ## allocating memory for the LHS and RHS matrices
    labels  = [:w, :ζ, :b]  # eigenfunction labels
    GEVPMat = GEVPMatrices(ComplexF64, Float64, N; nblocks=4, labels=labels)

    ## the horizontal Laplacian operator
    ∇ₕ² = (1.0 * prob.D²ʸ - 1.0 * params.k^2 * I⁰)

    ## inverse of the horizontal Laplacian operator
    H = inverse_Lap_hor(∇ₕ²)
    @assert norm(∇ₕ² * H - I⁰) ≤ 1.0e-4 "difference in L2-norm should be small"

    ## Construct the 4th order derivative
    D⁴  = (1.0 * prob.D⁴ʸ 
        + 1.0/params.ε^4 * prob.D⁴ᶻᴰ 
        + 1.0 * params.k^4 * I⁰ 
        - 2.0 * params.k^2 * prob.D²ʸ 
        - 2.0/params.ε^2 * params.k^2 * prob.D²ᶻᴰ
        + 2.0/params.ε^2 * prob.D²ʸ²ᶻᴰ)
        
    ## Construct the 2nd order derivative
    D²  = (1.0/params.ε^2 * prob.D²ᶻᴰ + 1.0 * ∇ₕ²)
    Dₙ² = (1.0/params.ε^2 * prob.D²ᶻᴺ + 1.0 * ∇ₕ²)

    ## Construct the matrix `A`
    ## ----------------------------------------------------------------------
    ## 1. w (vertical velocity)  equation (bcs: w = ∂ᶻᶻw = 0 @ z = 0, 1)
    ## ----------------------------------------------------------------------
    GEVPMat.As.w[:,    1:1s₂] = (-1.0 * params.E * D⁴ 
                                + 1.0im * params.k * bs.fields.U₀ * D²) * params.ε^2

    GEVPMat.As.w[:,1s₂+1:2s₂] = 1.0 * prob.Dᶻᴺ 

    GEVPMat.As.w[:,2s₂+1:3s₂] = -1.0 * ∇ₕ²

    ## ----------------------------------------------------------------------
    ## 2. ζ (vertical vorticity) equation (bcs: ∂ᶻζ = 0 @ z = 0, 1)
    ## ----------------------------------------------------------------------
    GEVPMat.As.ζ[:,    1:1s₂] = - 1.0 * bs.fields.∂ᶻU₀ * prob.Dʸ - 1.0 * prob.Dᶻᴰ

    GEVPMat.As.ζ[:,1s₂+1:2s₂] = (1.0im * params.k * bs.fields.U₀ * I⁰ 
                                - 1.0 * params.E * Dₙ²)

    GEVPMat.As.ζ[:,2s₂+1:3s₂] = 0.0 * I⁰

    ## ----------------------------------------------------------------------
    ## 3. b (buoyancy) equation (bcs: b = 0 @ z = 0, 1)
    ## ----------------------------------------------------------------------
    GEVPMat.As.b[:,    1:1s₂] = (1.0 * bs.fields.∂ᶻB₀ * I⁰
                                - 1.0 * bs.fields.∂ʸB₀ * H * prob.Dʸᶻᴰ) 

    GEVPMat.As.b[:,1s₂+1:2s₂] = 1.0im * params.k * bs.fields.∂ʸB₀ * H * I⁰

    GEVPMat.As.b[:,2s₂+1:3s₂] = (-1.0 * params.E * Dₙ² 
                                + 1.0im * params.k * bs.fields.U₀ * I⁰) 

    GEVPMat.A = ([GEVPMat.As.w; 
                    GEVPMat.As.ζ; 
                    GEVPMat.As.b]);


    ## Construct the matrix `B`
    cnst = -1.0 
    GEVPMat.Bs.w[:,    1:1s₂] = 1.0cnst * params.ε^2 * D²;
    GEVPMat.Bs.ζ[:,1s₂+1:2s₂] = 1.0cnst * I⁰;
    GEVPMat.Bs.b[:,2s₂+1:3s₂] = 1.0cnst * I⁰;

    GEVPMat.B = ([GEVPMat.Bs.w; 
                GEVPMat.Bs.ζ; 
                GEVPMat.Bs.b])

    return GEVPMat.A, GEVPMat.B
end
nothing #hide


# ### Define the eigenvalue solver
function EigSolver(Op, mf, grid, params, σ₀)

    A, B = construct_matrices(Op, mf, grid, params)

    if params.method == "shift_invert"
        λ, Χ = solve_shift_invert_arnoldi(A, B; σ₀=σ₀, which=:LR, sortby=:R)

    elseif params.method == "krylov"

        λ, Χ = solve_shift_invert_krylov(A, B; σ₀=σ₀, which=:LR)

    elseif params.method == "arnoldi"

        λ, Χ = solve_shift_invert_arnoldi(A, B; σ₀=σ₀, which=:LR)
    end
    ## ======================================================================
    @assert length(λ) > 0 "No eigenvalue(s) found!"

    @printf "||𝓛Χ - λₛℳΧ||₂: %f \n" norm(𝓛 * Χ[:,1] - λ[1] * ℳ * Χ[:,1])

    @printf "largest growth rate : %1.4e%+1.4eim\n" real(λ[1]) imag(λ[1])

    return λ[1], Χ[:,1]
end
nothing #hide

# ### Solving the Stone problem
function solve_Stone1971(k::Float64=0.0)

    Op          = Operator{params.Ny * params.Nz}()
    mf          = MeanFlow{params.Ny * params.Nz}()

    Construct_DerivativeOperator!(diffMatrix, grid, params)
    ImplementBCs_cheb!(Op, diffMatrix, params)

    σ₀   = 0.02 # initial guess for the growth rate
    params.k = k

    λ, Χ = EigSolver(Op, mf, grid, params, σ₀)

    ## Analytical solution of Stone (1971) for the growth rate
    cnst = 1.0 + 1.0 * params.Ri + 5.0 * params.ε^2 * params.k^2 / 42.0
    λₜ = 1.0 / (2.0 * √3.0) * (params.k - 2.0 / 15.0 * params.k^3 * cnst)

    @printf "Analytical solution of Stone (1971) for the growth rate: %f \n" λₜ

    return abs(λ.re - λₜ) < 1e-3

end
nothing #hide

# ## Result
solve_Stone1971(0.1) # growth rate is at k=0.1  
nothing #hide
