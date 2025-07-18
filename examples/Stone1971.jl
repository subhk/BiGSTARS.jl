# ### Linear stability analysis of baroclinic instability of a 2D front based on Stone (1971)
#
# ## Introduction
# Baroclinic instability (BCI) arises when a rotating, stratified fluid has tilted density surfaces, 
# enabling eddies to tap available potential energy and convert it to kinetic energy.
# Stone (1971) investigated non-hydrostatic effects on BCI using Eady’s framework. 
# He found that as the $Ri$ decreases, the wavelength of the most unstable mode increases 
# while the growth rate diminishes relative to predictions from the quasigeostrophic (QG) approximation.
# 
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
# \begin{align}
#  \mathcal{D}^4  = (\mathcal{D}^2 )^2 = \big(\partial_y^2 + (1/\epsilon^2)\partial_z^2 - k^2\big)^2,
# \,\,\,\, \text{and} \,\, \mathcal{D}_h^2 = (\partial_y^2 - k^2).
# \end{align} 
# ```
#
# The eigenfunctions ``\tilde{u}``, ``\tilde{v}`` are related to ``\tilde{w}``, ``\tilde{\zeta}`` by the relations 
# ```math
# \begin{align}
#     -\mathcal{D}_h^2 \tilde{u} &= i k \partial_{z} \tilde{w} + \partial_y \tilde{\zeta},
# \\   
#     -\mathcal{D}_h^2 \tilde{v} &= \partial_{yz} \tilde{w} -  i k \tilde{\zeta}.
# \end{align}
# ```
#
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
# ## Generalized eigenvalue problem (GEVP)
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
#          & \partial_z  & -\mathcal{D}_h^2 
#   \\  
#         -\partial_z U \partial_y - \partial_z  
#           & i k U - E \mathcal{D}^2 & 0 
#  \\ 
#       \partial_z B -  \partial_y B (\mathcal{D}_h^1)^{-1} \partial_{yz}   
#       &  k \partial_y B (\mathcal{D}_h^1)^{-1}  & ikU - E \mathcal{D}^2 
#     \end{bmatrix}, 
# \,\,\,\,\,\,\,
#     B &= \begin{bmatrix}  
#         \epsilon^2 \mathcal{D}^2 & 0 & 0 \\   
#         0 & 1 & 0 \\
#         0 & 0 & 1
#     \end{bmatrix},
# \end{align}
# ```
#
# ## Numerical Implementation
# To implement the above GEVP in a numerical code, we need to actually write 
# following sets of equations: 
#
# ```math
# \begin{align}
#     A &= \begin{bmatrix}
#         \epsilon^2(i k diag(U) \mathcal{D}^{2D} - E \mathcal{D}^{4D}) 
#        & -{D}_z^D & 0_n 
# \\
#         -diag(\partial_z U) \mathcal{D}^y & i k diag(U) - E \mathcal{D}^{2N} & 0_n 
# \\ 
#         diag(\partial_z B) -  diag(\partial_y B) H \mathcal{D}^{yzD} 
#         & k diag(\partial_y B) H & ik diag(U) - E \mathcal{D}^{2N} 
#     \end{bmatrix},
# \,\,\,\,\,\,\,
#     B &= \begin{bmatrix}
#         \epsilon^2 \mathcal{D}^{2D} & 0_n & 0_n \\
#         0_n & I_n & 0_n \\    
#         0_n & 0_n & I_n 
#     \end{bmatrix}.
# \end{align}
# ```
# where $I_n$ is the identity matrix of size $(n \times n)$, where $n=N_y N_z$, $N_y$ and $N_z$
# are the number of grid points in the $y$ and $z$ directions respectively.
# $0_n$ is the zero matrix of size $(n \times n)$.
# The differential operator matrices are given by
#
# ```math
# \begin{align}
# {D}^{2D} &= \mathcal{D}_y^2 \otimes {I}_z + {I}_y \otimes \mathcal{D}_z^{2D} - k^2 {I}_n,
# \\
# {D}^{2N} &= \mathcal{D}_y^2 \otimes {I}_z + {I}_y \otimes \mathcal{D}_z^{2N} - k^2 {I}_n,
# \\
#  {D}^{4D} &= \mathcal{D}_y^4 \otimes {I}_z
#    + {I}_y \otimes \mathcal{D}_z^{4D} + k^4 {I}_n - 2 k^2 {D}_y^2 \otimes {I}_z
#    - 2 k^2 {I}_y \otimes {D}_z^{2D} + 2 {D}_y^2 \otimes {D}_z^{2D}
# \end{align}
# ```
# where $\otimes$ is the Kronecker product. ${I}_y$ and ${I}_z$ are 
# identity matrices of size $(N_y \times N_y)$ and $(N_z \times N_z)$ respectively, 
# and ${I}={I}_y \otimes {I}_z$. The superscripts $D$ and $N$ in the operator matrices
# denote the type of boundary conditions applied ($D$ for Dirichlet or $N$ for Neumann).
# $\mathcal{D}_y$, $\mathcal{D}_y^2$ and $\mathcal{D}_y^3$ are the first, second and third order
# Fourier differentiation matrix of size of $(N_y \times N_y)$. 
# $\mathcal{D}_z$, $\mathcal{D}_z^2$ and $\mathcal{D}_z^4$ are the first, second and fourth order
# Chebyshev differentiation matrix of size of $(N_z \times N_z)$.
#
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
    Ny::Int64           = 20            # no. of y-grid points
    Nz::Int64           = 20            # no. of z-grid points
    w_bc::String        = "rigid_lid"   # boundary condition for vertical velocity
    ζ_bc::String        = "free_slip"   # boundary condition for vertical vorticity
    b_bc::String        = "zero_flux"   # boundary condition for buoyancy
    eig_solver::String  = "arpack"      # eigenvalue solver
end
nothing #hide

# ## Basic state
function basic_state(grid, params)
    
    Y, Z = ndgrid(grid.y, grid.z)
    Y    = transpose(Y)
    Z    = transpose(Z)

    ## Define the basic state
    B₀   = @. params.Ri * Z - Y          # buoyancy
    U₀   = @. 1.0 * Z - 0.5 * params.H   # along-front velocity

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
nothing #hide

# ## Constructing Generalized EVP
function generalized_EigValProb(prob, grid, params)

    bs = basic_state(grid, params)

    N  = params.Ny * params.Nz
    I⁰ = sparse(Matrix(1.0I, N, N))  # Identity matrix
    s₁ = size(I⁰, 1); 
    s₂ = size(I⁰, 2);

    ## the horizontal Laplacian operator:  ∇ₕ² = ∂ʸʸ - k²
    ∇ₕ² = SparseMatrixCSC(Zeros(N, N))
    ∇ₕ² = (1.0 * prob.D²ʸ - 1.0 * params.k^2 * I⁰)

    ## inverse of the horizontal Laplacian operator
    H = inverse_Lap_hor(∇ₕ²)

    ## Construct the 4th order derivative
    D⁴ᴰ = (1.0 * prob.D⁴ʸ 
        + 1.0/params.ε^4 * prob.D⁴ᶻᴰ 
        + 1.0 * params.k^4 * I⁰ 
        - 2.0 * params.k^2 * prob.D²ʸ 
        - 2.0/params.ε^2 * params.k^2 * prob.D²ᶻᴰ
        + 2.0/params.ε^2 * prob.D²ʸ²ᶻᴰ)
        
    ## Construct the 2nd order derivative
    D²ᴰ = (1.0/params.ε^2 * prob.D²ᶻᴰ + 1.0 * ∇ₕ²) # with Dirchilet BC
    D²ᴺ = (1.0/params.ε^2 * prob.D²ᶻᴺ + 1.0 * ∇ₕ²) # with Neumann BC

    ## See `Numerical Implementation' section for the theory
    ## ──────────────────────────────────────────────────────────────────────────────
    ## 1) Now define your 3×3 block-rows in a NamedTuple of 3-tuples
    ## ──────────────────────────────────────────────────────────────────────────────
    ## Construct the matrix `A`
    Ablocks = (
        w = (  # w-equation: [z⁴+z²], [∂ᶻ Neumann], [–∇ₕ²]
                sparse(complex.(-params.E * D⁴ᴰ + 1.0im * params.k * bs.fields.U₀ * D²ᴰ) * params.ε^2),
                sparse(complex.(prob.Dᶻᴺ)),
                sparse(complex.(-∇ₕ²))
        ),
        ζ = (  # ζ-equation: [∂ᶻU + Dirichlet], [kU–Ek], [zero]
                sparse(complex.(-bs.fields.∂ᶻU₀ * prob.Dʸ - prob.Dᶻᴰ)),
                sparse(complex.(1.0im *params.k * bs.fields.U₀ * I⁰ - params.E * D²ᴺ)),
                spzeros(ComplexF64, s₁, s₂)
        ),
        b = (  # b-equation: [∂ᶻB – Dʸᶻᴰ], [k∂ʸB], [–Ek + kU]
                sparse(complex.(bs.fields.∂ᶻB₀ * I⁰ - bs.fields.∂ʸB₀ * H * prob.Dʸᶻᴰ)),
                sparse(1.0im * params.k * bs.fields.∂ʸB₀ * H * I⁰),
                sparse(-params.E * D²ᴺ + 1.0im * params.k * bs.fields.U₀ * I⁰)
        )
    )

    ## Construct the matrix `A`
    Bblocks = (
        w = (  # w-equation mass: [–ε²∂²], zero, zero
                sparse(-params.ε^2 * D²ᴰ),
                spzeros(Float64, s₁, s₂),
                spzeros(Float64, s₁, s₂)
        ),
        ζ = (  # ζ-equation mass: zero, [–I], zero
                spzeros(Float64, s₁, s₂),
                sparse(-I⁰),
                spzeros(Float64, s₁, s₂)
        ),
        b = (  # b-equation mass: zero, zero, [–I]
                spzeros(Float64, s₁, s₂),
                spzeros(Float64, s₁, s₂),
                sparse(-I⁰)
        )
    )

    ## ──────────────────────────────────────────────────────────────────────────────
    ## 2) Assemble in beautiful line
    ## ──────────────────────────────────────────────────────────────────────────────
    gevp = GEVPMatrices(Ablocks, Bblocks)

    ## ──────────────────────────────────────────────────────────────────────────────
    ## 3) And now you have exactly:
    ##    gevp.A, gevp.B                    → full sparse matrices
    ##    gevp.As.w, gevp.As.ζ, gevp.As.b   → each block-row view of matrix A
    ##    gevp.Bs.w, gevp.Bs.ζ, gevp.Bs.b   → each block-row view of matrix B
    ## ──────────────────────────────────────────────────────────────────────────────

    return gevp.A, gevp.B
end
nothing #hide

# ## Eigenvalue solver
function EigSolver(prob, grid, params, σ₀)

    A, B = generalized_EigValProb(prob, grid, params)

    if params.eig_solver == "arpack"
        λ, Χ = solve_shift_invert_arnoldi(A, B; σ₀=σ₀, which=:LR, sortby=:R)

    elseif params.eig_solver == "krylov"

        λ, Χ = solve_shift_invert_krylov(A, B; σ₀=σ₀, which=:LR)

    elseif params.eig_solver == "arnoldi"

        λ, Χ = solve_shift_invert_arnoldi(A, B; σ₀=σ₀, which=:LR)
    end
    ## ======================================================================
    @assert length(λ) > 0 "No eigenvalue(s) found!"

    @printf "largest growth rate : %1.4e%+1.4eim\n" real(λ[1]) imag(λ[1])

    return λ[1], Χ[:,1]
end
nothing #hide

# ## Solving the problem
function solve_Stone1971(k::Float64)

    ## Calling problem parameters
    params = Params{Float64}()

    ## Construct grid and derivative operators
    grid  = TwoDGrid(params)

    ## Construct the necesary operator
    ops  = OperatorI(params)
    prob = Problem(grid, ops)

    ## update the wavenumber
    params.k = k

    ## initial guess for the growth rate
    σ₀   = 0.02 

    λ, Χ = EigSolver(prob, grid, params, σ₀)

    ## Analytical solution of Eady (1949) for the growth rate
    μ  = 1.0 * params.k * √params.Ri
    λₜ = 1.0/√params.Ri * √( (coth(0.5μ) - 0.5μ)*(0.5μ - tanh(0.5μ)) )

    @printf "Analytical solution of Eady (1949) for the growth rate: %f \n" λₜ

    return abs(λ.re - λₜ) < 1e-3

end
nothing #hide

# ## Result
solve_Stone1971(0.1) # growth rate is at k=0.1  
nothing #hide
