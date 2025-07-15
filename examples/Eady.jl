# ### Linear stability analysis of baroclinic instability of a 2D front based on Eady (1949)
#
# ## Introduction
# Eady (1949) showed that in a uniformly sheared, stratified layer between two rigid lids on 
# an ``f``-plane, two counter-propagating Rossby edge waves can phase lock and convert available potential energy 
# into kinetic energy, producing baroclinic eddies that grow fastest at wavelengths about 
# four deformation radii and on timescales of a few days.
#
# ## Basic state
# The basic state is given by
# ```math
# \begin{align}
#     B(y, z) &= Ri z - y, \\
#     U(y, z) &= z - {1}/{2},
# \end{align}
# ```
# where ``Ri`` is the Richardson number, and $N^2 = Ri$ is the stratification.
#
# ## Governing equations
# The non-dimensional form of the linearized version of the QG PV perturbation equation under 
# the $f$-plane approximation can be expressed as,
# ```math
# \begin{align}
#     \frac{\partial q^\text{qg}}{\partial t} + U \frac{\partial q^\text{qg}}{\partial x} + \frac{\partial \psi}{\partial x}
#     \frac{\partial Q^\text{qg}}{\partial y} = E \, \nabla_h^2 q^\text{qg}, 
# \,\,\,\,\,\,\  \text{for} \,\,\, 0 < z <1, 
# \end{align}
# ```
# where $q^\text{qg}$ is the perturbation QG PV, and it is defined as 
# ```math
# \begin{align}
#     q^\text{qg} = \nabla_h^2 \psi^\text{qg} + 
#     \frac{\partial}{\partial z}
#     \left(\frac{1}{N^2} \frac{\partial \psi^\text{qg}}{\partial z}\right),
# \end{align}
# ```
#
# The variable $\psi^\text{qg}$ describes the QG perturbation streamfunction with 
# $u^\text{qg}=-\partial_y \psi^\text{qg}$ and $v^\text{qg}=\partial_x \psi^\text{qg}$. 
# The variable $Q^\text{qg}$ describes the QG PV of the basic state, which is defined as \citep{pedlosky2013geophysical}
# ```math
# \begin{align}
#     Q^\text{qg} = -\frac{\partial U}{\partial y} + \frac{\partial}{\partial z}\left(\frac{B}{N^2} \right),
# \end{align}
# ```
# and the cross-front gradient of $Q^\text{qg}$ is defined as
# ```math
# \begin{align}
#     \frac{\partial Q^\text{qg}}{\partial y} = - \frac{\partial}{\partial z}\left(\frac{\partial_z U}{N^2} \right).
# \end{align}
# ```
#
# The linearized perturbation buoyancy equation at the top and the bottom boundary is
# ```math
# \begin{align}
#     \frac{\partial b^\text{qg}}{\partial t} + U \frac{\partial b^\text{qg}}{\partial x} 
#       + \frac{\partial \psi^\text{qg}}{\partial x}
#     \frac{\partial B}{\partial y} = 0,
#     \,\,\,\,\,\,\ \text{at} \, z=0 \,\ \text{and} \,\, 1,
# \end{align}
# ```
# where $b^\text{qg}=\partial_z \psi^\text{qg}$.
#
# ## Normal-mode solutions
# Next, we seek normal-mode solutions for $\psi^\text{qg}$ and $q^\text{qg}$ in the form of 
# ```math
# \begin{align}
#     [\psi^\text{qg}, q^\text{qg}] = \mathfrak{R}\big([\widetilde{\psi}^\text{qg}, 
#   \widetilde{q}^\text{qg}] \big)(y, z) e^{i kx-\sigma t},    
# \end{align}
# ```
# where $\widetilde{\psi}^\text{qg}$, $\widetilde{q}^\text{qg}$ are the eigenfunctions of $\psi^\text{qg}$ and $q^\text{qg}$, respectively.
# In terms of streamfunction $\psi^\text{qg}$, 
# ```math
# \begin{align}
#     [(\sigma + i k U) - E] \mathscr{L}\widetilde{\psi}^\text{qg} 
#   + i k \partial_y Q^\text{qg} \widetilde{\psi}^\text{qg} &= 0, \,\,\,\,\  \text{for} \,\, 0 < z <1, 
# \\
#     (\sigma + i k U_{-})\partial_z \widetilde{\psi}^\text{qg}_{-} 
#   + i k \partial_y B_{-} \widetilde{\psi}^\text{qg}_{-} &= 0, \,\,\,\,\, \text{at} \,\, z = 0,
# \\
#     (\sigma + i k U_{+})\partial_z \widetilde{\psi}^\text{qg}_{+} 
#   + i k \partial_y B_{+} \widetilde{\psi}^\text{qg}_{+} &= 0, \,\,\,\,\, \text{at} \,\, z = 1,
# \end{align}
# ```
# where $\mathscr{L}$ is a linear operator, and is defined as
# $\mathscr{L} \equiv \mathcal{D}_h^2 + 1/N^2 \partial_z^2$,
# where $\mathcal{D}_h^2 = (\partial_y^2 - k^2)$. 
# The subscripts $-,+$ denote the values of the fields at $z=0$ and $z=1$, respectively. 
#
# ## Generalized eigenvalue problem
# The above set of equations can be cast into a generalized eigenvalue problem 
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
#         ik U \mathcal{D}_0^2 
#        + ik \partial_y Q 
#       - E \mathcal{D}_h^2 \mathcal{D}_0^2 
#   \end{bmatrix}, 
#   \,\,\,\,\,\,\,
#     B &= \begin{bmatrix}
#       - \mathcal{D}_0^2
#     \end{bmatrix},
# \end{align}
# ```
# where $D_0^2 = \mathcal{D}_h^2 + (1/N^2) \mathcal{D}_z^2$. 
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

using BiGSTARS
using BiGSTARS: AbstractParams
using BiGSTARS: Problem, OperatorI, TwoDGrid

# ### Define the parameters
@with_kw mutable struct Params{T} <: AbstractParams
    L::T                = 1.0         # horizontal domain size
    H::T                = 1.0         # vertical domain size
    Ri::T               = 1.0         # the Richardson number
    k::T                = 0.1         # x-wavenumber
    E::T                = 1.0e-9     # Ekman number 
    Ny::Int64           = 50          # no. of y-grid points
    Nz::Int64           = 30          # no. of z-grid points
    eig_solver::String  = "krylov"      # eigenvalue solver
end
nothing #hide

# ### Define the basic state
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

    return bs, deriv
end


# ### Constructing Generalized EVP
function generalized_EigValProb(prob, grid, params)

    bs, deriv = basic_state(grid, params)

    N  = params.Ny * params.Nz
    I⁰ = sparse(Matrix(1.0I, N, N)) 
    Iʸ = sparse(Matrix(1.0I, params.Ny, params.Ny)) 
    s₁ = size(I⁰, 1); 
    s₂ = size(I⁰, 2);

    ## the horizontal Laplacian operator
    ∇ₕ² = SparseMatrixCSC(Zeros(N, N))
    ∇ₕ² = (1.0 * prob.D²ʸ - 1.0 * params.k^2 * I⁰)

    # some quanntities required later
    bs_∂ᶻB₀⁻¹  = @. 1.0/deriv.∂ᶻB₀
    bs_∂ᶻB₀⁻²  = @. 1.0/(deriv.∂ᶻB₀ * deriv.∂ᶻB₀) 
    
    ∂ᶻB₀⁻¹::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
    ∂ᶻB₀⁻²::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
    ∂ʸQ₀::Array{Float64, 2}   = SparseMatrixCSC(Zeros(N, N)) # PV gradient is zero

    ## converting to matrics 
    ∂ᶻB₀⁻¹[diagind(∂ᶻB₀⁻¹)] = bs_∂ᶻB₀⁻¹
    ∂ᶻB₀⁻²[diagind(∂ᶻB₀⁻²)] = bs_∂ᶻB₀⁻²

    ## definition of perturbation PV, q = D₂³ᵈ{ψ}
    D₂³ᵈ = (1.0 * ∇ₕ²
            + 1.0  * ∂ᶻB₀⁻¹ * prob.D²ᶻ
            - 1.0  * bs.fields.∂ᶻᶻB₀  * ∂ᶻB₀⁻² * prob.Dᶻ)

    ## Construct the matrix `A`
    ## ──────────────────────────────────────────────────────────────────────────────
    ## 1) Now define your 3×3 block-rows in a NamedTuple of 3-tuples
    ## ──────────────────────────────────────────────────────────────────────────────
    ## Construct the matrix `A`
    Ablocks = (
        ψ = (  # ψ-equation
                sparse(1.0im * params.k * bs.fields.U₀ * D₂³ᵈ
                    + 1.0im * params.k * ∂ʸQ₀ * I⁰
                    - 1.0 * params.E * ∇ₕ² * D₂³ᵈ
                ) 
        ),
    )

    ## Construct the matrix `B`
    Bblocks = (
        ψ = (  # ψ-equation: [-D₂³ᵈ]
                sparse(-D₂³ᵈ)
        ),
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

    B = SparseMatrixCSC(Zeros{ComplexF64}(s₁, s₂))
    C = SparseMatrixCSC(Zeros{ Float64  }(s₁, s₂))

    ## ──────────────────────────────────────────────────────────────────────────────
    ## 4) Implementing boundary conditions
    ## ──────────────────────────────────────────────────────────────────────────────
    _, zi = ndgrid(1:1:params.Ny, 1:1:params.Nz)
    zi    = transpose(zi);
    zi    = zi[:];
    bcᶻᵇ  = findall( x -> (x==1),         zi )
    bcᶻᵗ  = findall( x -> (x==params.Nz), zi )

    ## Implementing boundary condition for 𝓛 matrix in the z-direction: 
    B[:,1:1s₂] = 1.0im * params.k * bs.fields.U₀ * prob.Dᶻ 
                - 1.0im * params.k * bs.fields.∂ᶻU₀ * I⁰
    
    ## Bottom boundary condition @ z=0  
    @. gevp.A[bcᶻᵇ, :] = B[bcᶻᵇ, :]
    
    ## Top boundary condition @ z = 1
    @. gevp.A[bcᶻᵗ, :] = B[bcᶻᵗ, :]

    ## Implementing boundary condition for ℳ matrix in the z-direction: 
    C[:,1:1s₂] = -1.0 * prob.Dᶻ

    ## Bottom boundary condition @ z=0  
    @. gevp.B[bcᶻᵇ, :] = C[bcᶻᵇ, :]

    ## Top boundary condition @ z = 1
    @. gevp.B[bcᶻᵗ, :] = C[bcᶻᵗ, :]

    return gevp.A, gevp.B
end
nothing #hide


# ### Define the eigenvalue solver
function EigSolver(prob, grid, params, σ₀)

    A, B = generalized_EigValProb(prob, grid, params)

    if params.eig_solver == "arpack"

        λ, Χ = solve_shift_invert_arpack(A, B; σ₀=σ₀, which=:LR, sortby=:R)

    elseif params.eig_solver == "krylov"

        λ, Χ = solve_shift_invert_krylov(A, B; σ₀=σ₀, which=:LR, sortby=:R)

    elseif params.eig_solver == "arnoldi"

        λ, Χ = solve_shift_invert_arnoldi(A, B; σ₀=σ₀, which=:LR, sortby=:R)
    end
    ## ======================================================================
    @assert length(λ) > 0 "No eigenvalue(s) found!"

    @printf "||AΧ - λBΧ||₂: %f \n" norm(A * Χ[:,1] - λ[1] * B * Χ[:,1])

    print_evals(λ)

    return λ[1], Χ[:,1]
end
nothing #hide


# ### Solving the Stone problem
function solve_Eady(k::Float64)

    params = Params{Float64}()

    # ### Construct grid and derivative operators
    grid  = TwoDGrid(params)

    # ### Construct the necesary operator
    ops  = OperatorI(params)
    prob = Problem(grid, ops)

    params.k = k

    σ₀   = 0.02 # initial guess for the growth rate

    λ, Χ = EigSolver(prob, grid, params, σ₀)

    ## Analytical solution of Eady (1949) for the growth rate
    μ  = 1.0 * params.k * √params.Ri
    λₜ = 1.0/√params.Ri * √( (coth(0.5μ) - 0.5μ)*(0.5μ - tanh(0.5μ)) )

    @printf "Analytical solution of Stone (1971) for the growth rate: %f \n" λₜ

    return abs(λ.re - λₜ) < 1e-3

end
nothing #hide

# # ## Result
solve_Eady(0.1) # growth rate is at k=0.1  
nothing #hide

