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
using ModelingToolkit
using NonlinearSolve

using BiGSTARS

# ### Define the parameters
@with_kw mutable struct Params{T<:Real} @deftype T
    L::T                = 1.0         # horizontal domain size
    H::T                = 1.0         # vertical domain size
    Ri::T               = 1.0         # the Richardson number
    k::T                = 0.0         # x-wavenumber
    E::T                = 1.0e-16     # Ekman number 
    Ny::Int64           = 50          # no. of y-grid points
    Nz::Int64           = 30          # no. of z-grid points
    w_bc::String        = "rigid_lid"   # boundary condition for vertical velocity
    ζ_bc::String        = "free_slip"   # boundary condition for vertical vorticity
    b_bc::String        = "zero_flux"   # boundary condition for buoyancy
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

    return bs
end

function BasicState!(diffMatrix, mf, grid, params)
    Y, Z = ndgrid(grid.y, grid.z)
    Y    = transpose(Y)
    Z    = transpose(Z)

    ## imposed buoyancy profile
    B₀      = @. 1.0params.Ri * Z - Y  
    ∂ʸB₀    = - 1.0 .* ones(size(Y))  
    ∂ᶻB₀    = 1.0params.Ri .* ones(size(Y))  
    ∂ᶻᶻB₀   = zeros(size(Y))  

    ∂ᶻB₀⁻¹  = @. 1.0/∂ᶻB₀ 
    ∂ᶻB₀⁻²  = @. 1.0/(∂ᶻB₀ * ∂ᶻB₀) 

    ## along-front profile 
    U₀      = @. 1.0 * Z - 0.5
    ∂ᶻU₀    = ones(size(Y))  
    ∂ʸU₀    = zeros(size(Y))  

    ## y-gradient of the QG PV
    ∂ʸQ₀    = zeros(size(Y))  

      B₀  = B₀[:]
      U₀  = U₀[:]
    ∂ʸB₀  = ∂ʸB₀[:] 
    ∂ᶻB₀  = ∂ᶻB₀[:] 
    ∂ᶻU₀  = ∂ᶻU₀[:]
    ∂ʸU₀  = ∂ʸU₀[:] 

    ∂ʸQ₀  = ∂ʸQ₀[:] 

    ∂ᶻᶻB₀ = ∂ᶻᶻB₀[:]

    ∂ᶻB₀⁻¹ = ∂ᶻB₀⁻¹[:] 
    ∂ᶻB₀⁻² = ∂ᶻB₀⁻²[:]

    mf.B₀[diagind(mf.B₀)] = B₀
    mf.U₀[diagind(mf.U₀)] = U₀

    mf.∇ʸU₀[diagind(mf.∇ʸU₀)]   = ∂ʸU₀
    mf.∇ᶻU₀[diagind(mf.∇ᶻU₀)]   = ∂ᶻU₀

    mf.∇ʸQ₀[diagind(mf.∇ʸQ₀)]   = ∂ʸQ₀

    mf.∇ᶻB₀⁻¹[diagind(mf.∇ᶻB₀⁻¹)] = ∂ᶻB₀⁻¹
    mf.∇ᶻB₀⁻²[diagind(mf.∇ᶻB₀⁻²)] = ∂ᶻB₀⁻²

    mf.∇ᶻᶻB₀[diagind(mf.∇ᶻᶻB₀)] = ∂ᶻᶻB₀

    return nothing
end

function construct_matrices(Op, mf, params)
    N  = params.Ny * params.Nz
    I⁰ = sparse(Matrix(1.0I, N, N)) 
    s₁ = size(I⁰, 1); s₂ = size(I⁰, 2)

    ## allocating memory for the LHS and RHS matrices
    𝓛 = SparseMatrixCSC(Zeros{ComplexF64}(s₁, s₂))
    ℳ = SparseMatrixCSC(Zeros{ Float64  }(s₁, s₂))

    B = SparseMatrixCSC(Zeros{ComplexF64}(s₁, s₂))
    C = SparseMatrixCSC(Zeros{ Float64  }(s₁, s₂))

    ## -------------------- construct matrix  ------------------------
    ## lhs of the matrix (size := 2 × 2)
    ## eigenvectors: [ψ]
    ∇ₕ² = SparseMatrixCSC(Zeros{Float64}(N, N))
    ∇ₕ² = (1.0 * Op.𝒟²ʸ - 1.0 * params.k^2 * I⁰)

    ## definition of perturbation PV, q = D₂³ᵈ{ψ}
    D₂³ᵈ = (1.0 * ∇ₕ²
            + 1.0  * mf.∇ᶻB₀⁻¹ * Op.𝒟²ᶻ
            - 1.0  * mf.∇ᶻᶻB₀  * mf.∇ᶻB₀⁻² * Op.𝒟ᶻ)

    ## 1. ψ equation
    𝓛[:,1:1s₂] = (1.0im * params.k * mf.U₀   * D₂³ᵈ
                + 1.0im * params.k * mf.∇ʸQ₀ * I⁰ #)
                - 1.0 * params.E * ∇ₕ² * D₂³ᵈ)

    ℳ[:,1:1s₂] = -1.0 * D₂³ᵈ

    ## Implementing boundary conditions
    _, zi = ndgrid(1:1:params.Ny, 1:1:params.Nz)
    zi    = transpose(zi);
    zi    = zi[:];
    bcᶻᵇ  = findall( x -> (x==1),         zi )
    bcᶻᵗ  = findall( x -> (x==params.Nz), zi )

    ## Implementing boundary condition for 𝓛 matrix in the z-direction: 
    B[:,1:1s₂] = 1.0im * params.k * mf.U₀ * Op.𝒟ᶻ - 1.0im * params.k * mf.∇ᶻU₀ * I⁰
    
    ## Bottom boundary condition @ z=0  
    @. 𝓛[bcᶻᵇ, :] = B[bcᶻᵇ, :]
    
    ## Top boundary condition @ z = 1
    @. 𝓛[bcᶻᵗ, :] = B[bcᶻᵗ, :]

    ## Implementing boundary condition for ℳ matrix in the z-direction: 
    C[:,1:1s₂] = -1.0 * Op.𝒟ᶻ

    ## Bottom boundary condition @ z=0  
    @. ℳ[bcᶻᵇ, :] = C[bcᶻᵇ, :]

    ## Top boundary condition @ z = 1
    @. ℳ[bcᶻᵗ, :] = C[bcᶻᵗ, :]

    return 𝓛, ℳ
end
nothing #hide


# ### Define the eigenvalue solver
function EigSolver(Op, mf, grid, params, σ₀)

    𝓛, ℳ = construct_matrices(Op, mf, params)
    
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
    
    #@printf "largest growth rate : %1.4e%+1.4eim\n" real(λₛ[1]) imag(λₛ[1])

    return λₛ[1] #, Χ[:,1]
end
nothing #hide

# ### Solving the Eady (1949) problem
function solve_Eady1949(k::Float64=0.0)
    params      = Params{Float64}(k=0.5)
    grid        = TwoDimGrid{params.Ny,  params.Nz}()
    diffMatrix  = ChebMarix{ params.Ny,  params.Nz}()
    Op          = Operator{params.Ny * params.Nz}()
    mf          = MeanFlow{params.Ny * params.Nz}()

    Construct_DerivativeOperator!(diffMatrix, grid, params)
    ImplementBCs_cheb!(Op, diffMatrix, params)

    BasicState!(diffMatrix, mf, grid, params)

    σ₀   = 0.01 # initial guess for the growth rate
    params.k = k
    
    λₛ = EigSolver(Op, mf, grid, params, σ₀)

    ## Analytical solution of Eady (1949) for the growth rate
    μ  = 1.0 * params.k * √params.Ri
    λₛₜ = 1.0/√params.Ri * √( (coth(0.5μ) - 0.5μ)*(0.5μ - tanh(0.5μ)) )

    @printf "Analytical solution of Eady (1949) for the growth rate: %f \n" λₛₜ

    return abs(λₛ.re - λₛₜ) < 1e-3
    
end
nothing #hide

# ## Result
solve_Eady1949(0.1) # growth rate is at k=0.1  
nothing #hide


