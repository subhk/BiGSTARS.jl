""""
This code finds critical Rayleigh number for rotating Rayleigh Benrad Convection (rRBC)
where the domain is periodic in y-direction.
The code is benchmarked against Chandrashekar's theoretical results.
# Hydrodynamic and hydromagnetic stability by S. Chandrasekhar, 1961 (page no-95)
parameter: Ek (Ekman number) = 10⁻⁴
eigenvalue: critical modified Rayleigh number (Raᶜ) = 189.7
"""
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
    L::T                = 2π      # horizontal domain size
    H::T                = 1.0       # vertical   domain size
    E::T                = 1.0e-4    # inverse of Reynolds number 
    k::T                = 0.0       # x-wavenumber
    Ny::Int64           = 120       # no. of y-grid points
    Nz::Int64           = 30        # no. of Chebyshev points
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
nothing #hide


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
    #@assert norm(∇ₕ² * H - I⁰) ≤ 1.0e-2 "difference in L2-norm should be small"

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
    # ──────────────────────────────────────────────────────────────────────────────
    # 1) Now define your 3×3 block-rows in a NamedTuple of 3-tuples
    # ──────────────────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────────────
    # 2) Assemble in beautiful line
    # ──────────────────────────────────────────────────────────────────────────────
    gevp = GEVPMatrices(Ablocks, Bblocks)


    # ──────────────────────────────────────────────────────────────────────────────
    # 3) And now you have exactly:
    #    gevp.A, gevp.B                    → full sparse matrices
    #    gevp.As.w, gevp.As.ζ, gevp.As.b   → each block-row view
    #    gevp.Bs.w, gevp.Bs.ζ, gevp.Bs.b
    # ──────────────────────────────────────────────────────────────────────────────

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


# ### Solving the rRBC problem
function solve_rRBC(k::Float64)

    params = Params{Float64}()

    # ### Construct grid and derivative operators
    grid  = TwoDGrid(params)

    # ### Construct the necesary operator
    ops  = OperatorI(params)
    prob = Problem(grid, ops, params)

    params.k = k

    σ₀   = 0.0 # initial guess for the growth rate
    params.k = k

    λ, Χ = EigSolver(prob, grid, params, σ₀)

    # Theoretical results from Chandrashekar (1961)
    λₜ = 189.7 
    @printf "Analytical solution of critical Ra: %1.4e \n" λₜ 

    return abs(real(λ) - λₜ)/λₜ < 1e-4

end
nothing #hide

# # ## Result
solve_rRBC(0.0) # growth rate is at k=0.1  
nothing #hide


