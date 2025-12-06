# Baroclinic instability of a 2D front 
#
# This example solves the linear stability problem of a 2D front based on the
# analytical solution of Stone (1971). The problem is defined in the `Stone1971.jl` file.
# The problem is solved using the `EigSolver` function, which constructs the
# generalized eigenvalue problem and solves it using the `Krylov` method.
#
#
# ## Load required packages
using LazyGrids
using LinearAlgebra
using Printf
using SparseArrays
using Parameters
using Test
using JLD2
using BiGSTARS
using BiGSTARS: AbstractParams
using BiGSTARS: Problem, OperatorI, TwoDGrid 

# ## Define the parameters
@with_kw mutable struct Params{T} <: AbstractParams
    L::T                = 1.0           # horizontal domain size
    H::T                = 1.0           # vertical domain size
    Ri::T               = 1.0           # the Richardson number 
    ε::T                = 0.1           # aspect ratio ε ≡ H/L
    k::T                = 0.1           # along-front wavenumber
    E::T                = 1.0e-8        # the Ekman number 
    Ny::Int64           = 50            # no. of y-grid points
    Nz::Int64           = 18            # no. of z-grid points
    w_bc::String        = "rigid_lid"   # boundary condition for vertical velocity
    ζ_bc::String        = "free_slip"   # boundary condition for vertical vorticity
    b_bc::String        = "zero_flux"   # boundary condition for buoyancy
    eig_solver::String  = "krylov"      # eigenvalue solver
end
nothing #hide

# ## Define the basic state
function basic_state(grid, params)
    
    Y, Z = ndgrid(grid.y, grid.z)
    Y    = transpose(Y)
    Z    = transpose(Z)

    ## Define the basic state
    B   = @. 1.0 * params.Ri * Z - Y    # buoyancy
    U   = @. 1.0 * Z - 0.5 * params.H   # along-front velocity

    ## Calculate all the 1st, 2nd and yz derivatives in 2D grids
    bs = compute_derivatives(U, B, grid.y; grid.Dᶻ, grid.D²ᶻ, gridtype = :All)
    precompute!(bs; which = :All)   # eager cache, returns bs itself
    @assert bs.U === U              # originals live in the same object
    @assert bs.B === B

    return bs
end

# ## Constructing Generalized EVP
function generalized_EigValProb(prob, grid, params)

    bs = basic_state(grid, params)

    N  = params.Ny * params.Nz
    I⁰ = sparse(Matrix(1.0I, N, N)) 
    s₁ = size(I⁰, 1); 
    s₂ = size(I⁰, 2);

    ## the horizontal Laplacian operator
    ∇ₕ² = (1.0 * prob.D²ʸ - 1.0 * params.k^2 * I⁰)

    ## inverse of the horizontal Laplacian operator
    H = sparse(inverse_Lap_hor(∇ₕ²))
    #@assert norm(∇ₕ² * H - I⁰) ≤ 1.0e-2 "difference in L2-norm should be small"

    ## Construct the 4th order derivative
    D⁴ᴰ = (1.0 * prob.D⁴ʸ 
        + 1.0/params.ε^4 * prob.D⁴ᶻᴰ 
        + 1.0 * params.k^4 * I⁰ 
        - 2.0 * params.k^2 * prob.D²ʸ 
        - 2.0/params.ε^2 * params.k^2 * prob.D²ᶻᴰ
        + 2.0/params.ε^2 * prob.D²ʸ²ᶻᴰ)
        
    ## Construct the 2nd order derivative
    D²ᴰ = (1.0/params.ε^2 * prob.D²ᶻᴰ + 1.0 * ∇ₕ²)
    D²ᴺ = (1.0/params.ε^2 * prob.D²ᶻᴺ + 1.0 * ∇ₕ²)

    ## Construct the matrix `A`
    # ──────────────────────────────────────────────────────────────────────────────
    # 1) Now define your 3×3 block-rows in a NamedTuple of 3-tuples
    # ──────────────────────────────────────────────────────────────────────────────
    ## Construct the matrix `A`
    Ablocks = (
        w = (  # w-equation: [ε²(ikDiagM(U) - ED⁴ᴰ)], [Dᶻᴺ], [–∇ₕ²]
                sparse(complex.(-params.E * D⁴ᴰ + 1.0im * params.k * DiagM(bs.U) * D²ᴰ) * params.ε^2),
                sparse(complex.(prob.Dᶻᴺ)),
                sparse(complex.(-∇ₕ²))
        ),
        ζ = (  # ζ-equation: [DiagM(∂ᶻU)Dʸ - Dᶻᴰ], [kDiagM(U) – ED²ᴺ], [zero]
                sparse(complex.(-DiagM(bs.∂ᶻU) * prob.Dʸ - prob.Dᶻᴰ)),
                sparse(complex.(1.0im *params.k * DiagM(bs.U) * I⁰ - params.E * D²ᴺ)),
                spzeros(ComplexF64, s₁, s₂)
        ),
        b = (  # b-equation: [DiagM(∂ᶻB) – DiagM(∂ʸB) H Dʸᶻᴰ], [ikDiagM(∂ʸB)], [–ED²ᴺ + ikDiagM(U)]
                sparse(complex.(DiagM(bs.∂ᶻB) * I⁰ - DiagM(bs.∂ʸB) * H * prob.Dʸᶻᴰ)),
                sparse(1.0im * params.k * DiagM(bs.∂ʸB) * H),
                sparse(-params.E * D²ᴺ + 1.0im * params.k * DiagM(bs.U))
        )
    )

    ## Construct the matrix `A`
    Bblocks = (
        w = (  # w-equation mass: [–ε²∂²], [zero], [zero]
                sparse(-params.ε^2 * D²ᴰ),
                spzeros(Float64, s₁, s₂),
                spzeros(Float64, s₁, s₂)
        ),
        ζ = (  # ζ-equation mass: [zero], [–I], [zero]
                spzeros(Float64, s₁, s₂),
                sparse(-I⁰),
                spzeros(Float64, s₁, s₂)
        ),
        b = (  # b-equation mass: [zero], [zero], [–I]
                spzeros(Float64, s₁, s₂),
                spzeros(Float64, s₁, s₂),
                sparse(-I⁰)
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

# ## Define the eigenvalue solver
function EigSolver(prob, grid, params, σ₀)

    A, B = generalized_EigValProb(prob, grid, params)

    ## Construct the eigenvalue solver
    ## Methods available: :Krylov, :Arnoldi (by default), :Arpack
    ## Here we are looking for largest growth rate (real part of eigenvalue)
    solver = EigenSolver(A, B; σ₀=σ₀, method=:Krylov, nev=1, which=:LR, sortby=:R)
    solve!(solver)
    λ, Χ = get_results(solver)
    print_summary(solver)

    ## Print the largest growth rate
    @printf "largest growth rate : %1.4e%+1.4eim\n" real(λ[1]) imag(λ[1])

    return λ[1], Χ[:,1]
end
nothing #hide

# ## Solving the Stone problem
function solve_Stone1971(k::Float64)

    params = Params{Float64}()

    # ### Construct grid and derivative operators
    grid  = TwoDGrid(params)

    # ### Construct the necessary operator
    ops  = OperatorI(params)
    prob = Problem(grid, ops)

    params.k = k

    σ₀   = 0.02 # initial guess for the growth rate
    params.k = k

    λ, Χ = EigSolver(prob, grid, params, σ₀)

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
