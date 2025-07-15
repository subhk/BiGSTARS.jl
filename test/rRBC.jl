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
using ModelingToolkit
using NonlinearSolve

using BiGSTARS
using BiGSTARS: AbstractParams
using BiGSTARS: Problem, OperatorI, TwoDGrid


# ### Define the parameters
@with_kw mutable struct Params{T} <: AbstractParams
    L::T                = 2π      # horizontal domain size
    H::T                = 1.0       # vertical   domain size
    E::T                = 1.0e-4    # inverse of Reynolds number 
    k::T                = 0.0       # x-wavenumber
    Ny::Int64           = 240       # no. of y-grid points
    Nz::Int64           = 30        # no. of Chebyshev points
    w_bc::String        = "rigid_lid"   # boundary condition for vertical velocity
    ζ_bc::String        = "free_slip"   # boundary condition for vertical vorticity
    b_bc::String        = "fixed"        # boundary condition for temperature
    eig_solver::String  = "krylov"      # eigenvalue solver
end

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

    ## --------------------------------------------------------
    ## allocating memory for the LHS and RHS matrices
    ## --------------------------------------------------------
    labels  = [:w, :ζ, :b]  # eigenfunction labels

    blocksA = [rand(ComplexF64, s₁, s₂) for _ in 1:3] # length must match length(labels)
    blocksB = [rand(Float64,    s₁, s₂) for _ in 1:3]

    #gevp   = GEVPMatrices(ComplexF64, Float64, N; nblocks=3, labels=labels)

    gevp    = GEVPMatrices(ComplexF64, Float64, blocksA, blocksB; labels=labels)

    ## Construct the matrix `A`
    ## ----------------------------------------------------------------------
    ## 1. w (vertical velocity)  equation (bcs: w = ∂ᶻᶻw = 0 @ z = 0, 1) 
    ## ----------------------------------------------------------------------
    gevp.As.w[:,    1:1s₂] = 1.0 * params.E * D⁴ 
    gevp.As.w[:,1s₂+1:2s₂] = 1.0 * prob.Dᶻᴺ 
    gevp.As.w[:,2s₂+1:3s₂] = 0.0 * I⁰ 


    ## ----------------------------------------------------------------------
    ## 2. ζ (vertical vorticity) equation (bcs: ∂ᶻζ = 0 @ z = 0, 1)
    ## ----------------------------------------------------------------------
    gevp.As.ζ[:,    1:1s₂] = 1.0 * prob.Dᶻᴰ
    gevp.As.ζ[:,1s₂+1:2s₂] = 1.0 * params.E * Dₙ²
    gevp.As.ζ[:,2s₂+1:3s₂] = 0.0 * I⁰

    ## ----------------------------------------------------------------------
    ## 3. b (buoyancy) equation (bcs: b = 0 @ z = 0, 1)
    ## ----------------------------------------------------------------------
    gevp.As.b[:,    1:1s₂] = 1.0 * I⁰ 
    gevp.As.b[:,1s₂+1:2s₂] = 0.0 * I⁰
    gevp.As.b[:,2s₂+1:3s₂] = 1.0 * D²    

    ## Construct the matrix `B`
    cnst = -1.0 
    gevp.Bs.w[:,2s₂+1:3s₂]  = -1.0 * ∇ₕ²

    return gevp.A, gevp.B
end

# ### Define the eigenvalue solver
function EigSolver(prob, grid, params, σ₀)

    A, B = generalized_EigValProb(prob, grid, params)

    if params.eig_solver == "arpack"

        λ, Χ = solve_shift_invert_arpack(A, B; σ₀=σ₀, which=:LM, sortby=:R)

    elseif params.eig_solver == "krylov"

        λ, Χ = solve_shift_invert_krylov(A, B; σ₀=σ₀, which=:LM)

    elseif params.eig_solver == "arnoldi"

        λ, Χ = solve_shift_invert_arnoldi(A, B; σ₀=σ₀, which=:LM)
    end
    ## ======================================================================
    @assert length(λ) > 0 "No eigenvalue(s) found!"

    @printf "||AΧ - λBΧ||₂: %f \n" norm(A * Χ[:,1] - λ[1] * B * Χ[:,1])

    print_evals(λ)

    return λ[1], Χ[:,1]
end
nothing #hide


# ### Solving the Stone problem
function solve_rRBC(k::Float64)

    params = Params{Float64}()

    # ### Construct grid and derivative operators
    grid  = TwoDGrid(params)

    # ### Construct the necesary operator
    ops  = OperatorI(params)
    prob = Problem(grid, ops, params)

    params.k = k

    σ₀   = 50.0 # initial guess for the growth rate
    params.k = k

    λ, Χ = EigSolver(prob, grid, params, σ₀)

    # Theoretical results from Chandrashekar (1961)
    λₜ = 189.7 
    @printf "Analytical solution of Stone (1971): %1.4e \n" λₛₜ 

    return abs(real(λₛ) - λₜ)/λₜ < 1e-4

end
nothing #hide

# # ## Result
solve_rRBC(0.0) # growth rate is at k=0.1  
nothing #hide


# function EigSolver(Op, params, σ₀)

#     printstyled("kₓ: $(params.kₓ) \n"; color=:blue)

#     𝓛, ℳ = construct_matrices(Op,  params)
    
#     N = params.Ny * params.Nz 
#     MatrixSize = 3N
#     @assert size(𝓛, 1)  == MatrixSize && 
#             size(𝓛, 2)  == MatrixSize &&
#             size(ℳ, 1)  == MatrixSize &&
#             size(ℳ, 2)  == MatrixSize "matrix size does not match!"

#     if params.method == "shift_invert"
#         printstyled("Eigensolver using Arpack eigs with shift and invert method ...\n"; 
#                     color=:red)

#         λₛ, Χ = EigSolver_shift_invert_arpack( 𝓛, ℳ, σ₀=σ₀, maxiter=40, which=:LM)
        
#         #@printf "found eigenvalue (at first): %f + im %f \n" λₛ[1].re λₛ[1].im

#     elseif params.method == "krylov"
#         printstyled("KrylovKit Method ... \n"; color=:red)

#         # look for the largest magnitude of eigenvalue (:LM)
#          λₛ, Χ = EigSolver_shift_invert_krylov( 𝓛, ℳ, σ₀=σ₀, maxiter=40, which=:LM)
        
#         #@printf "found eigenvalue (at first): %f + im %f \n" λₛ[1].re λₛ[1].im

#     elseif params.method == "arnoldi"

#         printstyled("Arnoldi: based on Implicitly Restarted Arnoldi Method ... \n"; 
#                         color=:red)

#         # decomp, history = partialschur(construct_linear_map(𝓛, ℳ), 
#         #                             nev=20, 
#         #                             tol=0.0, 
#         #                             restarts=50000, 
#         #                             which=LM())

#         # println(history)

#         # λₛ⁻¹, Χ = partialeigen(decomp)
#         # λₛ = @. 1.0 / λₛ⁻¹

#         λₛ, Χ = EigSolver_shift_invert_arnoldi( 𝓛, ℳ, 
#                                             σ₀=0.0, 
#                                             maxiter=50000, 
#                                             which=LM())

#         λₛ, Χ = remove_evals(λₛ, Χ, 10.0, 1.0e15, "R")
#         λₛ, Χ = sort_evals(λₛ, Χ, "R", "")

#     end
#     # ======================================================================

#     @printf "||𝓛Χ - λₛℳΧ||₂: %f \n" norm(𝓛 * Χ[:,1] - λₛ[1] * ℳ * Χ[:,1])
    
#     #print_evals(λₛ, length(λₛ))
#     @printf "largest growth rate : %1.4e%+1.4eim\n" real(λₛ[1]) imag(λₛ[1])

#     # 𝓛 = nothing
#     # ℳ = nothing

#     #return nothing #
#     return λₛ[1] #, Χ[:,1]
# end

# function solve_rRBC(kₓ::Float64)
#     params      = Params{Float64}(kₓ=0.5)
#     grid        = TwoDimGrid{params.Ny,  params.Nz}()
#     diffMatrix  = ChebMarix{ params.Ny,  params.Nz}()
#     Op          = Operator{params.Ny * params.Nz}()
#     Construct_DerivativeOperator!(diffMatrix, grid, params)
#     ImplementBCs_cheb!(Op, diffMatrix, params)
    
#     σ₀   = 50.0
#     params.kₓ = kₓ
    
#     λₛ = EigSolver(Op, params, σ₀)

#     # Theoretical results from Chandrashekar (1961)
#     λₛₜ = 189.7 
#     @printf "Analytical solution of Stone (1971): %1.4e \n" λₛₜ 

#     return abs(real(λₛ) - λₛₜ)/λₛₜ < 1e-4
    
# end

# #end #module
# # ========== end of the module ==========================
