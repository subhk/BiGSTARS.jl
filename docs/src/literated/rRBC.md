```@meta
EditURL = "../../../examples/rRBC.jl"
```

### Critical Rayleigh number for rotating Rayleigh-Benard Convection

## Introduction
This code finds critical Rayleigh number for the onset of convection for rotating Rayleigh Benrad Convection (rRBC)
where the domain is periodic in y-direction.
The code is benchmarked against Chandrashekar's theoretical results.
Hydrodynamic and hydromagnetic stability by S. Chandrasekhar, 1961 (page no-95).

Parameter:

Ekman number $E = 10⁻⁴$

Eigenvalue: critical modified Rayleigh number $Ra_c = 189.7$

In this module, we do a linear stability analysis of a 2D rotating Rayleigh-Bernard case where the domain
is periodic in the ``y``-direction, in the ``x``-direction is of infinite extent and vertically bounded.

The background temperature profile $\overline{\theta}$ is given by
```math
\overline{\theta} = 1 - z.
```

## Governing equations
The non-dimensional form of the equations governing the perturbation is given by
```math
\begin{align}
    \frac{E}{Pr} \frac{\partial \mathbf{u}}{\partial t}
    + \hat{z} \times \mathbf{u} &=
    -\nabla p + Ra \theta \hat{z} + E \nabla^2 \mathbf{u},
\\
    \frac{\partial \theta}{\partial t}
    &= \mathbf{u} \cdot \hat{z} + \nabla^2 \theta,
\\
    \nabla \cdot \mathbf{u} &= 0,
\end{align}
```
where $E=\nu/(fH^2)$ is the Ekman number and $Ra = g\alpha \Delta T/(f \kappa)$,
$\Delta T$ is the temperature difference between the bottom and the top walls) is the modified Rayleigh number.
By applying the operators $(\nabla \times \nabla \times)$ and $(\nabla \times)$ and
taking the $z$-component of the equations and assuming wave-like perturbations,
we obtained the equations for vertical velocity $w$, vertical vorticity $\zeta$ and temperature $\theta$,
```math
\begin{align}
    E \mathcal{D}^4 w - \partial_z \zeta &= -Ra \mathcal{D}_h^2 \theta,
\\
    E \mathcal{D}^2 \zeta + \partial_z w &= 0,
\\
    \mathcal{D}^2 \theta + w &= 0.
\end{align}
```

## Normal mode solutions
Next we consider normal-mode perturbation solutions in the form of (we seek stationary solutions at the marginal state, i.e., $\sigma = 0$),
```math
\begin{align}
 [w, \zeta, \theta](x,y,z,t) = \mathfrak{R}\big([\tilde{w}, \tilde{\zeta}, \tilde{\theta}](y, z)  e^{i kx + \sigma t}\big),
\end{align}
```
where the symbol $\mathfrak{R}$ denotes the real part and a variable with `tilde' denotes an eigenfunction.
Finally following systems of differential equations are obtained,
```math
\begin{align}
    E \mathcal{D}^4  \tilde{w} - \partial_z \tilde{\zeta} &= - Ra \mathcal{D}_h^2 \tilde{\theta},
\\
    E \mathcal{D}^2 \tilde{\zeta} + \partial_z \tilde{w} &= 0,
\\
    \mathcal{D}^2 \tilde{\theta} + \tilde{w} &= 0,
\end{align}
```
where
```math
\begin{align}
\mathcal{D}^4  = (\mathcal{D}^2 )^2 = \big(\partial_y^2 + \partial_z^2 - k^2\big)^2,
\,\,\,\, \text{and} \,\, \mathcal{D}_h^2 = (\partial_y^2 - k^2).
\end{align}
```
The eigenfunctions ``\tilde{u}``, ``\tilde{v}`` are related to ``\tilde{w}``, ``\tilde{\zeta}`` by the relations
```math
\begin{align}
    -\mathcal{D}_h^2 \tilde{u} &= i k \partial_{z} \tilde{w} + \partial_y \tilde{\zeta},
\\
    -\mathcal{D}_h^2 \tilde{v} &= \partial_{yz} \tilde{w} -  i k \tilde{\zeta}.
\end{align}
```

## Boundary conditions
We choose periodic boundary conditions in the ``y``-direction and free-slip, rigid lid, with zero buoyancy flux in the ``z`` direction, i.e.,
```math
\begin{align}
    \tilde{w} = \partial_{zz} \tilde{w} =
    \partial_z \tilde{\zeta} = \tilde{\theta} = 0,
    \,\,\,\,\,\,\, \text{at} \,\,\, {z}=0, 1.
\end{align}
```

## Generalized eigenvalue problem (GEVP)
The above sets of equations with the boundary conditions can be expressed as a
standard generalized eigenvalue problem,
```math
\begin{align}
 AX = λBX,
\end{align}
```
where $\lambda=Ra$ is the eigenvalue, and $X=[\tilde{w} \, \tilde{\zeta} \, \tilde{\theta}]^T$ is the eigenvector.
The matrices $A$ and $B$ are given by
```math
\begin{align}
    A &= \begin{bmatrix}
        E \mathcal{D}^4 & -\partial_z & 0 \\
        \partial_z & E \mathcal{D}^2 & 0 \\
        1 & 0 & \mathcal{D}^2
    \end{bmatrix},
\,\,\,\,\,\,\,
    B &= \begin{bmatrix}
        0 & 0 & -\mathcal{D}_h^2 \\
        0 & 0 & 0 \\
        0 & 0 & 0
    \end{bmatrix}.
\end{align}
```

## Numerical Implementation
To implement the above GEVP in a numerical code, we need to actually code
following sets of equations:

```math
\begin{align}
    A &= \begin{bmatrix}
        E {D}^{4D} & -{D}_z^D & 0_n \\
        \mathcal{D}^{zD} & E {D}^{2N} & 0_n \\
        I_n & 0_n & \mathcal{D}^{2D}
    \end{bmatrix},
\,\,\,\,\,\,\,
    B &= \begin{bmatrix}
        0_n & 0_n & -{D}^{2D} \\
        0_n & 0_n & 0_n \\
        0_n & 0_n & 0_n
    \end{bmatrix}.
\end{align}
```
where $I_n$ is the identity matrix of size $(n \times n)$, where $n=N_y N_z$, $N_y$ and $N_z$
are the number of grid points in the $y$ and $z$ directions respectively.
$0_n$ is the zero matrix of size $(n \times n)$.
The differential operator matrices are given by

```math
\begin{align}
{D}^{2D} &= \mathcal{D}_y^2 \otimes {I}_z + {I}_y \otimes \mathcal{D}_z^{2D} - k^2 {I}_n,
\\
{D}^{2N} &= \mathcal{D}_y^2 \otimes {I}_z + {I}_y \otimes \mathcal{D}_z^{2N} - k^2 {I}_n,
\\
 {D}^{4D} &= \mathcal{D}_y^4 \otimes {I}_z
   + {I}_y \otimes \mathcal{D}_z^{4D} + k^4 {I}_n - 2 k^2 {D}_y^2 \otimes {I}_z
   - 2 k^2 {I}_y \otimes {D}_z^{2D} + 2 {D}_y^2 \otimes {D}_z^{2D}
\end{align}
```
where $\otimes$ is the Kronecker product. ${I}_y$ and ${I}_z$ are
identity matrices of size $(N_y \times N_y)$ and $(N_z \times N_z)$ respectively,
and ${I}={I}_y \otimes {I}_z$. The superscripts $D$ and $N$ in the operator matrices
denote the type of boundary conditions applied ($D$ for Dirichlet or $N$ for Neumann).
$\mathcal{D}_y$, $\mathcal{D}_y^2$ and $\mathcal{D}_y^4$ are the first, second and fourth order
Fourier differentiation matrix of size of $(N_y \times N_y)$, respectively.
$\mathcal{D}_z$, $\mathcal{D}_z^2$ and $\mathcal{D}_z^4$ are the first, second and fourth order
Chebyshev differentiation matrix of size of $(N_z \times N_z)$, respectively.


## Load required packages

````julia
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
````

## Parameters

````julia
@with_kw mutable struct Params{T} <: AbstractParams
    L::T                = 2π            # horizontal domain size
    H::T                = 1.0           # vertical   domain size
    E::T                = 1.0e-4        # the Ekman number
    Pr::T               = 1.0           # the Prandtl number
    k::T                = 0.0           # x-wavenumber
    Ny::Int64           = 180           # no. of y-grid points
    Nz::Int64           = 20            # no. of Chebyshev points
    w_bc::String        = "rigid_lid"   # boundary condition for vertical velocity
    ζ_bc::String        = "free_slip"   # boundary condition for vertical vorticity
    b_bc::String        = "fixed"       # boundary condition for temperature
    eig_solver::String  = "arnoldi"     # eigenvalue solver
end
````

## Basic state

````julia
function basic_state(grid, params)

    Y, Z = ndgrid(grid.y, grid.z)
    Y    = transpose(Y)
    Z    = transpose(Z)

    # Define the basic state
    B   = @. 1.0 - Z          # basic state temperature
    U   = @. 0.0 * Z          # basic state along-front velocity

    # Calculate all the 1st, 2nd and yz derivatives in 2D grids
    bs = compute_derivatives(U, B, grid.y; grid.Dᶻ, grid.D²ᶻ, gridtype = :All)
    precompute!(bs; which = :All)   # eager cache, returns bs itself
    @assert bs.U === U              # originals live in the same object
    @assert bs.B === B

    return bs
end
````

## Constructing GEVP

````julia
function generalized_EigValProb(prob, grid, params)

    # basic state
    bs = basic_state(grid, params)

    n  = params.Ny * params.Nz
    I⁰ = sparse(Matrix(1.0I, n, n)) # Identity matrix
    s₁ = size(I⁰, 1);
    s₂ = size(I⁰, 2);

    # the horizontal Laplacian operator: ∇ₕ² = ∂ʸʸ - k²
    ∇ₕ² = SparseMatrixCSC(Zeros(n, n))
    ∇ₕ² = (1.0 * prob.D²ʸ - 1.0 * params.k^2 * I⁰)

    # Construct the 4th order derivative
    D⁴ᴰ = (1.0 * prob.D⁴ʸ
        + 1.0 * prob.D⁴ᶻᴰ
        + 1.0 * params.k^4 * I⁰
        - 2.0 * params.k^2 * prob.D²ʸ
        - 2.0 * params.k^2 * prob.D²ᶻᴰ
        + 2.0 * prob.D²ʸ²ᶻᴰ)

    # Construct the 2nd order derivative
    D²ᴰ = (1.0 * prob.D²ᶻᴰ  + 1.0 * ∇ₕ²)  # with Dirchilet BC
    D²ᴺ = (1.0 * prob.D²ᶻᴺ  + 1.0 * ∇ₕ²)  # with Neumann BC

    # See `Numerical Implementation' section for the theory
    # ──────────────────────────────────────────────────────────────────────────────
    # 1) Now define your 3×3 block-rows in a NamedTuple of 3-tuples
    # ──────────────────────────────────────────────────────────────────────────────
    # Construct the matrix `A`
    Ablocks = (
        w = (  # w-equation: [ED⁴] [-Dᶻᴺ] [zero]
                sparse(params.E * D⁴ᴰ),
                sparse(-prob.Dᶻᴺ),
                spzeros(Float64, s₁, s₂)
        ),
        ζ = (  # ζ-equation: [Dᶻᴰ] [ED²ᴺ] [zero]
                sparse(prob.Dᶻᴰ),
                sparse(params.E * D²ᴺ),
                spzeros(Float64, s₁, s₂)
        ),
        θ = (  # b-equation: [I₀] [zero] [D²ᴰ]
                sparse(I⁰),
                spzeros(Float64, s₁, s₂),
                sparse(D²ᴰ)
        )
    )

    # Construct the matrix `B`
    Bblocks = (
        w = (  # w-equation: [zero], [zero] [-∇ₕ²]
                spzeros(Float64, s₁, s₂),
                spzeros(Float64, s₁, s₂),
                sparse(-∇ₕ²)
        ),
        ζ = (  # ζ-equation: [zero], [zero], [zero]
                spzeros(Float64, s₁, s₂),
                spzeros(Float64, s₁, s₂),
                spzeros(Float64, s₁, s₂)
        ),
        θ = (  # b-equation: [zero], [zero], [zero]
                spzeros(Float64, s₁, s₂),
                spzeros(Float64, s₁, s₂),
                spzeros(Float64, s₁, s₂)
        )
    )

    # ──────────────────────────────────────────────────────────────────────────────
    # 2) Assemble the block-row matrices into a GEVPMatrices object
    # ──────────────────────────────────────────────────────────────────────────────
    gevp = GEVPMatrices(Ablocks, Bblocks)


    # ──────────────────────────────────────────────────────────────────────────────
    # 3) And now you have exactly:
    #    gevp.A, gevp.B                    → full sparse matrices
    #    gevp.As.w, gevp.As.ζ, gevp.As.θ   → each block-row view
    #    gevp.Bs.w, gevp.Bs.ζ, gevp.Bs.θ
    # ──────────────────────────────────────────────────────────────────────────────

    return gevp.A, gevp.B
end
````

## Eigenvalue solver

````julia
function EigSolver(prob, grid, params, σ₀)

    A, B = generalized_EigValProb(prob, grid, params)

    # Construct the eigenvalue solver
    # Methods available: :Krylov (by default), :Arnoldi, :Arpack
    # Here we are looking for minimum Rayleigh number (real part of eigenvalue)
    # `method=:Arnoldi' is good when looking for largest magnitude eigenvalue
    solver = EigenSolver(A, B; σ₀=σ₀, method=:Arnoldi, nev=10, which=:LM, sortby=:R)
    solve!(solver)
    λ, Χ = get_results(solver)
    #print_summary(solver)

    # looking for min Ra
    λ, Χ = remove_evals(λ, Χ, 10.0, 1.0e15, "R")
    λ, Χ = sort_evals_(λ, Χ,  :R, rev=false)
    print_evals(complex.(λ))

    return λ[1], Χ[:,1]
end
````

## Solving the problem

````julia
function solve_rRBC(k::Float64)

    # Calling problem parameters
    params = Params{Float64}()

    # Construct grid and derivative operators
    grid  = TwoDGrid(params)

    # Construct the necesary operator
    ops  = OperatorI(params)
    prob = Problem(grid, ops)

    # update the wavenumber
    params.k = k

    # initial guess for the growth rate
    σ₀   = 0.0

    λ, Χ = EigSolver(prob, grid, params, σ₀)

    # Theoretical results from Chandrashekar (1961)
    λₜ = 189.7
    @printf "Analytical solution of critical Ra: %1.4e \n" λₜ

    return abs(real(λ) - λₜ)/λₜ < 1e-4
end
````

## Result

````julia
solve_rRBC(0.0) # Critical Rayleigh number is at k=0.0
````

````
(attempt  1/16) trying σ = 0.000000 with Arnoldi
  ✓ converged: λ₁ = 193.728586 + 0.000000i
(attempt  2/16) trying σ = 0.000000 with Arnoldi
  ✓ converged: λ₁ = 193.728586 + 0.000000i
  ✓ successive eigenvalues converged: |Δλ| = 2.01e-09 < 1.00e-05
Top 9 eigenvalues (sorted):
Idx │ Real Part     Imag Part
────┼──────────────────────────────
  9 │  1.937286e+02          
  8 │  1.933564e+02          
  7 │  1.933564e+02          
  6 │  1.907175e+02          
  5 │  1.907175e+02          
  4 │  1.906031e+02          
  3 │  1.906031e+02          
  2 │  1.897041e+02          
  1 │  1.897041e+02          
Analytical solution of critical Ra: 1.8970e+02 

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

