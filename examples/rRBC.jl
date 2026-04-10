# ### Critical Rayleigh number for rotating Rayleigh-Bénard Convection
#
# ## Introduction
#
# This code finds the critical Rayleigh number for the onset of convection for rotating Rayleigh-Bénard Convection (rRBC)
# where the domain is periodic in y-direction.
# The code is benchmarked against Chandrasekhar's theoretical results [chandra2013](@cite).
#
# Parameter: 
#
# * Ekman number $\text{E} = 10⁻⁴$
# * Eigenvalue: critical modified Rayleigh number $\text{Ra}_c = 189.7$
#
# In this module, we do a linear stability analysis of a 2D rotating Rayleigh-Bénard case where the domain
# is periodic in the ``y``-direction, in the ``x``-direction is of infinite extent and vertically bounded.
#
# The background temperature profile $\overline{\theta}$ is given by 
# ```math
# \overline{\theta} = 1 - z.
# ```
# 
# ## Governing equations
# The non-dimensional form of the equations governing the perturbation is given by 
# ```math
# \begin{align}
#     \frac{\text{E}}{\text{Pr}} \frac{\partial \mathbf{u}}{\partial t}
#     + \hat{z} \times \mathbf{u} &=
#     -\nabla p + \text{Ra}\, \theta \hat{z} + \text{E} \nabla^2 \mathbf{u},
# \\
#     \frac{\partial \theta}{\partial t} 
#     &= \mathbf{u} \cdot \hat{z} + \nabla^2 \theta,
# \\
#     \nabla \cdot \mathbf{u} &= 0,
# \end{align}
# ```
# where $\text{E}=\nu/(fH^2)$ is the Ekman number and $\text{Ra} = g\alpha \Delta T/(f \kappa)$,
# $\Delta T$ is the temperature difference between the bottom and the top walls) is the modified Rayleigh number.
# Following [teed2010](@cite), by applying the operators $(\nabla \times \nabla \times)$ and $(\nabla \times)$ and 
# taking the $z$-component of the equations and assuming wave-like perturbations, 
# we obtained the equations for vertical velocity $w$, vertical vorticity $\zeta$ and temperature $\theta$,
# ```math
# \begin{align}
#     \text{E} \mathcal{D}^4 w - \partial_z \zeta &= -\text{Ra}\, \mathcal{D}_h^2 \theta,
# \\
#     \text{E} \mathcal{D}^2 \zeta + \partial_z w &= 0,
# \\
#     \mathcal{D}^2 \theta + w &= 0.
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
# Finally, the following system of differential equations is obtained,
# ```math
# \begin{align}
#     \text{E} \mathcal{D}^4  \tilde{w} - \partial_z \tilde{\zeta} &= -\text{Ra}\, \mathcal{D}_h^2 \tilde{\theta},
# \\
#     \text{E} \mathcal{D}^2 \tilde{\zeta} + \partial_z \tilde{w} &= 0,
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
# The eigenfunctions ``\tilde{u}``, ``\tilde{v}`` are related to ``\tilde{w}``, ``\tilde{\zeta}`` by
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
#     \partial_z \tilde{\zeta} = \tilde{\theta} = 0, 
#     \,\,\,\,\,\,\, \text{at} \,\,\, {z}=0, 1.
# \end{align}
# ```
#
# ## Generalized eigenvalue problem (GEVP)
# The above sets of equations with the boundary conditions can be expressed as a 
# standard generalized eigenvalue problem,
# ```math
# \begin{align}
#  AX = λBX, 
# \end{align}
# ```
# where $\lambda=\text{Ra}$ is the eigenvalue, and $X=[\tilde{w} \, \tilde{\zeta} \, \tilde{\theta}]^T$ is the eigenvector.
# The matrices $A$ and $B$ are given by
# ```math
# \begin{align}
#     A &= \begin{bmatrix}
#         \text{E} \mathcal{D}^4 & -\partial_z & 0 \\
#         \partial_z & \text{E} \mathcal{D}^2 & 0 \\
#         1 & 0 & \mathcal{D}^2
#     \end{bmatrix},
# \,\,\,\,\,\,\,
#     B &= \begin{bmatrix}
#         0 & 0 & -\mathcal{D}_h^2 \\
#         0 & 0 & 0 \\    
#         0 & 0 & 0 
#     \end{bmatrix}.
# \end{align}
# ```
#
# ## Numerical Implementation
# To implement the above GEVP in a numerical code, we need to actually code 
# following sets of equations: 
#
# ```math
# \begin{align}
#     A &= \begin{bmatrix}
#         \text{E} {D}^{4D} & -{D}_z^D & 0_n \\
#         \mathcal{D}^{zD} & \text{E} {D}^{2N} & 0_n \\
#         I_n & 0_n & \mathcal{D}^{2D}
#     \end{bmatrix},
# \,\,\,\,\,\,\,
#     B &= \begin{bmatrix}
#         0_n & 0_n & -{D}^{2D} \\
#         0_n & 0_n & 0_n \\    
#         0_n & 0_n & 0_n 
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
# $\mathcal{D}_y$, $\mathcal{D}_y^2$ and $\mathcal{D}_y^4$ are the first, second and fourth order
# Fourier differentiation matrix of size of $(N_y \times N_y)$, respectively. 
# $\mathcal{D}_z$, $\mathcal{D}_z^2$ and $\mathcal{D}_z^4$ are the first, second and fourth order
# Chebyshev differentiation matrix of size of $(N_z \times N_z)$, respectively.
#
#
# ## Implementation using the BiGSTARS equation DSL
# The DSL allows writing the governing equations directly. For the rRBC problem,
# the eigenvalue is the Rayleigh number Ra (not a growth rate), and the system
# is stationary at the marginal state (sigma = 0 → Ra is the eigenvalue).
#
# Note: In this formulation, Ra appears as the eigenvalue coupling the w-equation
# to the theta-equation. The DSL handles this naturally as a 3-variable GEVP.
#
# ## Load required packages
using BiGSTARS
using Printf
using LinearAlgebra

# ## 1. Domain
domain = Domain(
    x = FourierTransformed(),
    y = Fourier(180, [0, 2π]),
    z = Chebyshev(20, [0, 1])
)

# ## 2. Problem — 3-variable system (w, zeta, theta)
# The eigenvalue here is Ra (Rayleigh number), not a growth rate.
prob = EVP(domain, variables=[:w, :zeta, :theta], eigenvalue=:Ra)

# ## 3. Parameters
prob[:E] = 1e-4   # Ekman number

# ## 4. Substitutions
# Full Laplacian: D² = dx² + dy² + dz²
@substitution D2(A) = dx(dx(A)) + dy(dy(A)) + dz(dz(A))

# Horizontal Laplacian: Dh² = dx² + dy²
@substitution Dh2(A) = dx(dx(A)) + dy(dy(A))

# Biharmonic: D⁴ = (D²)²
@substitution D4(A) = D2(D2(A))

# ## 5. Governing equations
# w-equation: E * D4(w) - dz(zeta) = -Ra * Dh2(theta)
# Rearranged: Ra * Dh2(theta) = -E * D4(w) + dz(zeta)
# In EVP form: Ra * (-Dh2(theta)) = E * D4(w) - dz(zeta)
@equation Ra * Dh2(theta) = -E * D4(w) + dz(zeta)

# zeta-equation: dz(w) + E * D2(zeta) = 0  (no eigenvalue, goes to A)
# We write: Ra * 0*zeta == dz(w) + E*D2(zeta) but Ra must appear on LHS.
# For equations without the eigenvalue, we use a zero-mass trick:
# Actually, let's reformulate. The standard GEVP form has Ra only in the
# w-equation coupling to theta. The other equations have zero on the B matrix.
# In the DSL, every equation must have the eigenvalue on the LHS.
# We handle this by writing sigma*0*var == ... which gives B=0 for that row.
# But our DSL requires sigma on LHS. Let's use a small reformulation:

# Constraint equations: zeta and theta equations don't involve Ra.
# When the eigenvalue is absent from the LHS, the DSL treats these as
# algebraic constraints — the B matrix rows remain zero for these equations.
@equation 0 = dz(w) + E * D2(zeta)
@equation 0 = w + D2(theta)

# ## 6. Boundary conditions
# w: rigid lid (Dirichlet) + free-slip (d²w/dz²=0)
@bc left(w) = 0
@bc right(w) = 0
@bc left(dz(dz(w))) = 0
@bc right(dz(dz(w))) = 0

# zeta: free-slip (Neumann dz(zeta)=0)
@bc left(dz(zeta)) = 0
@bc right(dz(zeta)) = 0

# theta: fixed temperature (Dirichlet)
@bc left(theta) = 0
@bc right(theta) = 0

# ## 7. Solve

function solve_rRBC(k_val::Float64)
    
    cache = discretize(prob)

    ## For rRBC, the critical Ra is the smallest positive eigenvalue.
    ## nev=10 finds several eigenvalues so we can filter for the physical one.
    A, B = assemble(cache, k_val)
    solver = EigenSolver(A, B; σ₀=10.0, method=:Arnoldi, nev=10, which=:LM, sortby=:R)
    solve!(solver)
    λ, Χ = get_results(solver)

    ## Filter for positive real eigenvalues (Ra must be real and positive)
    λ, Χ = remove_evals(λ, Χ, 10.0, 1.0e15, "R")
    λ_sorted, _ = sort_evals(λ, Χ, :R; rev=false)
    print_evals(complex.(λ_sorted))

    Ra_numerical = real(λ_sorted[1])
    @printf "Numerical critical Ra: %1.4e\n" Ra_numerical

    ## Theoretical results from Chandrasekhar (1961)
    Ra_theory = 189.7
    @printf "Analytical solution of critical Ra: %1.4e\n" Ra_theory
    @printf "Relative error: %1.4e\n" abs(Ra_numerical - Ra_theory) / Ra_theory

    return abs(Ra_numerical - Ra_theory) / Ra_theory < 1e-2
end

# ## Result
solve_rRBC(1e-6)  # Critical Ra at k≈0 (exact k=0 is structurally singular in coefficient space)
