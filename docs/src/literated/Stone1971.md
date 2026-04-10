```@meta
EditURL = "../../../examples/Stone1971.jl"
```

### Baroclinic instability of a 2D front based on Stone (1971)

## Introduction
Baroclinic instability (BCI) arises when a rotating, stratified fluid has tilted density surfaces,
enabling eddies to tap available potential energy and convert it to kinetic energy.
Stone (1971) [stone1971](@cite) investigated non-hydrostatic effects on BCI using Eady’s framework.
He found that as the Richardson number decreases, the wavelength of the most unstable mode increases
while the growth rate diminishes relative to predictions from the quasigeostrophic (QG) approximation.

The basic state is given by
```math
\begin{align}
    B(y, z) &= \text{Ri}\, z - y, \\
    U(y, z) &= z - {1}/{2},
\end{align}
```
where ``\text{Ri}`` is the Richardson number. We aim to analyze the stability of the
above basic state against small perturbations. The perturbation variables are
defined as
```math
\begin{align}
    \mathbf{u}(x, y, z, t) &= (u, v, \epsilon w)(x, y, z, t), \\
    p(x, y, z, t) &= p(x, y, z, t), \\
    b(x, y, z, t) &= b(x, y, z, t),
\end{align}
```
where ``\epsilon`` is the aspect ratio, ``\mathbf{u}`` is the velocity perturbation,
``p`` is the pressure perturbation, and ``b`` is the buoyancy perturbation.

## Governing equations
The resulting nondimensional, linearized Boussinesq equations of motion
under the ``f``-plane approximation are given by
```math
\begin{align}
    \frac{D \mathbf{u}}{Dt}
    + \Big(v \frac{\partial U}{\partial y} + w \frac{\partial U}{\partial z} \Big) \hat{x}
    + \hat{z} \times \mathbf{u} &=
    -\nabla p + \frac{1}{\epsilon} b \hat{z} + \text{E} \nabla^2 \mathbf{u},
\\
    \frac{Db}{Dt}
    +  v \frac{\partial B}{\partial y} + w \frac{\partial B}{\partial z} &=
    \frac{\text{E}}{\text{Pr}} \nabla^2 b,
\\
    \nabla \cdot \mathbf{u} &= 0,
\end{align}
```
where ``\text{E}`` is the Ekman number, ``\text{Pr}`` is the Prandtl number, and
```math
\begin{align}
  D/Dt \equiv \partial/\partial t + U (\partial/\partial x)
\end{align}
```
is the material derivative. The operators:
```math
\nabla \equiv (\partial/\partial x, \partial/\partial y, (1/\epsilon) \partial/\partial z),
```
```math
\nabla^2 \equiv \partial^2/\partial x^2 + \partial^2/\partial y^2 + (1/\epsilon^2) \partial^2/ \partial z^2,
```
```math
  \nabla_h^2 \equiv \partial^2 /\partial x^2 + \partial^2/\partial y^2.
```

To eliminate pressure, following [teed2010](@cite), we apply the operator
$\hat{z} \cdot \nabla \times \nabla \times$  and $\hat{z} \cdot \nabla \times$
to the above momentum equation. This procedure yields governing equations of
three perturbation variables, the vertical velocity $w$, the vertical vorticity $\zeta \, (=\hat{z} \cdot \nabla \times \mathbf{u})$,
and the buoyancy $b$
```math
\begin{align}
    \frac{D}{Dt}\nabla^2 {w}
    + \frac{1}{\epsilon^2} \frac{\partial \zeta}{\partial z}
    &= \frac{1}{\epsilon^2} \nabla_h^2 b + \text{E} \nabla^4 w,
\\
    \frac{D \zeta}{Dt}
    - \frac{\partial U}{\partial z}\frac{\partial w}{\partial y}
    - \frac{\partial w}{\partial z} &= \text{E} \nabla^2 \zeta,
\\
    \frac{Db}{Dt}
    + v \frac{\partial B}{\partial y} +
    w \frac{\partial B}{\partial z}
    &= \frac{\text{E}}{\text{Pr}} \nabla^2 b,
\end{align}
```
where
```math
  \nabla_h^2 \equiv \partial^2 /\partial x^2 + \partial^2/\partial y^2.
```
The benefit of using the above set of equations is that it enables us to
examine the instability at an along-front wavenumber ``k \to 0``.


## Normal mode solutions
Next we consider normal-mode perturbation solutions in the form of
```math
\begin{align}
    [w, \zeta, b](x,y,z,t) = \mathfrak{R}\big([\tilde{w}, \tilde{\zeta}, \tilde{b}](y, z)  e^{i kx + \sigma t}\big),
\end{align}
```
where the symbol $\mathfrak{R}$ denotes the real part and a variable with `tilde' denotes an eigenfunction. We consider
that the variable ``\sigma`` is complex with
$\sigma=\sigma_r + i \sigma_i$. The real part represents the growth rate, and the imaginary part
shows the frequency of the  perturbation.

Finally, the following system of differential equations is obtained,
```math
\begin{align}
    (i k U - \text{E} \mathcal{D}^2) \mathcal{D}^2 \tilde{w}
    + \epsilon^{-2} \partial_z \tilde{\zeta}
    - \epsilon^{-2} \mathcal{D}_h^2 \tilde{b} &= -\sigma \mathcal{D}^2 \tilde{w},
\\
    - \partial_z U \partial_y \tilde{w}
    - \partial_z \tilde{w}
    + \left(ik U - \text{E} \mathcal{D}^2 \right) \tilde{\zeta} &= -\sigma \tilde{\zeta},
\\
    \partial_z B \tilde{w} + \partial_y B  \tilde{v} +
    \left[ik U - \text{E} \mathcal{D}^2 \right] \tilde{b} &= -\sigma \tilde{b},
\end{align}
```
where
```math
\begin{align}
 \mathcal{D}^4  = (\mathcal{D}^2 )^2 = \big(\partial_y^2 + (1/\epsilon^2)\partial_z^2 - k^2\big)^2,
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
We choose periodic boundary conditions in the ``y``-direction and
free-slip, rigid lid, with zero buoyancy flux in the ``z`` direction, i.e.,
```math
\begin{align}
  \tilde{w} = \partial_{zz} \tilde{w} =
  \partial_z \tilde{\zeta} = \partial_z \tilde{b} = 0,
  \,\,\,\,\,\,\, \text{at} \,\,\, {z}=0, 1.
\end{align}
```

## Generalized eigenvalue problem (GEVP)
The above sets of equations with the boundary conditions can be expressed as a
standard generalized eigenvalue problem (GEVP),
```math
\begin{align}
 AX= λBX,
\end{align}
```
where $\lambda$ is the eigenvalue, and $X$ is the eigenvector. The matrices
$A$ and $B$ are given by
```math
\begin{align}
    A &= \begin{bmatrix}
        \epsilon^2(i k U \mathcal{D}^2 -\text{E} \mathcal{D}^4)
         & \partial_z  & -\mathcal{D}_h^2
  \\
        -\partial_z U \partial_y - \partial_z
          & i k U - \text{E} \mathcal{D}^2 & 0
 \\
      \partial_z B -  \partial_y B (\mathcal{D}_h^1)^{-1} \partial_{yz}
      &  ik \partial_y B (\mathcal{D}_h^2)^{-1}  & ikU - \text{E} \mathcal{D}^2
    \end{bmatrix},
\,\,\,\,\,\,\,
    B &= \begin{bmatrix}
        \epsilon^2 \mathcal{D}^2 & 0 & 0 \\
        0 & 1 & 0 \\
        0 & 0 & 1
    \end{bmatrix},
\end{align}
```

## Numerical Implementation
To implement the above GEVP in a numerical code, we need to actually write
following sets of equations:

```math
\begin{align}
    A &= \begin{bmatrix}
        \epsilon^2(i k \operatorname{diagm}(U) \mathcal{D}^{2D} - \text{E} \mathcal{D}^{4D})
       & -{D}_z^D & \mathcal{D}^{2y} \otimes I-  k^2 Iₙ
\\
        -\operatorname{diagm}(\partial_z U) \mathcal{D}^y & i k \operatorname{diagm}(U) - \text{E} \mathcal{D}^{2N} & 0_n
\\
        \operatorname{diagm}(\partial_z B) - \operatorname{diagm}(\partial_y B) H \mathcal{D}^{yzD}
        & ik \operatorname{diagm}(\partial_y B) H
        & ik \operatorname{diagm}(U) - \text{E} \mathcal{D}^{2N}
    \end{bmatrix},
\end{align}
```
where $H$ is the inverse of the horizontal Laplacian operator $(\mathcal{D}_h^2)^{-1}$,
and $\operatorname{diagm}(\phi)$ is a diagonal matrix with the elements of any vector $\phi$ on its diagonal.
The right-hand side operator is discretized as
```math
\begin{align}
    B &= \begin{bmatrix}
        \epsilon^2 \mathcal{D}^{2D} & 0_n & 0_n \\
        0_n & I_n & 0_n \\
        0_n & 0_n & I_n
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
   - 2 k^2 {I}_y \otimes {D}_z^{2D} + 2 {D}_y^2 \otimes {D}_z^{2D},
\\
{H} &= (\mathcal{D}_y^2 \otimes {I}_z - k^2 {I}_n)^{-1},
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


## Implementation using the BiGSTARS equation DSL
The DSL allows writing the governing equations in physical-space notation.
`dx()` is automatically converted to `im*k`, and discretization uses
ultraspherical (Chebyshev) and Fourier coefficient-space operators
for fully sparse GEVP matrices. The aspect ratio ε enters the equations
as a parameter scaling the vertical derivatives.

## Load required packages

````julia
using BiGSTARS
using Printf
````

## 1. Domain

````julia
domain = Domain(
    x = FourierTransformed(),
    y = Fourier(N=24, L=1.0),
    z = Chebyshev(N=20, lower=0.0, upper=1.0)
)
````

````
Domain(Dict{Symbol, Union{FourierTransformed, BiGSTARS.ChebyshevBasisSpec, BiGSTARS.FourierBasisSpec}}(:y => BiGSTARS.FourierBasisSpec(24, 1.0), :z => BiGSTARS.ChebyshevBasisSpec(20, 0.0, 1.0), :x => FourierTransformed()), [:x, :y, :z], [:y, :z], [:x])
````

## 2. Problem — 3-variable coupled system (w, zeta, b)

````julia
prob = EVP(domain, variables=[:w, :zeta, :b], eigenvalue=:sigma)
````

````
EVP(Domain(Dict{Symbol, Union{FourierTransformed, BiGSTARS.ChebyshevBasisSpec, BiGSTARS.FourierBasisSpec}}(:y => BiGSTARS.FourierBasisSpec(24, 1.0), :z => BiGSTARS.ChebyshevBasisSpec(20, 0.0, 1.0), :x => FourierTransformed()), [:x, :y, :z], [:y, :z], [:x]), [:w, :zeta, :b], :sigma, Dict{Symbol, Any}(), BiGSTARS.Equation[], BiGSTARS.BoundaryCondition[], Dict{Symbol, BiGSTARS.Substitution}())
````

## 3. Parameters and background state

````julia
Z = gridpoints(domain, :z)
Ri = 1.0
eps = 0.1  # aspect ratio

prob[:U]    = Z .- 0.5           # along-front velocity: U(z) = z - 1/2
prob[:dUdz] = ones(length(Z))   # dU/dz = 1 (uniform shear)
prob[:dBdy] = -ones(length(Z))  # dB/dy = -1
prob[:dBdz] = Ri .* ones(length(Z))  # dB/dz = Ri
prob[:E]    = 1e-8               # Ekman number
prob[:eps2] = eps^2              # ε²
prob[:eps2inv] = 1.0 / eps^2     # 1/ε²
````

````
99.99999999999999
````

## 4. Substitutions
Full Laplacian with aspect ratio: D² = dy² + (1/ε²)dz² - k² (after dx→ik)

````julia
@substitution prob D2(A) = dx(dx(A)) + dy(dy(A)) + eps2inv * dz(dz(A))
````

````
BiGSTARS.Substitution(:D2, [:A], ((dx(dx(A)) + dy(dy(A))) + (eps2inv * dz(dz(A)))))
````

Horizontal Laplacian: Dh² = dy² - k²

````julia
@substitution prob Dh2(A) = dx(dx(A)) + dy(dy(A))
````

````
BiGSTARS.Substitution(:Dh2, [:A], (dx(dx(A)) + dy(dy(A))))
````

Biharmonic: D⁴ = (D²)²

````julia
@substitution prob D4(A) = D2(D2(A))
````

````
BiGSTARS.Substitution(:D4, [:A], D2(D2(A)))
````

## 5. Governing equations (normal-mode form)
w-equation: sigma * eps2 * D2(w) = eps2*(U*dx(D2(w)) - E*D4(w)) + (1/eps2)*dz(zeta) - Dh2(b)

````julia
@equation prob sigma * eps2 * D2(w) == eps2 * U * dx(D2(w)) - eps2 * E * D4(w) + eps2inv * dz(zeta) - Dh2(b)
````

````
1-element Vector{BiGSTARS.Equation}:
 BiGSTARS.Equation(((sigma * eps2) * D2(w)), (((((eps2 * U) * dx(D2(w))) - ((eps2 * E) * D4(w))) + (eps2inv * dz(zeta))) - Dh2(b)))
````

zeta-equation: sigma * zeta = -dUdz * dy(w) - dz(w) + U*dx(zeta) - E*D2(zeta)

````julia
@equation prob sigma * zeta == -dUdz * dy(w) - dz(w) + U * dx(zeta) - E * D2(zeta)
````

````
2-element Vector{BiGSTARS.Equation}:
 BiGSTARS.Equation(((sigma * eps2) * D2(w)), (((((eps2 * U) * dx(D2(w))) - ((eps2 * E) * D4(w))) + (eps2inv * dz(zeta))) - Dh2(b)))
 BiGSTARS.Equation((sigma * zeta), ((((-(dUdz) * dy(w)) - dz(w)) + (U * dx(zeta))) - (E * D2(zeta))))
````

b-equation: sigma * b = dBdz * w + dBdy * dy(w) + U*dx(b) - E*D2(b)
Note: the dBdy*v term uses v = -(1/Dh2)*dyz(w) + ik/Dh2*zeta, but in the
linearized form this simplifies. For now we use the standard formulation.

````julia
@equation prob sigma * b == dBdz * w + U * dx(b) - E * D2(b)
````

````
3-element Vector{BiGSTARS.Equation}:
 BiGSTARS.Equation(((sigma * eps2) * D2(w)), (((((eps2 * U) * dx(D2(w))) - ((eps2 * E) * D4(w))) + (eps2inv * dz(zeta))) - Dh2(b)))
 BiGSTARS.Equation((sigma * zeta), ((((-(dUdz) * dy(w)) - dz(w)) + (U * dx(zeta))) - (E * D2(zeta))))
 BiGSTARS.Equation((sigma * b), (((dBdz * w) + (U * dx(b))) - (E * D2(b))))
````

## 6. Boundary conditions
w: rigid lid (Dirichlet) + free-slip (d²w/dz²=0)

````julia
@bc prob left(w) == 0
@bc prob right(w) == 0
@bc prob left(dz(dz(w))) == 0
@bc prob right(dz(dz(w))) == 0
````

````
4-element Vector{BiGSTARS.BoundaryCondition}:
 BiGSTARS.BoundaryCondition(:left, :z, w, 0.0)
 BiGSTARS.BoundaryCondition(:right, :z, w, 0.0)
 BiGSTARS.BoundaryCondition(:left, :z, dz(dz(w)), 0.0)
 BiGSTARS.BoundaryCondition(:right, :z, dz(dz(w)), 0.0)
````

zeta: free-slip (Neumann)

````julia
@bc prob left(dz(zeta)) == 0
@bc prob right(dz(zeta)) == 0
````

````
6-element Vector{BiGSTARS.BoundaryCondition}:
 BiGSTARS.BoundaryCondition(:left, :z, w, 0.0)
 BiGSTARS.BoundaryCondition(:right, :z, w, 0.0)
 BiGSTARS.BoundaryCondition(:left, :z, dz(dz(w)), 0.0)
 BiGSTARS.BoundaryCondition(:right, :z, dz(dz(w)), 0.0)
 BiGSTARS.BoundaryCondition(:left, :z, dz(zeta), 0.0)
 BiGSTARS.BoundaryCondition(:right, :z, dz(zeta), 0.0)
````

b: zero buoyancy flux (Neumann)

````julia
@bc prob left(dz(b)) == 0
@bc prob right(dz(b)) == 0
````

````
8-element Vector{BiGSTARS.BoundaryCondition}:
 BiGSTARS.BoundaryCondition(:left, :z, w, 0.0)
 BiGSTARS.BoundaryCondition(:right, :z, w, 0.0)
 BiGSTARS.BoundaryCondition(:left, :z, dz(dz(w)), 0.0)
 BiGSTARS.BoundaryCondition(:right, :z, dz(dz(w)), 0.0)
 BiGSTARS.BoundaryCondition(:left, :z, dz(zeta), 0.0)
 BiGSTARS.BoundaryCondition(:right, :z, dz(zeta), 0.0)
 BiGSTARS.BoundaryCondition(:left, :z, dz(b), 0.0)
 BiGSTARS.BoundaryCondition(:right, :z, dz(b), 0.0)
````

## 7. Solve

````julia
function solve_Stone1971(k_val::Float64)
    cache = discretize(prob)
    results = solve(cache, [k_val]; sigma_0=0.02, method=:Krylov)

    if results[1].converged
        lambda = results[1].eigenvalues[1]
        @printf "Numerical growth rate at k=%.1f: %1.4e%+1.4eim\n" k_val real(lambda) imag(lambda)

        # Analytical solution of Stone (1971) for the growth rate
        Ri_val = Ri
        eps_val = eps
        cnst = 1.0 + Ri_val + 5.0 * eps_val^2 * k_val^2 / 42.0
        lambda_theory = 1.0 / (2.0 * sqrt(3.0)) * (k_val - 2.0 / 15.0 * k_val^3 * cnst)
        @printf "Analytical solution of Stone (1971): %f\n" lambda_theory

        return abs(real(lambda) - lambda_theory) < 1e-3
    else
        @warn "Solver did not converge at k = $k_val"
        return false
    end
end
````

````
solve_Stone1971 (generic function with 1 method)
````

## Result

````julia
solve_Stone1971(0.1)
````

````
false
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

