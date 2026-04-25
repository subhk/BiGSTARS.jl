```@meta
EditURL = "../../../examples/Eady.jl"
```

### Baroclinic instability of a 2D front based on Eady (1949)

## Introduction
Eady (1949) [eady1949long](@cite) showed that in a uniformly sheared, stratified layer between two rigid lids on
an ``f``-plane, two counter-propagating Rossby edge waves can phase lock and convert available potential energy
into kinetic energy, producing baroclinic eddies that grow fastest at wavelengths about
four deformation radii and on timescales of a few days.

## Basic state
The basic state is given by
```math
\begin{align}
    B(y, z) &= \text{Ri}\, z - y, \\
    U(y, z) &= z - {1}/{2},
\end{align}
```
where ``\text{Ri}`` is the Richardson number, and $N^2 = \text{Ri}$ is the stratification.

## Governing equations
The non-dimensional form of the linearized version of the QG PV perturbation equation under
the $f$-plane approximation can be expressed as,
```math
\begin{align}
    \frac{\partial q^\text{qg}}{\partial t} + U \frac{\partial q^\text{qg}}{\partial x} + \frac{\partial \psi}{\partial x}
    \frac{\partial Q^\text{qg}}{\partial y} = \text{E} \, \nabla_h^2 q^\text{qg},
\,\,\,\,\,\,\  \text{for} \,\,\, 0 < z <1,
\end{align}
```
where $q^\text{qg}$ is the perturbation QG PV, and it is defined as
```math
\begin{align}
    q^\text{qg} = \nabla_h^2 \psi^\text{qg} +
    \frac{\partial}{\partial z}
    \left(\frac{1}{N^2} \frac{\partial \psi^\text{qg}}{\partial z}\right),
\end{align}
```

The variable $\psi^\text{qg}$ describes the QG perturbation streamfunction with
$u^\text{qg}=-\partial_y \psi^\text{qg}$ and $v^\text{qg}=\partial_x \psi^\text{qg}$.
The variable $Q^\text{qg}$ describes the QG PV of the basic state, which is defined as [pedlosky2013geophysical](@cite)
```math
\begin{align}
    Q^\text{qg} = -\frac{\partial U}{\partial y} + \frac{\partial}{\partial z}\left(\frac{B}{N^2} \right),
\end{align}
```
and the cross-front gradient of $Q^\text{qg}$ is defined as
```math
\begin{align}
    \frac{\partial Q^\text{qg}}{\partial y} = - \frac{\partial}{\partial z}\left(\frac{\partial_z U}{N^2} \right).
\end{align}
```

The linearized perturbation buoyancy equation at the top and the bottom boundary is
```math
\begin{align}
    \frac{\partial b^\text{qg}}{\partial t} + U \frac{\partial b^\text{qg}}{\partial x}
      + \frac{\partial \psi^\text{qg}}{\partial x}
    \frac{\partial B}{\partial y} = 0,
    \,\,\,\,\,\,\ \text{at} \, z=0 \,\ \text{and} \,\, 1,
\end{align}
```
where $b^\text{qg}=\partial_z \psi^\text{qg}$.

## Normal-mode solutions
Next, we seek normal-mode solutions for $\psi^\text{qg}$ and $q^\text{qg}$ in the form of
```math
\begin{align}
    [\psi^\text{qg}, q^\text{qg}] = \mathfrak{R}\big([\widetilde{\psi}^\text{qg},
  \widetilde{q}^\text{qg}] \big)(y, z) e^{i kx-\sigma t},
\end{align}
```
where $\widetilde{\psi}^\text{qg}$, $\widetilde{q}^\text{qg}$ are the eigenfunctions
of $\psi^\text{qg}$ and $q^\text{qg}$, respectively.
In terms of streamfunction $\psi^\text{qg}$,
```math
\begin{align}
    [(\sigma + i k U) - \text{E}] \mathscr{L}\widetilde{\psi}^\text{qg}
  + i k \partial_y Q^\text{qg} \widetilde{\psi}^\text{qg} &= 0, \,\,\,\,\  \text{for} \,\, 0 < z <1,
\\
    (\sigma + i k U_{-})\partial_z \widetilde{\psi}^\text{qg}_{-}
  + i k \partial_y B_{-} \widetilde{\psi}^\text{qg}_{-} &= 0, \,\,\,\,\, \text{at} \,\, z = 0,
\\
    (\sigma + i k U_{+})\partial_z \widetilde{\psi}^\text{qg}_{+}
  + i k \partial_y B_{+} \widetilde{\psi}^\text{qg}_{+} &= 0, \,\,\,\,\, \text{at} \,\, z = 1,
\end{align}
```
where $\mathscr{L}$ is a linear operator, and is defined as
$\mathscr{L} \equiv \mathcal{D}_h^2 + 1/N^2 \partial_z^2$,
where $\mathcal{D}_h^2 = (\partial_y^2 - k^2)$.
The subscripts $-,+$ denote the values of the fields at $z=0$ and $z=1$, respectively.

## Generalized eigenvalue problem
The above set of equations can be cast into a generalized eigenvalue problem
```math
\begin{align}
 AX= \lambda BX,
\end{align}
```
where $\lambda$ is the eigenvalue, and $X$ is the eigenvector.

## Implementation using the BiGSTARS equation DSL
The DSL allows us to write the equations in physical-space notation.
The `dx()` operator is automatically converted to `im*k` multiplication,
and the discretization is performed once with k-separated caching for
efficient wavenumber sweeps.

## Load required packages

````julia
using BiGSTARS
using Printf
````

## 1. Domain
x is the along-front direction (Fourier-transformed → wavenumber k)
y is the cross-front direction (Fourier, periodic)
z is the vertical direction (Chebyshev, rigid lids at 0 and 1)

````julia
domain = Domain(
    x = FourierTransformed(),
    y = Fourier(60, [0, 1]),
    z = Chebyshev(30, [0, 1])
)
````

````
Domain(x=FourierTransformed(), y=Fourier(N=60, [0.0,1.0]), z=Chebyshev(N=30, [0.0,1.0]))
````

## 2. Problem

````julia
prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)
````

````
EVP Problem
  Domain: Domain(x=FourierTransformed(), y=Fourier(N=60, [0.0,1.0]), z=Chebyshev(N=30, [0.0,1.0]))
  Variables: [:psi]
  Eigenvalue: sigma
  Parameters: 
  Equations: 0
  BCs: 0 (0 dynamic)
  Substitutions: 

````

## 3. Parameters and background state

````julia
Y, Z = gridpoints(domain, :y, :z)
H  = 1.0
Ri = 1.0

prob[:U]    = Z .- 0.5 * H     # along-front velocity: U(z) = z - 1/2
prob[:Ri]   = Ri               # Richardson number
prob[:E]    = 1e-12            # Ekman number (hyperviscosity)
prob[:dBdy] = -ones(length(Z)) # dB/dy = -1 for Eady basic state
````

````
30-element Vector{Float64}:
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
 -1.0
````

Background PV gradient (zero for the Eady problem)

````julia
prob[:dQdy] = zeros(length(Z))
````

````
30-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
````

## 4. Substitutions
QG PV operator: Lap = dx² + dy² + (1/Ri)*dz²
For Eady with constant N² = Ri: Lap(A) = dx(dx(A)) + dy(dy(A)) + (1/Ri)*dz(dz(A))

````julia
@substitution Lap(A) = dx(dx(A)) + dy(dy(A)) + Ri * dz(dz(A))
````

````
BiGSTARS.Substitution(:Lap, [:A], ((dx(dx(A)) + dy(dy(A))) + (Ri * dz(dz(A)))))
````

## 5. Governing equation (interior QG PV)
  sigma * Lap(psi) = U * dx(Lap(psi)) + dQdy * dx(psi) - E * Lap(Lap(psi))

````julia
@equation sigma * Lap(psi) = U * dx(Lap(psi)) + dQdy * dx(psi) - E * Lap(Lap(psi))
````

````
1-element Vector{BiGSTARS.Equation}:
 BiGSTARS.Equation((sigma * Lap(psi)), (((U * dx(Lap(psi))) + (dQdy * dx(psi))) - (E * Lap(Lap(psi)))))
````

## 6. Boundary conditions
Eady BCs: linearized buoyancy equation at z=0 and z=1
sigma * dz(psi) + U * dx(dz(psi)) + dBdy * dx(psi) = 0
These are eigenvalue-dependent (dynamic) BCs — the DSL detects sigma
and puts the sigma side into the B matrix, the rest into A.

````julia
@bc left(sigma * dz(psi)  + U * dx(dz(psi)) + dBdy * dx(psi)) = 0
@bc right(sigma * dz(psi) + U * dx(dz(psi)) + dBdy * dx(psi)) = 0
````

````
2-element Vector{BiGSTARS.BoundaryCondition}:
 BiGSTARS.BoundaryCondition(:left, :z, (((sigma * dz(psi)) + (U * dx(dz(psi)))) + (dBdy * dx(psi))), 0.0, true)
 BiGSTARS.BoundaryCondition(:right, :z, (((sigma * dz(psi)) + (U * dx(dz(psi)))) + (dBdy * dx(psi))), 0.0, true)
````

## 7. Discretize

````julia
cache = discretize(prob)
````

````
DiscretizationCache
  System size: 1800 x 1800 (1 variables, 1800 per var)
  A components (k-powers): [0, 1, 2, 3, 4]
  B components (k-powers): [0, 1, 2]
  Total nnz: 81206 (1.25% dense)

````

## 8. Solve over wavenumbers with adaptive shift

````julia
function solve_eady(cache, k_values, Ri)
    growth_rates = zeros(length(k_values))
    sigma_shift = 0.02

    for (i, k_val) in enumerate(k_values)
        A, B = assemble(cache, Float64(k_val))
        solver = EigenSolver(A, B; σ₀=sigma_shift, method=:Krylov, nev=1, which=:LR, sortby=:R)
        try
            solve!(solver; verbose=false)
            λ, _ = get_results(solver)
            growth_rates[i] = real(λ[1])
            sigma_shift = real(λ[1])  ## adapt shift for next wavenumber
        catch
            growth_rates[i] = NaN
        end
    end

    # Results
    valid_idx = findall(!isnan, growth_rates)
    if !isempty(valid_idx)
        idx_max = valid_idx[argmax(growth_rates[valid_idx])]
        @printf "Most unstable wavenumber: k = %.3f\n" k_values[idx_max]
        @printf "Maximum growth rate: sigma = %.6f\n" growth_rates[idx_max]
    else
        println("No wavenumber converged.")
    end

    # Compare with analytical Eady solution
    println("\nComparison with Eady (1949) analytical solution:")
    for k_test in [0.1, 0.5, 1.0, 1.5]
        mu = k_test * sqrt(Ri)
        arg1 = coth(0.5mu) - 0.5mu
        arg2 = 0.5mu - tanh(0.5mu)
        sigma_theory = arg1 * arg2 > 0 ? (1.0 / sqrt(Ri)) * sqrt(arg1 * arg2) : 0.0
        idx = argmin(abs.(collect(k_values) .- k_test))
        @printf "  k=%.1f: numerical=%.6f, analytical=%.6f\n" k_test growth_rates[idx] sigma_theory
    end

    return growth_rates
end

k_values = [0.1]
growth_rates = solve_eady(cache, k_values, Ri)
````

````
1-element Vector{Float64}:
 0.02882903442149731
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

