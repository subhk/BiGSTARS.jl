# Baroclinic instability

The resulting nondimensional, linearized Boussinesq equations of motion under the ``f``-plane approximation are given by
```math
\begin{align}
    \frac{D \mathbf{u}}{Dt}
    + \Big(v \frac{\partial U}{\partial y} + w \frac{\partial U}{\partial z} \Big) \hat{x}
    + \hat{z} \times \mathbf{u} &=
    -\nabla p + \frac{1}{\epsilon} b \hat{z} + E \nabla^2 \mathbf{u}, \\
    \frac{Db}{Dt}
    +  v \frac{\partial B}{\partial y} + w \frac{\partial B}{\partial z} &= \frac{E}{Pr} \nabla^2 b, \\
    \nabla \cdot \mathbf{u} &= 0,
\end{align}
```
where 
```math
D/Dt \equiv \partial/\partial t + U (\partial/\partial x)$
```
is the material derivative, ``\mathbf{u} \equiv (u, v, \epsilon w)`` is the velocity perturbation, ``\epsilon=H/R`` is the aspect ratio, ``p`` is the pressure perturbation, and ``b`` is the buoyancy perturbation. The operator 
```math
\nabla \equiv (\partial/\partial x, \partial/\partial y, (1/\epsilon) \partial/\partial z),
```
```math
\nabla^2 \equiv \partial^2/\partial x^2 + \partial^2/\partial y^2 + (1/\epsilon^2) \partial^2/ \partial z^2,
```
where 
```math
\nabla_h^2 \equiv \partial^2 /\partial x^2 + \partial^2/\partial y^2.
```
To eliminate pressure, following [teed2010rapidly@citet, we apply the operator ``\hat{z} \cdot \nabla \times \nabla \times``  and ``\hat{z} \cdot \nabla \times`` to the above momentum equation. This procedure yields governing equations of three perturbation variables, the vertical velocity ``w``, the vertical vorticity ``\zeta \, (=\hat{z} \cdot \nabla \times \mathbf{u})``, and the buoyancy ``b`` 
```math
\begin{align}
    \frac{D}{Dt}\nabla^2 {w} 
    + \frac{1}{\epsilon^2} \frac{\partial \zeta}{\partial z} 
    &= \frac{1}{\epsilon^2} \nabla_h^2 b + E \nabla^4 w,
\\
    \frac{D \zeta}{Dt}
    - \frac{\partial U}{\partial z}\frac{\partial w}{\partial y}
    - \frac{\partial w}{\partial z} &= E \nabla^2 \zeta, 
\\
    \frac{Db}{Dt}
    + v \frac{\partial B}{\partial y} + 
    w \frac{\partial B}{\partial z}
    &= \frac{E}{Pr} \nabla^2 b,
\end{align}
```
where 
```math
\nabla_h^2 \equiv \partial^2 /\partial x^2 + \partial^2/\partial y^2$
```
The benefit of using the above sets of equations is that it enables us to examine the instability at an along-front wavenumber ``k \to 0``. 
The horizontal velocities ``u`` and ``v`` are related to the vertical velocity ``w`` and vertical vorticity ``\zeta`` by the identities, 
```math
\begin{align}
    \nabla_h^2 u &= -\frac{\partial \zeta}{\partial y} - \frac{\partial^2 w}{\partial x \partial z}, 
\\
    \nabla_h^2 v &= \frac{\partial \zeta}{\partial x} - \frac{\partial^2 w}{\partial y \partial z}.    
\end{align}
```
In deriving the above equations, we make use of the continuity equation and the definition of vertical vorticity ``\zeta``.

## Normal mode 
Next we consider normal-mode perturbation solutions in the form of 
```math
\begin{align}
    [w, \zeta, b](x,y,z,t) = \mathfrak{R}\big([\tilde{w}, \, \tilde{\zeta}, \, \tilde{b}](y, z) \, e^{i kx + \sigma t}\big),
\end{align}
```
where the symbol ``\mathfrak{R}`` denotes the real part and a variable with `tilde' denotes an eigenfunction. The variable 
``\sigma=\sigma_r + i \sigma_i``. The real part represents the growth rate, and the imaginary part 
shows the frequency of the  perturbation. 

Finally following systems of differential equations are obtained,
```math
\begin{align}
    (i k U - E \mathcal{D}^2) \mathcal{D}^2 \tilde{w}
    + \epsilon^{-2} \partial_z \tilde{\zeta}
    - \epsilon^{-2} \mathcal{D}_h^2 \tilde{b} &= -\sigma \mathcal{D}^2 \tilde{w},
\\
    - \partial_z U \partial_y \tilde{w}
    - \partial_z \tilde{w}
    + \left(ik U - E \mathcal{D}^2 \right) \tilde{\zeta} &= -\sigma \tilde{\zeta},
\\
    \partial_z B \tilde{w} + \partial_y B  \tilde{v} + 
    \left[ik U - E \mathcal{D}^2 \right] \tilde{b} &= -\sigma \tilde{b}, 
\end{align}
```
where 
```math
\mathcal{D}^4  = (\mathcal{D}^2 )^2 = \big(\partial_y^2 +
(1/\epsilon^2)\partial_z^2 - k^2\big)^2, \,\,\,\, \text{and} \,\, \mathcal{D}_h^2 = (\partial_y^2 - k^2).
```

The eigenfunctions ``\tilde{u}``, ``\tilde{v}`` are related to ``\tilde{w}``, ``\tilde{\zeta}`` by the relations 
```math
\begin{align}
    -\mathcal{D}_h^2 \tilde{u} &= i k \partial_{z} \tilde{w} + \partial_y \tilde{\zeta},
\\   
    -\mathcal{D}_h^2 \tilde{v} &= \partial_{yz} \tilde{w} -  i k \tilde{\zeta}.
\end{align}
```

We choose periodic boundary conditions in the ``y``-direction and free-slip, rigid lid, with zero buoyancy flux in the ``z`` direction, i.e., 
```math
\begin{align}
    \tilde{w} = \partial_{zz} \tilde{w} = 
    \partial_z \tilde{\zeta} = \partial_z \tilde{b} = 0, 
    \,\,\,\,\,\,\, \text{at} \,\,\, {z}=0, 1.
\end{align}
```

The boundary conditions are implemented in 
```@docs
GeophysicalFlows.TwoDNavierStokes.Equation
```

The above sets of equations with the boundary conditions can be expressed as a standard generalized eigenvalue problem,
```math
\begin{align}
    \mathsfit{A} \mathsf{X}= \sigma \mathsfit{B} \mathsf{X},   
\end{align}
```
where ``\sigma`` is the eigenvalue, ``\mathsf{X}=[\tilde{w}, \tilde{\zeta}, \tilde{b}]^T`` is the eigenvector and the matrices ``\mathsfit{A}``, ``\mathsfit{B}`` are the complex and real non-symmetric matrices, respectively. 


![Alt text](images/stone1971_Ri2.png)
