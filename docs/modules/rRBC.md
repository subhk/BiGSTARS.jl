# rotating Rayleigh Benard convection (rRBC)

### Problem setup

In this module, we do a linear stability analysis of a 2D rotating Rayleigh-Bernard case where the domain is periodic in the ``y``-direction, 
in the ``x``-direction is of infinite extent and vertically bounded. The reason to choose this simple case is because we can find an analytical solution for this case. Here we seek stationary solutions at the marginal state, i.e., ```\sigma = 0```. The background temperature profile is given by 
```math
\overline{\theta} = 1 - z.
```
The non-dimensional form of the equations governing the perturbation is given by \citep{chandrasekhar2013hydrodynamic}
\begin{subequations}
\begin{align}
    \frac{E}{Pr} \frac{\partial \mathbf{u}^\prime}{\partial t} 
    + \hat{z} \times \mathbf{u}^\prime &=
    -\nabla p + Ra \theta^\prime \hat{z} + E \nabla^2 \mathbf{u}^\prime, 
\\
    \frac{\partial \theta^\prime}{\partial t} 
    &= \mathbf{u}^\prime \cdot \hat{z} + \nabla^2 \theta^\prime,
\\
    \nabla \cdot \mathbf{u}^\prime &= 0,
\end{align}
\end{subequations}
where $E=\nu/(fH^2)$ is the Ekman number and $Ra = g\alpha \Delta T/(f \kappa)$ ($\Delta T$ is the temperature difference between the bottom and the top walls) is the modified Rayleigh number.
By applying the operators $(\nabla \times \nabla \times)$ and $(\nabla \times)$ and taking the $z$-component of the equations and assuming wave-like perturbations as done previously, we obtained the equations for vertical velocity $u_z^\prime$, vertical vorticity $\omega_z^\prime$ and temperature $\theta^\prime$,
