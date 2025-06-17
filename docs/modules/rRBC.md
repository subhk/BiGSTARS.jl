# rotating Rayleigh Benard convection (rRBC)

### Problem setup

In this module, we do a linear stability analysis of a 2D rotating Rayleigh-Bernard case where the domain is periodic in the ``y``-direction, 
in the ``x``-direction is of infinite extent and vertically bounded. The reason to choose this simple case is because we can find an analytical solution for this case. Here we seek stationary solutions at the marginal state, i.e., ```\sigma = 0```. The background temperature profile is given by 
```math
\overline{\theta} = 1 - z.
```
The non-dimensional form of the equations governing the perturbation is given by 
```math
    \frac{E}{Pr} \frac{\partial \mathbf{u}^\prime}{\partial t} 
    + \hat{z} \times \mathbf{u}^\prime =
    -\nabla p + Ra \theta^\prime \hat{z} + E \nabla^2 \mathbf{u}^\prime,
```
```math
    \frac{\partial \theta^\prime}{\partial t} 
    = \mathbf{u}^\prime \cdot \hat{z} + \nabla^2 \theta^\prime,
```
```math
    \nabla \cdot \mathbf{u}^\prime = 0,
```
where ```E=\nu/(fH^2)``` is the Ekman number and ```Ra = g\alpha \Delta T/(f \kappa)```, ```\Delta T``` is the temperature difference between the bottom and the top walls) is the modified Rayleigh number.
By applying the operators ```(\nabla \times \nabla \times)``` and ```(\nabla \times)``` and taking the ```z```-component of the equations and assuming wave-like perturbations as done previously, we obtained the equations for vertical velocity ```u_z^\prime```, vertical vorticity ```\omega_z^\prime``` and temperature ```\theta^\prime```,
```math
\begin{align}
    E \mathcal{D}^4 u_z - \partial_z \omega_z &= -Ra \mathcal{D}_H^2 \theta,
\\
    E \mathcal{D}^2 \omega_z + \partial_z u_z &= 0,
\\
    \mathcal{D}^2 b + u_z &= 0.
\end{align}
```
The boundary conditions are: 
```math
\begin{align}
    u_z = \partial_z^2 u_z = \partial_z \omega_z = \theta = 0
    \,\,\,\,\,\ \text{at} \,\,\, z=0,1
\end{align}
```
The above governing equations with the boundary conditions are transformed into the form 
```math
\mathcal{L} \mathbf{X} = \lambda \mathcal{M} \mathbf{X}
```
where ```\lambda=Ra``` is the eigenvalue. 
