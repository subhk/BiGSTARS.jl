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
    \nabla \cdot \mathbf{u}^\prime &= 0,
```
