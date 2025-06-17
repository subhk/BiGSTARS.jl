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


### Result

For the parameter of ```E=10^{-4}```, the obtained critical Rayleigh number $Ra_c=189.7$ (matched with [Chandrasekhar](@citet)) analysis for horizontally infinite domain). Due to the finite aspect ratio, our stability analysis shows a finite number of convective rolls (figure ) with ```y```-wavenumber ```m_c=28```, i.e., there are ```28``` pairs of rolls in the horizontal direction. According to [Chandrasekhar](@citet) analysis, the resultant wavenumber ```a_c=28.02``` for this parameter regime, and it can be shown that for a finite ```L_x```, the resultant wavenumber ```a``` is related as
```math
    a^2 = \Big( \frac{2m\pi}{L_y} \Big)^2 + k^2,
```
where ```k``` is the ```x```-wavenumber and in this analysis ```k=0```. 
So for ```L_y=2\pi``` and ```a_c=28.02```, we obtained ```m \approx 28``` which we also get from stability analysis.

![Alt text](images/rRBC.png)
