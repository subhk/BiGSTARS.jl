# rotating Rayleigh Benard convection (rRBC)

### Basic Equations

This module solves the *quasi-linear* quasi-geostrophic barotropic vorticity equation on a beta
plane of variable fluid depth ``H - h(x, y)``. Quasi-linear refers to the dynamics that *neglects*
the eddy--eddy interactions in the eddy evolution equation after an eddy--mean flow decomposition,
e.g.,

```math
\phi(x, y, t) = \overline{\phi}(y, t) + \phi'(x, y, t) ,
```
