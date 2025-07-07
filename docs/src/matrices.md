# Construction of Chebyshev differentiation matrix
A standard approach is followed in the construction of the differentiation matrices 
\citep{trefethen2000spectral}. The transformed Gauss–Lobatto points for 
$z \in [0, 1]$ are given by
```math
\begin{align}
    z_j = \frac{1}{2} \cos{(j\pi/N_z)} + \frac{1}{2},
    \,\,\,\,
    j = 0, \cdots, N_z.
\end{align}
and the first-order Chebyshev differentiation matrix is given by
```
