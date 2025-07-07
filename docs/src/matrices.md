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
```
and the first-order Chebyshev differentiation matrix is given by
```math
\begin{equation}
  (\mathrm{D}_z)_{ij} = \begin{cases}
     \dfrac{2N_z^2+1}{3},  \,\,\,\,\, i=j=0, 
\\
    \dfrac{c_i}{c_j} \dfrac{(-1)^{i+j}}{z_i-z_j},
     \,\,\,\,\, i \neq j,
     \,\,\,\,\, c_i
     \begin{cases} 
        2, \,\,\,\,\, i=0,N_z, \\
        1, \,\,\,\,\, \text{otherwise},
     \end{cases}
\\
    \dfrac{-\cos{(j\pi/N_z)}}{1-\cos^2{(j\pi/N_z)}},
    \,\,\,\,\, 0 < i = j < N_z,
\\
    -\dfrac{2N_z^2+1}{3}, \,\,\,\,\, i=j=N_z.
  \end{cases}
\end{equation}
```

To construct the Chebyshev differentiation matrix:
```markdown
```jldoctest
using BiGSTARS

N = 10 # number of grid points
z, D = cheb(N-1)

nothing # hide
```

# Construction of Fourier differentiation matrix
For $y \in [0,L_y]$, the first-order Fourier differentiation matrix for even $N_y$ is,
```math
\begin{equation}
    (\mathrm{D}_y)_{ij} = \begin{cases}
        0, \,\,\,\,\, i=j, 
\\
        \dfrac{\pi}{L_y} (-1)^{i-j} \cot{\left(\dfrac{(i-j)h}{2} \right)}
        \,\,\,\,\, i \neq j,
    \end{cases}
\end{equation}
```
and for odd $N_y$,
```math
\begin{equation}
    (\mathrm{D}_y)_{ij} = \begin{cases}
        0, \,\,\,\,\, i=j, 
\\
        \dfrac{\pi}{L_y} (-1)^{i-j} \csc{\left(\dfrac{(i-j)h}{2} \right)}
        \,\,\,\,\, i \neq j,
    \end{cases}
\end{equation}
```
where $h=2\pi/N_y$.

To construct the Fourier differentiation matrix:
```markdown
```jldoctest
using BiGSTARS

N    = 10 # number of grid points
mder = 1  # order of Fourier derivative
y, D = FourierDiff(N, mder)

The domain size of `y` is 0 to 2π.
To convert into an arbitrary domain [0, L]: 

y0 = L/(2π) * y
𝒟 = (2π/L)^1 * D

For an n-th derivative:

y, D = FourierDiff(N, n)
𝒟   = (2π/L)^n * D


nothing # hide
```
