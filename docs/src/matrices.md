# Spectral Operators

BiGSTARS.jl uses **coefficient-space spectral methods** for fully sparse operator matrices. All resolved directions operate in spectral coefficient space — no dense matrices anywhere in the system.

## Chebyshev Direction: Ultraspherical Method

For non-periodic (bounded) directions, BiGSTARS uses the ultraspherical spectral method [olver2013fast](@cite). The solution is represented in Chebyshev T polynomials, and differentiation maps into a hierarchy of ultraspherical (Gegenbauer) bases: T -> C^(1) -> C^(2) -> ...

### Key Operators

**Differentiation** `D_p`: maps from C^(p) to C^(p+1) basis.
- `D_0`: T -> C^(1). Superdiagonal entries: `D_0[n, n+1] = n` (from d/dx T_n = n * U_{n-1})
- `D_p` (p >= 1): C^(p) -> C^(p+1). Constant superdiagonal: `D_p[n, n+1] = 2p`

**Conversion** `S_p`: converts from C^(p) to C^(p+1) basis (same function, different basis).
- `S_0`: T -> C^(1). Diagonal + super-diagonal (bandwidth 2)
- `S_p` (p >= 1): `S_p[n,n] = p/(n-1+p)`, `S_p[n-2,n] = -p/(n-1+p)`

**Multiplication** `M_f`: multiply by a function f(x) in Chebyshev T basis.
- Uses the identity: T_m * T_n = (T_{|m-n|} + T_{m+n})/2
- Almost-banded: banded when f is smooth (coefficients decay rapidly)

All operators are **sparse** — the assembled GEVP matrices remain sparse throughout.

### Equation Assembly

For a 2nd-order equation `a(z)*d2psi/dz2 + b(z)*dpsi/dz + c(z)*psi = sigma*psi`:
- Highest derivative order p = 2, so the equation lives in C^(2) basis
- 2nd derivative term: `M_a^(2) * D_1 * D_0` (already in C^(2))
- 1st derivative term: `M_b^(2) * S_1 * D_0` (convert result to C^(2))
- 0th order term: `M_c^(2) * S_1 * S_0` (convert T to C^(2))

### Domain Scaling

Derivative operators automatically account for the physical domain extent:

- **Chebyshev** on [a, b]: each derivative picks up a factor `2/(b-a)` from the chain rule. For `Chebyshev(N, [0, 1])`, the first derivative is scaled by 2, the second by 4, etc.
- **Fourier** on [a, b): wavenumbers are `m * 2pi / (b-a)`. For `Fourier(N, [0, 2pi])`, wavenumber m has frequency m. For `Fourier(N, [0, 1])`, wavenumber m has frequency `2pi * m`.

Both constructor forms pass the domain extent through to all operators:
```julia
z = Chebyshev(30, [0, 1])                   # compact form
z = Chebyshev(N=30, lower=0.0, upper=1.0)   # keyword form (equivalent)
y = Fourier(60, [0, 1])                      # compact form
y = Fourier(N=60, L=1.0)                     # keyword form (equivalent, domain [0, L))
```

### Sparsity Summary

| Operator | Structure | Bandwidth |
|----------|-----------|-----------|
| D_p (differentiation) | Superdiagonal | 1 |
| S_p (conversion) | Diagonal + super-diagonal | 2 |
| M_f (multiplication) | Almost-banded | depends on smoothness of f |

### API

```julia
# Low-level (used internally by the DSL):
D0 = differentiation_operator(0, N)   # D_0: T -> C^(1)
D1 = differentiation_operator(1, N)   # D_1: C^(1) -> C^(2)
S0 = conversion_operator(0, N)        # S_0: T -> C^(1)
S1 = conversion_operator(1, N)        # S_1: C^(1) -> C^(2)
Mf = multiplication_operator(f_coeffs, N)  # M_f in T basis

# Grid and transforms:
x = chebyshev_points(N, a, b)         # Chebyshev-Gauss-Lobatto points on [a,b]
c = chebyshev_coefficients(f_values)  # DCT: values -> T coefficients
f = chebyshev_evaluate(c, x)          # Clenshaw: T coefficients -> values
```

## Fourier Direction: Coefficient Space

For periodic directions, BiGSTARS operates in Fourier coefficient space where derivatives are diagonal.

### Differentiation

In Fourier coefficient space, the p-th derivative is a diagonal matrix:

```math
D_y^{(p)}[j,j] = \left(i m_j \frac{2\pi}{L}\right)^p
```

where m_j are the wavenumbers [0, 1, ..., N/2-1, -N/2, ..., -1].

### Field Multiplication

Pointwise multiplication becomes circular convolution in coefficient space — a circulant matrix. For smooth fields, this is effectively banded.

### API

```julia
# Low-level (used internally by the DSL):
D = fourier_diff_operator(N, L, p)          # p-th derivative, diagonal
Mf = fourier_multiply_operator(f_hat, N)    # circulant multiplication

# Grid:
x = fourier_points(N, L)                    # N equally-spaced points on [0, L)
```

## Boundary Conditions: Boundary Bordering

BCs are applied via **boundary bordering**: the last rows of the banded operator matrix are replaced with BC rows. This preserves the sparse structure.

In coefficient space, evaluating at a boundary means:
- psi(-1) = sum_n (-1)^n c_n (left boundary)
- psi(+1) = sum_n c_n (right boundary)

For derivative BCs, evaluate the derivative polynomial at the boundary.

### Generalized BC Form

The most general linear BC: alpha_0*psi + alpha_1*dz(psi) + alpha_2*dz(dz(psi)) + ... = g

This covers Dirichlet, Neumann, Robin, higher-order, coupled, and inhomogeneous BCs — all through the same mechanism.

### API

```julia
# In the DSL:
@bc left(psi) = 0               # Dirichlet
@bc right(dz(psi)) = 0          # Neumann
@bc left(3*psi + dz(psi)) = 0   # Robin
@bc right(dz(dz(psi))) = 0      # Higher-order
@bc left(psi + b) = 0           # Coupled
@bc right(psi) = 1.0            # Inhomogeneous
```
