````markdown
# Shift and Invert Method

The **shift-and-invert** transformation is a common technique for solving the generalized eigenvalue problem

```math
A x = \lambda B x
````

by focusing on eigenvalues near a target shift $\sigma$:

```math
(A - \sigma B)^{-1} B x = \mu x,
\quad \mu = (\lambda - \sigma)^{-1}.
```

## Key Advantages

* **Selective targeting:** Eigenvalues $\lambda$ closest to $\sigma$ correspond to the largest magnitudes of $\mu$, enabling efficient extraction via Krylov solvers (e.g., Arnoldi, Lanczos).
* **Accelerated convergence:** Inverting the shifted operator $(A - \sigma B)$ compresses the spectrum so that desired eigenmodes dominate.

## Algorithm Outline

1. **Select a shift** $\sigma$ near the eigenvalue of interest.
2. **Factor** or set up an efficient solver for $A - \sigma B$ (e.g., LU, sparse direct, or preconditioned iterative solver).
3. **Iterate** with a Krylov-based eigensolver on $(A - \sigma B)^{-1} B$ to compute $\mu$.
4. **Recover** the original eigenvalue:

   ```math
   \lambda = \sigma + \mu^{-1}.
   ```

