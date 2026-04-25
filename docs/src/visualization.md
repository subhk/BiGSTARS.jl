# Post-Processing and Visualization

## Computing Fields from Eigenvectors

After solving the eigenvalue problem, use `@compute` to evaluate any expression on the eigenvector fields. The syntax is the same DSL used for defining equations.

### Setup

```julia
cache = discretize(prob)
results = solve(cache, [k_val]; sigma_0=0.02)

# Set up the post-processing context:
@compute_setup cache results[1].eigenvectors[:, 1] k_val
```

### Evaluate Expressions

```julia
# Reconstruct velocity components:
@compute v = -dy(dz(w)) + dx(zeta)
@compute u = -dx(dz(w)) - dy(zeta)

# Derived quantities:
@compute vorticity = dx(dx(psi)) + dy(dy(psi))
@compute buoyancy = dz(psi)

# Expressions with parameters and substitutions:
@compute momentum_flux = U * dy(b) - E * D2(zeta)

# Products of fields (nonlinear):
@compute buoyancy_flux = w * b
@compute kinetic_energy = u * u + v * v + eps2 * w * w
```

All DSL features work inside `@compute`: derivatives (`dx`, `dy`, `dz`), parameters, substitutions, and derived variables.

### Multiple Eigenmodes

Switch eigenvector to compute fields for a different mode:

```julia
# Most unstable mode:
@compute_setup cache results[1].eigenvectors[:, 1] k_val
@compute v1 = -dy(dz(w)) + dx(zeta)

# Second mode:
@compute_setup cache results[1].eigenvectors[:, 2] k_val
@compute v2 = -dy(dz(w)) + dx(zeta)
```

### Different Wavenumbers

```julia
for (i, k) in enumerate(k_values)
    @compute_setup cache results[i].eigenvectors[:, 1] k
    @compute v = -dy(dz(w)) + dx(zeta)
    # ... plot v at each k
end
```

## Reconstructing Derived Variables

For variables defined via `@derive`, use `reconstruct`:

```julia
v_coeffs = reconstruct(cache, prob, eigvec, k_val, :v)
```

For a scalar solve, `k_val` is the same wavenumber used to assemble and solve the matrices. In domains with several transformed directions, `reconstruct` currently uses the scalar value for each transformed direction; use direct expression evaluation only when that matches the assembled problem.

Or get all fields at once:

```julia
fields = reconstruct_all(cache, prob, eigvec, k_val)
# fields[:w], fields[:zeta], fields[:b] — from eigenvector
# fields[:v] — reconstructed via @derive inverse operator
```

## Converting to Physical Space

Results from `@compute` are in spectral coefficient space. Convert for plotting:

```julia
z = gridpoints(domain, :z)
v_physical = to_physical(v, :chebyshev; x=z)

y = gridpoints(domain, :y)
f_physical = to_physical(f, :fourier)
```

## Complete Post-Processing Example

```julia
using BiGSTARS

# ... (define and solve the problem) ...
cache = discretize(prob)
results = solve(cache, [0.1]; sigma_0=0.02)

# Post-process the most unstable mode:
@compute_setup cache results[1].eigenvectors[:, 1] 0.1

# Velocity reconstruction:
@compute v = -dy(dz(w)) + dx(zeta)
@compute u = -dx(dz(w)) - dy(zeta)

# Convert to physical space for plotting:
z = gridpoints(domain, :z)
v_phys = to_physical(v, :chebyshev; x=z)
u_phys = to_physical(u, :chebyshev; x=z)
w_phys = to_physical(reconstruct_all(cache, prob, results[1].eigenvectors[:, 1], 0.1)[:w], :chebyshev; x=z)
```

## Sample Output

Typical eigenfunction structure from the Stone (1971) example:

![Eigenfunction visualization](eigfun_stone.png)
