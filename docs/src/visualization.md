# Visualization

This document presents a Julia implementation for visualizing eigenfunctions from 
the stability analysis using the script `examples/visualization.jl`.

## Overview

The code performs eigenfunction analysis for a two-dimensional geophysical flow problem, computing velocity components and plotting the real parts of perturbation fields. 

## Key Components

### Parameters Structure

The analysis uses a comprehensive parameter set from the Stone exmaple:

```julia
@with_kw mutable struct Params{T} <: AbstractParams
    L::T = 1.0          # horizontal domain size
    H::T = 1.0          # vertical domain size
    Ri::T = 1.0         # Richardson number 
    ε::T = 0.1          # aspect ratio ε ≡ H/L
    k::T = 0.1          # along-front wavenumber
    E::T = 1.0e-8       # Ekman number 
    Ny::Int64 = 24      # no. of y-grid points
    Nz::Int64 = 20      # no. of z-grid points
end
```

### Velocity Calculation

The code computes horizontal velocity components (u, v) from the vertical velocity (w) and vorticity (ζ) using the relations:

- **u-component**: `-∇ₕ²ũ = ik∂ᶻw̃ + ∂ʸζ̃`
- **v-component**: `-∇ₕ²ṽ = ∂ʸᶻw̃ - ikζ̃`

### Normalization

The perturbation fields are normalized so that the kinetic energy equals unity:

```julia
KE = 0.5 * (|u|² + |v|² + ε²|w|²)
```

This ensures consistent comparison between different eigenmode solutions.

## Visualization Output

The main plotting function `plot_eigenfunctions!()` generates a 2×2 subplot arrangement showing the real parts of the perturbation fields:

### Expected Figure Structure

The visualization displays four contour plots arranged as follows:

- **(a) Real part of u-velocity perturbation** - Shows horizontal velocity component in x-direction
- **(b) Real part of v-velocity perturbation** - Shows horizontal velocity component in y-direction  
- **(c) Real part of w-velocity perturbation** - Shows vertical velocity component
- **(d) Real part of vorticity perturbation** - Shows the vorticity field

### Sample Output Visualization

Here's what the typical eigenfunction structure looks like for the Stone example:
![Alt text](eigfun_stone.png)

## Usage Example

```julia
# Basic usage - plot and display
plot_eigenfunctions!()

# Save the figure
plot_eigenfunctions!(save_plot=true, output_name="my_eigenfunctions.png")

# Custom figure size
plot_eigenfunctions!(figure_size=(2400, 1536), save_plot=true)
```
