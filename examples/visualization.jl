# ## Load required packages
using LazyGrids
using LinearAlgebra
using Printf
using SparseArrays
using Parameters
using JLD2
using Dierckx
using Parameters: @with_kw
using CairoMakie

using BiGSTARS
using BiGSTARS: AbstractParams, Problem, OperatorI, TwoDGrid

# ## Parameters struct 
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

"""
    Interp2D_smooth(yn, zn, An, yint, zint)

Smooth 2D interpolation using splines.
"""
function Interp2D_smooth(yn, zn, An, yint, zint)
    spl = Spline2D(yn, zn, transpose(An); s=0.0)
    return [spl(yᵢ, zᵢ) for yᵢ ∈ yint, zᵢ ∈ zint]
end

"""
    cal_u_v(X, params, grid, prob)

Calculate u and v velocity components from w and ζ using:
```math
-∇ₕ²û = ik∂ᶻŵ + ∂ʸζ̂
-∇ₕ²v̂ = ∂ʸᶻŵ - ikζ̂ 
```
"""
function cal_u_v(X, params, grid, prob)
    Ny, Nz = params.Ny, params.Nz
    N = Ny * Nz

    ## Extract state variables
    w = view(X, 1:N, 1)
    ζ = view(X, N+1:2N, 1)
    
    ## Separate real and imaginary parts for efficient computation
    wʳ, wⁱ = real(w), imag(w)
    ζʳ, ζⁱ = real(ζ), imag(ζ)
    
    ## Create Laplacian operator
    I⁰ = sparse(I, N, N)
    ∇ₕ² = prob.D²ʸ - params.k^2 * I⁰
    
    ## Setup inverse operator
    H = InverseLaplace(∇ₕ²)

    ## Compute derivatives efficiently
    ∂zw = complex.(prob.Dᶻᴰ * wʳ, prob.Dᶻᴰ * wⁱ)
    ∂yζ = complex.(prob.Dʸ * ζʳ, prob.Dʸ * ζⁱ)
    ∂yzw = complex.(prob.Dʸᶻᴰ * wʳ, prob.Dʸᶻᴰ * wⁱ)

    ## Calculate velocity components
    u = -H(1im * params.k * ∂zw + ∂yζ)
    v = -H(∂yzw - 1im * params.k * ζ)

    return u, v
end

"""
    normalize_perturb_ke!(u_tilde, v_tilde, w_tilde, ζ_tilde, b_tilde, params, grid)

Normalize perturbation fields so that kinetic energy equals 1.
"""
function normalize_perturb_ke!(u_tilde, v_tilde, w_tilde, ζ_tilde, b_tilde, params, grid)
    ## Calculate kinetic energy
    KE = @. 0.5 * (abs2(u_tilde) + abs2(v_tilde) + params.ε^2 * abs2(w_tilde))
    
    ## Domain-averaged kinetic energy
    KE_yzavg = trapz((grid.z, grid.y), KE)
    
    ## Normalization factor
    ratio = √KE_yzavg
    
    ## Normalize all fields
    fields = (u_tilde, v_tilde, w_tilde, ζ_tilde, b_tilde)
    for field in fields
        field .*= inv(ratio)
    end
    
    ## Optional verification - uncomment for debugging
    ## KE_normalized = @. 0.5 * (abs2(u_tilde) + abs2(v_tilde) + params.ε^2 * abs2(w_tilde))
    ## KE_yzavg_normalized = trapz((grid.z, grid.y), KE_normalized)
    ## @assert KE_yzavg_normalized ≈ 1.0 "Normalization failed: KE = $KE_yzavg_normalized"
    
    return nothing
end

"""
    create_contour_plot!(ax, grid, field, title; colormap=:balance, nlevels=8)

Helper function to create consistent contour plots.
"""
function create_contour_plot!(ax, grid, field, title; colormap=:balance, nlevels=8)
    max_val = maximum(abs, field)
    levels = range(-0.8max_val, 0.8max_val, length=nlevels)
    
    contourf!(ax, grid.y, grid.z, transpose(field), 
              colormap=cgrad(colormap, rev=false),
              levels=levels, 
              extendlow=:auto, 
              extendhigh=:auto)
    
    ax.xlabel = L"$y$"
    ax.ylabel = L"$z$"
    ax.title = title
    xlims!(ax, 0, maximum(grid.y))
    ylims!(ax, 0, 1)
    
    return nothing
end

"""
    load_eigenfunction_data(filename::String)

Load eigenfunction data from JLD2 file with error handling.
"""
function load_eigenfunction_data(filename::String)
    if !isfile(filename)
        error("File $filename not found. Please check the file path.")
    end
    
    return jldopen(filename, "r") do file
        k = file["k"]
        λ = file["λ"]
        X = file["X"]
        @printf "Loaded data: k = %.6f, λ = %.6f + %.6fi\n" k real(λ) imag(λ)
        return k, λ, X
    end
end

"""
    setup_problem(k::Real)

Initialize parameters, grid, and problem operators.
"""
function setup_problem(k::Real)
    params = Params{Float64}(k=k)
    grid = TwoDGrid(params)
    ops = OperatorI(params)
    prob = Problem(grid, ops)
    
    return params, grid, prob
end

"""
    extract_and_reshape_fields(X, params, grid, prob)

Extract and reshape all field variables from the state vector.
"""
function extract_and_reshape_fields(X, params, grid, prob)
    Ny, Nz = params.Ny, params.Nz
    N = Ny * Nz
    
    ## Calculate velocity components
    u_tilde, v_tilde = cal_u_v(X, params, grid, prob)
    
    ## Extract other fields
    w_tilde = X[1:N, 1]
    ζ_tilde = X[N+1:2N, 1]
    b_tilde = X[2N+1:3N, 1]
    
    ## Reshape to 2D grids
    grid_shape = (length(grid.z), length(grid.y))
    fields = (u_tilde, v_tilde, w_tilde, ζ_tilde, b_tilde)
    reshaped_fields = map(field -> reshape(field, grid_shape), fields)
    
    return reshaped_fields
end

"""
    plot_eigenfunctions(filename::String="stone_ms_eigenval.jld2"; 
                       save_plot::Bool=false, output_name::String="eigfun_stone.png")

Main function to plot eigenfunctions with improved organization and error handling.
"""
function plot_eigenfunctions!(filename::String="stone_ms_eigenval.jld2"; 
                           save_plot::Bool=false, 
                           output_name::String="eigfun_stone.png",
                           figure_size::Tuple{Int,Int}=(1800, 1152))
    
    ## Load data
    k, λ, X = load_eigenfunction_data(filename)
    
    ## Setup problem
    params, grid, prob = setup_problem(k)
    
    ## Extract and reshape fields
    u_tilde, v_tilde, w_tilde, ζ_tilde, b_tilde = extract_and_reshape_fields(X, params, grid, prob)
    
    ## Normalize fields
    normalize_perturb_ke!(u_tilde, v_tilde, w_tilde, ζ_tilde, b_tilde, params, grid)
    
    ## Create figure
    fig = Figure(fontsize=32, size=figure_size)
    
    ## Plot configuration
    plot_configs = [
        (fig[1, 1], real(u_tilde), L"$\mathfrak{R}(\tilde{u})$"),
        (fig[1, 2], real(v_tilde), L"$\mathfrak{R}(\tilde{v})$"),
        (fig[2, 1], real(w_tilde), L"$\mathfrak{R}(\tilde{w})$"),
        (fig[2, 2], real(ζ_tilde), L"$\mathfrak{R}(\tilde{\zeta})$")
    ]
    
    ## Create subplots
    axes = []
    for (ax_pos, field_data, title) in plot_configs
        ax = Axis(ax_pos, xlabelsize=40, ylabelsize=40, titlesize=40)
        create_contour_plot!(ax, grid, field_data, title)
        push!(axes, ax)
    end
    
    ## Add subplot labels using text! which is more universally compatible
    labels = [L"$(a)$", L"$(b)$", L"$(c)$", L"$(d)$"]
    for (i, ax) in enumerate(axes)
        text!(ax, 0.05, 0.95, text=labels[i], 
              space=:relative, fontsize=40, 
              color=:white, align=(:left, :top))
    end
    
    ## Save if requested
    if save_plot
        save(output_name, fig, px_per_unit=6)
        @printf "Figure saved as: %s\n" output_name
    end
    
    fig
    return nothing
end


## Example usage with error handling
function main()
    try
        plot_eigenfunctions!(save_plot=true)
        return true
    catch e
        @error "Error in eigenfunction plotting" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

# Call the main function
main()