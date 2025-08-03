# ## Load required packages
using LazyGrids
using LinearAlgebra
using Printf
using StaticArrays
using SparseArrays
using SparseMatrixDicts
using FillArrays
using SpecialFunctions
using Parameters
using Test
using BenchmarkTools
using JLD2
using Dierckx
using Parameters: @with_kw
using CairoMakie

using BiGSTARS
using BiGSTARS: AbstractParams
using BiGSTARS: Problem, OperatorI, TwoDGrid

# ## Parameters
@with_kw mutable struct Params{T} <: AbstractParams
    L::T                = 1.0           # horizontal domain size
    H::T                = 1.0           # vertical domain size
    Ri::T               = 1.0           # the Richardson number 
    ε::T                = 0.1           # aspect ratio ε ≡ H/L
    k::T                = 0.1           # along-front wavenumber
    E::T                = 1.0e-8        # the Ekman number 
    Ny::Int64           = 24            # no. of y-grid points
    Nz::Int64           = 20            # no. of z-grid points
end
nothing #hide


function Interp2D_smooth(yn, zn, An, yint, zint)
    spl = Spline2D(yn, zn, transpose(An); s=0.0)
    A₀ = zeros(Float64, length(yint), length(zint))
    A₀ = [spl(yᵢ, zᵢ) for yᵢ ∈ yint, zᵢ ∈ zint]
    return A₀
end

"""
    calculation of u and v from w and ζ
    
    ```math
    -∇ₕ²\\hat{u} = ik∂ᶻ\\hat{w} + ∂ʸ\\hat{ζ}
    -∇ₕ²\\hat{v} =  ∂ʸᶻ\\hat{w} - ik\\hat{ζ} 
"""
function cal_u_v(X, params, grid, prob)
    Ny = params.Ny
    Nz = params.Nz
    N  = Ny * Nz

    u = zeros(ComplexF64, N)
    v = zeros(ComplexF64, N)

    w  = X[1:1N,1]
    ζ  = X[1N+1:2N,1]
    
    wʳ = real(w)    # real part of w
    wⁱ = imag(w)    # imaginary part of w
    
    ζʳ = real(ζ)
    ζⁱ = imag(ζ) 
    
    I⁰ = sparse(Matrix(1.0I, N, N)) 
    s₁ = size(I⁰, 1); 
    s₂ = size(I⁰, 2)

    ∇ₕ² = SparseMatrixCSC(Zeros(N, N))
    ∇ₕ² = (1.0 * prob.D²ʸ - 1.0 * params.kₓ^2 * I⁰)

    # Setup inverse operator (see utils.jl)
    H   = InverseLaplace(∇ₕ²)

    tmp1 = prob.Dᶻᴰ * wʳ; tmp2 = prob.Dᶻᴰ * wⁱ
    ∂zw  = @. tmp1 + 1.0im * tmp2

    tmp1 = prob.Dʸ * ζʳ; tmp2 = prob.Dʸ * ζⁱ
    ∂yζ  = @. tmp1 + 1.0im * tmp2

    tmp1 = prob.Dʸᶻᴰ * wʳ; tmp2 = prob.Dʸᶻᴰ * wⁱ
    ∂yzw = @. tmp1 + 1.0im * tmp2

    ## calculating `u`
    tmp1 = 1.0im * k * ∂zw + ∂yζ
    u = -1.0 * H(tmp1)

    ## calculating `v` 
    tmp1 = 1.0 * ∂yzw - 1.0im * kₓ * ζ
    v = -1.0 * H(tmp1)

    return u, v
end


function normalize_perturb_ke!(u_tilde, v_tilde, w_tilde, 
                            ζ_tilde, b_tilde, params, grid)

	KE = @. 0.5 * (u_tilde * conj(u_tilde) + v_tilde * conj(v_tilde) 
                    + params.ε^2 * w_tilde * conj(w_tilde)) |> real

    KE_yzavg = trapz((grid.z, grid.y), KE) 

    ## You need to multiply with a constant `1/ratio' such that 
    ## ther normalized perturbation energy is one, 
    ## where ratio² = Eₚ → ratio = √Eₚ
    ratio = √KE_yzavg

    @. u_tilde *= 1.0/ratio
    @. v_tilde *= 1.0/ratio
    @. w_tilde *= 1.0/ratio
	@. ζ_tilde *= 1.0/ratio 
    @. b_tilde *= 1.0/ratio

    ## making sure perturbation KE is 1 (after normalization)
	KE = @. 0.5 * (u_tilde * conj(u_tilde) + v_tilde * conj(v_tilde) 
                    + params.ε^2 * w_tilde * conj(w_tilde)) |> real

    KE_yzavg = trapz((grid.z, grid.y), KE) 
    @assert KE_yzavg ≈ 1.0

    return nothing
end

# ## function to plot the eigenfunctions
function plot_eigfun()

    ## here we are doing it for Stone (1971)
    filename = "stone_ms_eigenval.jld2"
	file = jldopen(filename, "r"); 
    k   = file["k"];
	λ   = file["λ"];
	X   = file["X"];
	close(file)

    ## problem parameters (make sure you've set all the correct parameters)
    params = Params{Float64}()

    ## Construct grid and derivative operators
    grid  = TwoDGrid(params)

    ## Construct the necesary operator
    ops  = OperatorI(params)
    prob = Problem(grid, ops)

    ## calculating the eigenfunctions: u and v
    u_tilde, v_tilde = cal_u_v(X, params, grid, prob)

    ## eigenfunction order: [w ζ b]ᵀ
    w_tilde = deepcopy(X[1:1N,1]   )
    ζ_tilde = deepcopy(X[1N+1:2N,1])
    b_tilde = deepcopy(X[2N+1:3N,1])

    ## reshaping the variables
    u_tilde = reshape( u_tilde, (length(grid.z), length(grid.y)) )
    v_tilde = reshape( v_tilde, (length(grid.z), length(grid.y)) )
    w_tilde = reshape( w_tilde, (length(grid.z), length(grid.y)) )
	ζ_tilde = reshape( ζ_tilde, (length(grid.z), length(grid.y)) )
    b_tilde = reshape( b_tilde, (length(grid.z), length(grid.y)) )  

    ## normalization the eigenfunction such that perturbation KE is 1.0
    normalize_perturb_ke!(u_tilde, v_tilde, w_tilde, 
                        ζ_tilde, b_tilde, params, grid)


    ## plotting the real part of eigenfunction u, v, w, b
	fig = Figure(fontsize=32, size = (1800, 640), )
    
    ax1 = Axis(fig[1, 1], xlabel=L"$y$", xlabelsize=40,
                          ylabel=L"$z$", ylabelsize=40, 
						  title=L"$\mathfrak{R}(u)$", titlesize=40)
	
    co = contourf!(grid.y, grid.z, real(u_tilde), 
		colormap=cgrad(:balance, rev=false),
        levels=levels₀, extendlow = :auto, extendhigh = :auto )
    
    tightlimits!(ax1)
    xlims!(0, 1)
    ylims!(minimum(z), maximum(z))

	ax1.yticks=([0, 0.5, 1], ["0", "0.5", "1"])
	ax1.xticks=([0, 0.5, 1], ["0", "0.5", "1"])

	ax2 = Axis(fig[1, 2], xlabel=L"$y$", xlabelsize=40,
                          ylabel=L"$z$", ylabelsize=40, 
						  title=L"$\mathfrak{R}(v)$", titlesize=40)
	
    co = contourf!(grid.y, grid.z, real(v_tilde), 
		colormap=cgrad(:balance, rev=false),
        levels=levels₀, extendlow = :auto, extendhigh = :auto )

	xlims!(0, 1)
	ylims!(0, 1)

	ax2.yticks=([0, 0.5, 1], ["0", "0.5", "1"])
	ax2.xticks=([0, 0.5, 1], ["0", "0.5", "1"])


	ax3 = Axis(fig[2, 1], xlabel=L"$y$", xlabelsize=40,
                          ylabel=L"$z$", ylabelsize=40, 
						  title=L"$\mathfrak{R}(w)$", titlesize=40)
	
    co = contourf!(grid.y, grid.z, real(w_tilde), 
		colormap=cgrad(:balance, rev=false),
        levels=levels₀, extendlow = :auto, extendhigh = :auto )

	xlims!(0, 1)
	ylims!(0, 1)

	ax3.yticks=([0, 0.5, 1], ["0", "0.5", "1"])
	ax3.xticks=([0, 0.5, 1], ["0", "0.5", "1"])

	ax4 = Axis(fig[2, 2], xlabel=L"$y$", xlabelsize=40,
                          ylabel=L"$z$", ylabelsize=40, 
						  title=L"$\mathfrak{R}(b)$", titlesize=40)
	
    co = contourf!(grid.y, grid.z, real(b_tilde), 
		colormap=cgrad(:balance, rev=false),
        levels=levels₀, extendlow = :auto, extendhigh = :auto )

	xlims!(0, 1)
	ylims!(0, 1)

	ax4.yticks=([0, 0.5, 1], ["0", "0.5", "1"])
	ax4.xticks=([0, 0.5, 1], ["0", "0.5", "1"])


	Label(fig[1, 1, TopLeft()], L"$(a)$", 
						fontsize=40, padding = (0, 0, 20, 0), halign = :right)
	Label(fig[1, 2, TopLeft()], L"$(b)$", 
						fontsize=40, padding = (0, 0, 20, 0), halign = :right)
	Label(fig[1, 3, TopLeft()], L"$(c)$", 
						fontsize=40, padding = (0, 0, 20, 0), halign = :right)
	Label(fig[1, 4, TopLeft()], L"$(d)$", 
						fontsize=40, padding = (0, 0, 20, 0), halign = :right)

    save("eigfun_stone.png", fig, px_per_unit=6)

    #fig

end

# ## calling the function
plot_eigfun()