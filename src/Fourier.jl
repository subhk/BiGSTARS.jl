using LinearAlgebra, FFTW, ToeplitzMatrices

"""
    FourierDiff(n::Int, L::Float64=2π, orders::Vector{Int}=Int[])

Fourier differentiation on periodic domain [0, L).

# Usage
```julia
# Basic usage - compute matrices on demand
fd = FourierDiff(16, 4.0)  # 16 points on domain [0, 4)
x = fd.grid
f = sin.(2π * x / fd.L)    # Function values on physical domain

# Pre-compute specific derivative orders
fd = FourierDiff(16, 4.0, [1, 2, 4])  # Pre-compute 1st, 2nd, and 4th derivatives
df = fd(f, 1)      # First derivative (already computed)
d2f = fd(f, 2)     # Second derivative (already computed)
d4f = fd(f, 4)     # Fourth derivative (already computed)
```
"""
struct FourierDiff
    n::Int
    L::Float64
    grid::Vector{Float64}
    dx::Float64
    matrices::Dict{Int, Matrix{Float64}}
    
    function FourierDiff(n::Int, L::Float64=2π, orders::Vector{Int}=Int[])
        n < 3 && throw(ArgumentError("Grid size must be ≥ 3"))
        L <= 0 && throw(ArgumentError("Domain length L must be positive"))
        any(o -> o < 0, orders) && throw(ArgumentError("All derivative orders must be ≥ 0"))
        
        dx = L / n
        grid = collect(range(0, L, length=n+1)[1:n])
        matrices = Dict{Int, Matrix{Float64}}()
        
        # Pre-compute requested derivative matrices
        for order in orders
            matrices[order] = _compute_matrix(n, order, L)
        end
        
        new(n, L, grid, dx, matrices)
    end
end

# Callable interface
(fd::FourierDiff)(f::AbstractVector, order::Int=1) = differentiate(fd, f, order)

function derivative_matrix(fd::FourierDiff, order::Int)
    order < 0 && throw(ArgumentError("Derivative order must be ≥ 0"))
    haskey(fd.matrices, order) && return fd.matrices[order]
    
    fd.matrices[order] = _compute_matrix(fd.n, order, fd.L)
    return fd.matrices[order]
end

function differentiate(fd::FourierDiff, f::AbstractVector, order::Int=1)
    length(f) != fd.n && throw(ArgumentError("Function length must match grid size"))
    return derivative_matrix(fd, order) * f
end

function _compute_matrix(n::Int, order::Int, L::Float64)
    order == 0 && return Matrix{Float64}(I, n, n)
    
    # Compute base matrix for [0, 2π) domain
    col1, row1 = _compute_vectors(n, order, 2π / n)
    base_matrix = Matrix(Toeplitz(col1, row1))
    
    # Apply domain transformation: [0, 2π) → [0, L)
    # D^k_L = (2π/L)^k * D^k_{2π}
    transform_factor = (2π / L)^order
    
    return transform_factor * base_matrix
end

function _compute_vectors(n::Int, order::Int, dx::Float64)
    if order > 2
        # FFT for higher derivatives
        nfo1 = (n - 1) ÷ 2
        nfo2 = iseven(order) && iseven(n) ? -n/2 : 0.0
        
        mwave = 1.0im * vcat(0:nfo1, nfo2, -nfo1:-1)
        impulse = vcat(1.0, zeros(n-1))
        col1 = real(ifft(mwave.^order .* fft(impulse)))
        
        if isodd(order)
            col1 = vcat(0.0, col1[2:n])
            return col1, -col1
        else
            return col1, col1
        end
    end
    
    # Analytical for 1st and 2nd derivatives
    nn1, nn2 = (n - 1) ÷ 2, (n - 1 + 1) ÷ 2
    half_dx = 0.5 * dx
    col1 = zeros(n)
    
    if order == 1
        if iseven(n)
            topc = @. 1.0 / tan((1:nn2) * half_dx)
            col1[2:nn2+1] = 0.5 * [(-1)^k for k in 1:nn2] .* topc
            col1[nn2+2:n] = -0.5 * [(-1)^k for k in nn2+1:n-1] .* reverse(topc[1:nn1])
        else
            topc = @. 1.0 / sin((1:nn2) * half_dx)
            col1[2:nn2+1] = 0.5 * [(-1)^k for k in 1:nn2] .* topc
            col1[nn2+2:n] = 0.5 * [(-1)^k for k in nn2+1:n-1] .* reverse(topc[1:nn1])
        end
        return col1, -col1
        
    else # order == 2
        if iseven(n)
            col1[1] = -π^2 / (3.0 * dx^2) - 1.0/6.0
            topc = @. 1.0 / sin((1:nn2) * half_dx)^2
            col1[2:nn2+1] = -0.5 * [(-1)^k for k in 1:nn2] .* topc
            col1[nn2+2:n] = -0.5 * [(-1)^k for k in nn2+1:n-1] .* reverse(topc[1:nn1])
        else
            col1[1] = -π^2 / (3.0 * dx^2) + 1.0/12.0
            topc = @. (1.0 / tan((1:nn2) * half_dx)) / sin((1:nn2) * half_dx)
            col1[2:nn2+1] = -0.5 * [(-1)^k for k in 1:nn2] .* topc
            col1[nn2+2:n] = 0.5 * [(-1)^k for k in nn2+1:n-1] .* reverse(topc[1:nn1])
        end
        return col1, col1
    end
end

# Convenience function
fourier_diff(n::Int, L::Float64=2π, orders::Vector{Int}=Int[], order::Int=1) = (fd = FourierDiff(n, L, orders); (fd.grid, derivative_matrix(fd, order)))

# Utility functions
Base.show(io::IO, fd::FourierDiff) = print(io, "FourierDiff(n=$(fd.n), L=$(fd.L), cached=$(sort(collect(keys(fd.matrices)))))")
clear_cache!(fd::FourierDiff) = (empty!(fd.matrices); fd)
memory_usage(fd::FourierDiff) = sum(length(m) for m in values(fd.matrices)) * sizeof(Float64) / 1024^2

# Testing
function test_fourier_diff(n::Int=16, L::Float64=2π, orders::Vector{Int}=[1,2])
    fd = FourierDiff(n, L, orders)
    
    # Test with sin function scaled to domain [0, L)
    f = sin.(2π * fd.grid / L)
    
    df_num = fd(f, 1)
    df_exact = (2π / L) * cos.(2π * fd.grid / L)
    d2f_num = fd(f, 2)
    d2f_exact = -(2π / L)^2 * sin.(2π * fd.grid / L)
    
    err1 = maximum(abs.(df_num - df_exact))
    err2 = maximum(abs.(d2f_num - d2f_exact))
    
    println("n=$n, L=$L, orders=$orders: 1st deriv error = $(err1), 2nd deriv error = $(err2)")
    return err1 < 1e-10 && err2 < 1e-10
end

# Demo
if abspath(PROGRAM_FILE) == @__FILE__
    # Test on standard domain [0, 2π) with pre-computed derivatives
    println("Testing on standard domain [0, 2π) with pre-computed derivatives:")
    fd1 = FourierDiff(32, 2π, [1, 2, 3])  # Pre-compute 1st, 2nd, 3rd derivatives
    f1 = sin.(fd1.grid)
    
    println("Derivatives of sin(x):")
    println("1st: max error = $(maximum(abs.(fd1(f1,1) - cos.(fd1.grid))))")
    println("2nd: max error = $(maximum(abs.(fd1(f1,2) - (-sin.(fd1.grid)))))")
    println("3rd: max error = $(maximum(abs.(fd1(f1,3) - (-cos.(fd1.grid)))))")
    
    # Test on custom domain [0, 4) with specific derivatives
    println("\nTesting on custom domain [0, 4) with specific derivatives:")
    L = 4.0
    fd2 = FourierDiff(32, L, [1, 2, 4])  # Pre-compute 1st, 2nd, 4th derivatives
    f2 = sin.(2π * fd2.grid / L)  # One period of sin over [0, 4)
    
    println("Derivatives of sin(2π*x/L):")
    df_exact = (2π / L) * cos.(2π * fd2.grid / L)
    d2f_exact = -(2π / L)^2 * sin.(2π * fd2.grid / L)
    d4f_exact = (2π / L)^4 * sin.(2π * fd2.grid / L)
    
    println("1st: max error = $(maximum(abs.(fd2(f2,1) - df_exact)))")
    println("2nd: max error = $(maximum(abs.(fd2(f2,2) - d2f_exact)))")
    println("4th: max error = $(maximum(abs.(fd2(f2,4) - d4f_exact)))")
    
    # Test computing 3rd derivative on demand (not pre-computed)
    println("3rd: max error = $(maximum(abs.(fd2(f2,3) - (-(2π / L)^3 * cos.(2π * fd2.grid / L)))))")
    
    println("\nCached matrices: $(fd2)")
    test_fourier_diff(16, 2π, [1,2])
    test_fourier_diff(16, 4.0, [1,2,4])
end