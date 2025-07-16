using LinearAlgebra, FFTW, ToeplitzMatrices

"""
    FourierDiff(n::Int)

Fourier differentiation on periodic domain [0, 2π).

# Usage
```julia
fd = FourierDiff(16)
x = fd.grid
f = sin.(x)

# Compute derivatives
df = fd(f, 1)      # First derivative
d2f = fd(f, 2)     # Second derivative
```
"""
struct FourierDiff
    n::Int
    grid::Vector{Float64}
    dx::Float64
    matrices::Dict{Int, Matrix{Float64}}
    
    function FourierDiff(n::Int)
        n < 3 && throw(ArgumentError("Grid size must be ≥ 3"))
        dx = 2π / n
        grid = collect(range(0, 2π, length=n+1)[1:n])
        new(n, grid, dx, Dict{Int, Matrix{Float64}}())
    end
end

# Callable interface
(fd::FourierDiff)(f::AbstractVector, order::Int=1) = differentiate(fd, f, order)

function derivative_matrix(fd::FourierDiff, order::Int)
    order < 0 && throw(ArgumentError("Derivative order must be ≥ 0"))
    haskey(fd.matrices, order) && return fd.matrices[order]
    
    fd.matrices[order] = _compute_matrix(fd.n, order, fd.dx)
    return fd.matrices[order]
end

function differentiate(fd::FourierDiff, f::AbstractVector, order::Int=1)
    length(f) != fd.n && throw(ArgumentError("Function length must match grid size"))
    return derivative_matrix(fd, order) * f
end

function _compute_matrix(n::Int, order::Int, dx::Float64)
    order == 0 && return Matrix{Float64}(I, n, n)
    
    col1, row1 = _compute_vectors(n, order, dx)
    return Matrix(Toeplitz(col1, row1))
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
fourier_diff(n::Int, order::Int=1) = (fd = FourierDiff(n); (fd.grid, derivative_matrix(fd, order)))

# Utility functions
Base.show(io::IO, fd::FourierDiff) = print(io, "FourierDiff(n=$(fd.n), cached=$(sort(collect(keys(fd.matrices)))))")
clear_cache!(fd::FourierDiff) = (empty!(fd.matrices); fd)
memory_usage(fd::FourierDiff) = sum(length(m) for m in values(fd.matrices)) * sizeof(Float64) / 1024^2

# Testing
function test_fourier_diff(n::Int=16)
    fd = FourierDiff(n)
    f = sin.(fd.grid)
    
    df_num = fd(f, 1)
    df_exact = cos.(fd.grid)
    d2f_num = fd(f, 2)
    d2f_exact = -sin.(fd.grid)
    
    err1 = maximum(abs.(df_num - df_exact))
    err2 = maximum(abs.(d2f_num - d2f_exact))
    
    println("n=$n: 1st deriv error = $(err1), 2nd deriv error = $(err2)")
    return err1 < 1e-10 && err2 < 1e-10
end

# Demo
if abspath(PROGRAM_FILE) == @__FILE__
    fd = FourierDiff(32)
    f = sin.(fd.grid)
    
    println("Derivatives of sin(x):")
    println("1st: max error = $(maximum(abs.(fd(f,1) - cos.(fd.grid))))")
    println("2nd: max error = $(maximum(abs.(fd(f,2) - (-sin.(fd.grid)))))")
    println("3rd: max error = $(maximum(abs.(fd(f,3) - (-cos.(fd.grid)))))")
    
    println("\nCached matrices: $(fd)")
    test_fourier_diff()
end