#══════════════════════════════════════════════════════════════════════════════#
#                  FOURIER COEFFICIENT-SPACE OPERATORS                          #
#                                                                              #
#  Sparse differentiation and multiplication operators for Fourier directions  #
#  operating entirely in spectral coefficient space.                           #
#══════════════════════════════════════════════════════════════════════════════#

# ──────────────────────────────────────────────────────────────────────────────
#  1. Differentiation operator (diagonal in Fourier space)
# ──────────────────────────────────────────────────────────────────────────────

"""
    fourier_diff_operator(N, L, p) -> SparseMatrixCSC{ComplexF64}

Build the N×N sparse diagonal differentiation matrix for the p-th derivative
in Fourier coefficient space on domain [0, L).

Wavenumbers: m = fftfreq(N, N) = [0, 1, ..., N/2-1, -N/2, -N/2+1, ..., -1]
Entry (j,j) = (im * m_j * 2π / L)^p
"""
function fourier_diff_operator(N::Int, L::Float64, p::Int)
    @assert N >= 1 "N must be positive"
    @assert L > 0 "L must be positive"
    @assert p >= 0 "Derivative order must be non-negative"

    wavenumbers = fftfreq(N, N)  # [0, 1, ..., N/2-1, -N/2, ..., -1]
    d = [(im * m * 2π / L)^p for m in wavenumbers]
    return spdiagm(0 => d)
end

# ──────────────────────────────────────────────────────────────────────────────
#  2. Multiplication operator (circulant in Fourier space)
# ──────────────────────────────────────────────────────────────────────────────

"""
    fourier_multiply_operator(f_hat, N; tol=1e-14) -> SparseMatrixCSC{ComplexF64}

Build the N×N sparse circulant multiplication matrix in Fourier coefficient space.

Given (scaled) FFT coefficients `f_hat` of a function f(x), constructs M_f such that
M_f * g_hat gives the FFT coefficients of f(x)*g(x).

In Fourier space, pointwise multiplication becomes circular convolution:
  (f*g)_m = Σ_k f_k * g_{m-k (mod N)}

`f_hat` should be the FFT of f divided by N (i.e., the spectral coefficients).
"""
function fourier_multiply_operator(f_hat::AbstractVector, N::Int; tol::Float64=1e-14)
    @assert length(f_hat) >= 1

    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]

    for i in 1:N
        for j in 1:N
            # M[i,j] = f_hat[(i-j) mod N + 1]  (1-indexed circular)
            k = mod(i - j, N) + 1
            val = k <= length(f_hat) ? f_hat[k] : zero(ComplexF64)
            if abs(val) > tol
                push!(rows, i); push!(cols, j); push!(vals, val)
            end
        end
    end

    return sparse(rows, cols, vals, N, N)
end

# ──────────────────────────────────────────────────────────────────────────────
#  3. Fourier grid points
# ──────────────────────────────────────────────────────────────────────────────

"""
    fourier_points(N, L) -> Vector{Float64}

Return N equally-spaced points on [0, L), excluding the right endpoint.
"""
function fourier_points(N::Int, L::Float64)
    @assert N >= 1 && L > 0
    return [j * L / N for j in 0:N-1]
end
