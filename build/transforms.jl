#══════════════════════════════════════════════════════════════════════════════#
#  Physical-space ↔ coefficient-space transforms and evaluation               #
#══════════════════════════════════════════════════════════════════════════════#

"""
    chebyshev_evaluate(c, x) -> Vector{Float64}

Evaluate a Chebyshev expansion with coefficients `c` at points `x` ∈ [-1, 1].
Uses Clenshaw's algorithm for numerical stability.
"""
function chebyshev_evaluate(c::AbstractVector, x::AbstractVector)
    N = length(c)
    result = similar(x, Float64)
    for (i, xi) in enumerate(x)
        # Clenshaw recurrence for T polynomials
        b_kp1 = 0.0
        b_kp2 = 0.0
        for k in N:-1:2
            b_temp = c[k] + 2xi * b_kp1 - b_kp2
            b_kp2 = b_kp1
            b_kp1 = b_temp
        end
        result[i] = c[1] + xi * b_kp1 - b_kp2
    end
    return result
end

"""
    differentiate(f, domain, coord; order=1, filter=:none) -> Vector

Spectrally differentiate a function given as physical-space values on the grid.

**Chebyshev** directions: converts to T coefficients, applies the differentiation
operator chain D_{order-1} * ⋯ * D_0, converts the C^(order) result back to
physical space. Chebyshev methods do not suffer from Gibbs ringing for smooth
non-periodic functions.

**Fourier** directions: applies `(ik)^order` in Fourier space. An optional spectral
filter can suppress Gibbs ringing from under-resolved gradients:

- `:none` — no filter (default; exact for smooth periodic functions)
- `:exp`  — exponential filter `exp(-α(|k|/k_max)^p)` with α=36, p=36
            (sharp cutoff that preserves well-resolved modes)
- `:2/3`  — 2/3 dealiasing rule (zeros out the top 1/3 of modes)
- A `Function` — custom filter `σ(k, k_max)` applied as multiplier

# Examples
```julia
domain = Domain(z = Chebyshev(N=64, lower=0.0, upper=1.0))
z = gridpoints(domain, :z)
U = @. z^2
dUdz = differentiate(U, domain, :z)            # ≈ 2z
d2Udz2 = differentiate(U, domain, :z; order=2) # ≈ 2

domain_y = Domain(y = Fourier(N=64, L=2π))
y = gridpoints(domain_y, :y)
g = sin.(y)
dgdy = differentiate(g, domain_y, :y)                  # cos(y), no filter
dgdy = differentiate(g, domain_y, :y; filter=:exp)     # cos(y), filtered
dgdy = differentiate(g, domain_y, :y; filter=:2/3)     # cos(y), dealiased
```
"""
function differentiate(f::AbstractVector, domain::Domain, coord::Symbol;
                       order::Int=1, filter::Union{Symbol,Function}=:none)
    @assert order >= 1 "Derivative order must be ≥ 1"
    spec = domain.coords[coord]

    if spec isa ChebyshevBasisSpec
        N = spec.N
        @assert length(f) == N "Input length ($(length(f))) must match grid size ($N)"
        scale = 2.0 / (spec.upper - spec.lower)

        # Physical values → T coefficients
        x_ref = chebyshev_points(N)
        c = chebyshev_coefficients(Float64.(f))

        # Apply differentiation chain: D_{order-1} * ⋯ * D_0 (each scaled)
        dc = Float64.(c)
        for p in 0:order-1
            D = differentiation_operator(p, N)
            dc = scale .* (D * dc)
        end

        # dc is now in C^(order) basis — convert back to T via S_{0→order}^{-1}
        S = get_conversion_operator(domain, coord, 0, order)
        c_T = Matrix(S) \ dc
        return chebyshev_evaluate(c_T, x_ref)

    elseif spec isa FourierBasisSpec
        N = spec.N
        @assert length(f) == N "Input length ($(length(f))) must match grid size ($N)"
        f_hat = fft(f) / N

        # Apply spectral filter before differentiation
        if filter !== :none
            k = fftfreq(N, N)   # wavenumber indices
            k_max = N ÷ 2
            σ = _fourier_filter(filter, k, k_max)
            f_hat = f_hat .* σ
        end

        D = fourier_diff_operator(N, spec.L, order)
        df_hat = D * f_hat
        return real.(ifft(df_hat * N))

    else
        error("Cannot differentiate in FourierTransformed direction :$coord (no grid)")
    end
end

"""Build the spectral filter vector for Fourier modes."""
function _fourier_filter(filter, k, k_max)
    if filter === :exp
        # Exponential filter: preserves low modes, sharply damps near Nyquist
        # exp(-α * (|k|/k_max)^p) with α=36, p=36 (Vandeven 1991)
        return @. exp(-36.0 * (abs(k) / k_max)^36)
    elseif filter === Symbol("2/3")
        # 2/3 dealiasing: zero out top 1/3 of modes
        return @. Float64(abs(k) ≤ 2 * k_max / 3)
    elseif filter isa Function
        return @. filter(k, k_max)
    else
        error("Unknown filter: $filter. Use :none, :exp, Symbol(\"2/3\"), or a Function(k, k_max)")
    end
end

"""
    to_coefficients(f_physical, coord_type, N) -> Vector

Transform physical-space values to spectral coefficients.

- `:chebyshev` → Chebyshev T coefficients via DCT
- `:fourier` → Fourier coefficients via FFT (scaled by 1/N)
"""
function to_coefficients(f::AbstractVector, coord_type::Symbol)
    if coord_type == :chebyshev
        return chebyshev_coefficients(f)
    elseif coord_type == :fourier
        return fft(f) / length(f)
    else
        error("Unknown coordinate type: $coord_type. Expected :chebyshev or :fourier")
    end
end

"""
    to_physical(c, coord_type, x) -> Vector

Transform spectral coefficients back to physical-space values.

- `:chebyshev` → evaluate Chebyshev expansion at points x
- `:fourier` → inverse FFT
"""
function to_physical(c::AbstractVector, coord_type::Symbol;
                     x::Union{AbstractVector, Nothing}=nothing)
    if coord_type == :chebyshev
        isnothing(x) && error("Must provide evaluation points x for Chebyshev transform")
        return chebyshev_evaluate(c, x)
    elseif coord_type == :fourier
        N = length(c)
        return real.(ifft(c * N))
    else
        error("Unknown coordinate type: $coord_type")
    end
end
