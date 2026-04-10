#══════════════════════════════════════════════════════════════════════════════#
#                    ULTRASPHERICAL SPECTRAL OPERATORS                         #
#                                                                              #
#  Sparse differentiation, conversion, and multiplication operators for the    #
#  ultraspherical spectral method (Olver & Townsend, 2013).                    #
#                                                                              #
#  Key references:                                                             #
#    S. Olver & A. Townsend, "A fast and well-conditioned spectral method,"    #
#    SIAM Review 55(3), 462-489, 2013.                                         #
#══════════════════════════════════════════════════════════════════════════════#

#──────────────────────────────────────────────────────────────────────────────#
#  1. Conversion operator  S_p : C^(p) -> C^(p+1)                             #
#──────────────────────────────────────────────────────────────────────────────#

"""
    conversion_operator(p, N) -> SparseMatrixCSC

Build the N x N sparse conversion matrix S_p that converts expansions in
the C^(p) ultraspherical basis to the C^(p+1) basis.

For p = 0 (Chebyshev T -> C^(1)):
  S_0[1,1] = 1
  S_0[n,n] = 1/2         for n >= 2
  S_0[n-2,n] = -1/2      for n >= 3

For p >= 1 (C^(p) -> C^(p+1)):
  S_p[1,1] = 1
  S_p[n,n] = p/(n-1+p)   for n >= 2
  S_p[n-2,n] = -p/(n-1+p)  for n >= 3

Returns an N x N sparse matrix.
"""
function conversion_operator(p::Int, N::Int)
    @assert p >= 0 "Order p must be non-negative"
    @assert N >= 1 "Size N must be at least 1"

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    if p == 0
        # S_0: T -> C^(1)
        # (1,1) entry
        push!(rows, 1); push!(cols, 1); push!(vals, 1.0)

        # Diagonal: S_0[n,n] = 1/2 for n >= 2
        for n in 2:N
            push!(rows, n); push!(cols, n); push!(vals, 0.5)
        end

        # Super-diagonal (offset +2): S_0[n-2,n] = -1/2 for n >= 3
        # From T_{n-1} = (U_{n-1} - U_{n-3})/2, the -1/2 goes to row n-2
        for n in 3:N
            push!(rows, n - 2); push!(cols, n); push!(vals, -0.5)
        end
    else
        # S_p for p >= 1: C^(p) -> C^(p+1)
        # (1,1) entry
        push!(rows, 1); push!(cols, 1); push!(vals, 1.0)

        # Diagonal: S_p[n,n] = p/(n-1+p) for n >= 2
        for n in 2:N
            push!(rows, n); push!(cols, n); push!(vals, p / (n - 1 + p))
        end

        # Super-diagonal (offset +2): S_p[n-2,n] = -p/(n-1+p) for n >= 3
        # From C^(p)_{n-1} expansion in C^(p+1) basis
        for n in 3:N
            push!(rows, n - 2); push!(cols, n); push!(vals, -p / (n - 1 + p))
        end
    end

    return sparse(rows, cols, vals, N, N)
end


#──────────────────────────────────────────────────────────────────────────────#
#  2. Differentiation operator  D_p : C^(p) -> C^(p+1)                        #
#──────────────────────────────────────────────────────────────────────────────#

"""
    differentiation_operator(p, N) -> SparseMatrixCSC

Build the N x N sparse differentiation matrix D_p that differentiates
expansions in the C^(p) basis and expresses the result in the C^(p+1) basis.

For p = 0 (d/dx in T -> C^(1)):
  D_0[n,n+1] = n   for n >= 1   (1-indexed, from d/dx T_n = n*C^(1)_{n-1})

For p >= 1 (d/dx in C^(p) -> C^(p+1)):
  D_p[n,n+1] = 2p   for all n >= 1   (constant superdiagonal)

Returns an N x N sparse matrix (superdiagonal, bandwidth 1).

Note: The output has N rows and N columns. The last row is always zero since
differentiating an (N-1)-degree polynomial yields an (N-2)-degree polynomial.
"""
function differentiation_operator(p::Int, N::Int)
    @assert p >= 0 "Order p must be non-negative"
    @assert N >= 2 "Size N must be at least 2"

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    if p == 0
        # D_0: T -> C^(1)
        # From d/dx T_n = n * C^(1)_{n-1}, so D_0[n, n+1] = n (1-indexed)
        for n in 1:(N - 1)
            push!(rows, n); push!(cols, n + 1); push!(vals, Float64(n))

        end
    else
        # D_p for p >= 1: C^(p) -> C^(p+1)
        # D_p[n,n+1] = 2p for all n >= 1
        for n in 1:(N - 1)
            push!(rows, n); push!(cols, n + 1); push!(vals, 2.0 * p)
        end
    end

    return sparse(rows, cols, vals, N, N)
end


#──────────────────────────────────────────────────────────────────────────────#
#  3. Multiplication operator  M_f in Chebyshev T basis                        #
#──────────────────────────────────────────────────────────────────────────────#

"""
    multiplication_operator(f_coeffs, N; tol=1e-14) -> SparseMatrixCSC

Build the N x N sparse multiplication operator M_f in the Chebyshev T basis,
such that M_f * c gives the Chebyshev coefficients of f(x) * g(x) where c
are the Chebyshev coefficients of g(x).

Uses the linearisation identity:
  T_m(x) * T_n(x) = (T_{|m-n|}(x) + T_{m+n}(x)) / 2

The matrix is constructed column by column: column j corresponds to multiplying
f(x) by T_{j-1}(x), and the result is expressed in the T basis.

`f_coeffs` should be a vector of Chebyshev coefficients of f(x), i.e.,
f(x) = f_coeffs[1]*T_0 + f_coeffs[2]*T_1 + f_coeffs[3]*T_2 + ...

Entries below `tol` in magnitude are dropped.
"""
function multiplication_operator(f_coeffs::AbstractVector, N::Int; tol::Float64=1e-14)
    @assert N >= 1 "Size N must be at least 1"

    # Pad or truncate f_coeffs to work with
    nf = length(f_coeffs)

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    # Column j corresponds to multiplication by T_{j-1}
    for j in 1:N
        n = j - 1  # T_n index (0-based)

        for k in 1:nf
            m = k - 1  # f coefficient index (0-based), coefficient f_coeffs[k]
            fk = f_coeffs[k]

            if abs(fk) < tol
                continue
            end

            # T_m * T_n = (T_{|m-n|} + T_{m+n}) / 2
            # Contribution to row |m-n|+1 and row m+n+1

            idx1 = abs(m - n) + 1   # row for T_{|m-n|}
            idx2 = m + n + 1        # row for T_{m+n}

            if idx1 <= N
                push!(rows, idx1); push!(cols, j); push!(vals, fk / 2.0)
            end
            if idx2 <= N
                push!(rows, idx2); push!(cols, j); push!(vals, fk / 2.0)
            end
        end
    end

    M = sparse(rows, cols, vals, N, N)

    # The sparse constructor sums duplicate entries, which is what we want.
    # Drop tiny entries.
    droptol!(M, tol)

    return M
end


#──────────────────────────────────────────────────────────────────────────────#
#  4. Chebyshev-Gauss-Lobatto points                                           #
#──────────────────────────────────────────────────────────────────────────────#

"""
    chebyshev_points(N, a=-1.0, b=1.0) -> Vector{Float64}

Compute N Chebyshev-Gauss-Lobatto points on the interval [a, b].

On [-1,1]:  x_j = cos(j * pi / (N-1))  for j = 0, ..., N-1

These are then linearly mapped to [a, b].
"""
function chebyshev_points(N::Int, a::Float64=-1.0, b::Float64=1.0)
    @assert N >= 2 "Need at least 2 points"
    @assert a < b "Require a < b"

    # Points on [-1,1]
    x_ref = [cos(j * pi / (N - 1)) for j in 0:(N - 1)]

    # Map to [a, b]:  x = (b-a)/2 * (x_ref + 1) + a
    x = @. (b - a) / 2.0 * (x_ref + 1.0) + a

    return x
end


#──────────────────────────────────────────────────────────────────────────────#
#  5. Chebyshev coefficients via DCT-I                                         #
#──────────────────────────────────────────────────────────────────────────────#

"""
    chebyshev_coefficients(f_values) -> Vector{Float64}

Compute Chebyshev expansion coefficients from function values at
Chebyshev-Gauss-Lobatto points (in the ordering from `chebyshev_points`,
i.e., from x=1 down to x=-1).

Uses the DCT-I (type 1) relation. Given N function values at the CGL points
x_j = cos(j*pi/(N-1)), j=0,...,N-1, the Chebyshev coefficients are:

  c_k = (2 / (N-1)) * sum_{j=0}^{N-1} '' f(x_j) * cos(k*j*pi/(N-1))

where '' means the first and last terms are halved.

The returned vector has length N, where c[k] is the coefficient of T_{k-1}.
"""
function chebyshev_coefficients(f_values::AbstractVector)
    N = length(f_values)
    @assert N >= 2 "Need at least 2 values"

    # DCT-I via FFTW REDFT00.
    # REDFT00 on a length-N input computes:
    #   Y_k = X_0 + (-1)^k X_{N-1} + 2 sum_{j=1}^{N-2} X_j cos(pi*j*k/(N-1))
    # The Chebyshev coefficient c_k (double-prime sum) satisfies:
    #   c_k = Y_k / (N-1), with c_0 and c_{N-1} halved.
    n = N - 1
    Y = FFTW.r2r(Float64.(f_values), FFTW.REDFT00)

    coeffs = Y ./ n
    coeffs[1] /= 2.0
    coeffs[N] /= 2.0

    return coeffs
end
