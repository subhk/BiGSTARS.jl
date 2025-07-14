### File: `ComputeDerivatives.jl`

# module ComputeDerivatives

# export compute_derivatives, compute_first_derivatives_fourier, compute_second_derivatives_fourier,
#        compute_first_derivatives_chebyshev, compute_second_derivatives_chebyshev

"""
    compute_first_derivatives_fourier(B₀, U₀, y)

Compute first derivatives of `B₀` and `U₀` using Fourier differentiation matrices.
Returns: ∂ʸB₀, ∂ᶻB₀, ∂ʸU₀, ∂ᶻU₀
"""
function compute_first_derivatives_fourier(U₀::Matrix{T}, 
                                            B₀::Matrix{T}, 
                                            y::AbstractVector{T}) where T

    ∂ʸU₀ = gradient(U₀, y, dims=2)
    ∂ʸB₀ = gradient(B₀, y, dims=2)

    return ∂ʸU₀, ∂ʸB₀
end

"""
    compute_second_derivatives_fourier(U₀, B₀, y)

Compute second derivatives of `U₀` using Fourier differentiation matrices.
Returns: ∂ʸʸU₀, ∂ᶻᶻU₀, ∂ʸᶻU₀
"""
function compute_second_derivatives_fourier(U₀::Matrix{T}, 
                                            B₀::Matrix{T}, 
                                            y::AbstractVector{T}) where T

    ∂ʸU₀ = gradient2(U₀, y, dims=2)
    ∂ʸB₀ = gradient2(B₀, y, dims=2)

    return ∂ʸU₀, ∂ʸB₀
end

"""
    compute_first_derivatives_chebyshev(U₀, B₀, Dᶻ)

Compute first derivatives of `B₀` and `U₀` using Chebyshev differentiation matrices.
Returns: ∂ʸB₀, ∂ᶻB₀, ∂ʸU₀, ∂ᶻU₀
"""
function compute_first_derivatives_chebyshev(U₀::Matrix{T}, 
                                            B₀::Matrix{T}, 
                                            Dᶻ::AbstractMatrix{T}) where T
    ∂ᶻU₀ = Dᶻ * U₀
    ∂ᶻB₀ = Dᶻ * B₀

    return ∂ᶻU₀, ∂ᶻB₀
end

function compute_cross_derivative(∂ʸU₀::Matrix{T}, 
                                  ∂ʸB₀::Matrix{T}, 
                                  Dᶻ::AbstractMatrix{T}) where T
    ∂ʸᶻU₀ = Dᶻ * ∂ʸU₀
    ∂ʸᶻB₀ = Dᶻ * ∂ʸB₀

    return ∂ʸᶻU₀, ∂ʸᶻB₀
end

"""
    compute_second_derivatives_chebyshev(U₀, Dy, Dz)

Compute second derivatives of `U₀` using Chebyshev differentiation matrices.
Returns: ∂ʸʸU₀, ∂ᶻᶻU₀, ∂ʸᶻU₀
"""
function compute_second_derivatives_chebyshev(U₀::Matrix{T}, 
                                            B₀::Matrix{T}, 
                                            D²ᶻ::AbstractMatrix{T}) where T
    ∂ᶻᶻU₀ = D²ᶻ * U₀
    ∂ᶻᶻB₀ = D²ᶻ * B₀

    return ∂ᶻᶻU₀, ∂ᶻᶻB₀
end

"""
    compute_derivatives(B₀, U₀, Dy, Dz, gridtype::Symbol)

Automatically dispatch to Chebyshev or Fourier based on `gridtype` (:Fourier or :Chebyshev).
Returns: ∂ʸB₀, ∂ᶻB₀, ∂ʸU₀, ∂ᶻU₀, ∂ʸʸU₀, ∂ᶻᶻU₀, ∂ʸᶻU₀
"""
function compute_derivatives(U₀::Matrix{T}, 
                            B₀::Matrix{T}, 
                            y::AbstractVector{T}, 
                            Dᶻ::AbstractMatrix{T}, 
                            D²ᶻ::AbstractMatrix{T}, 
                            gridtype::Symbol) where T

    if gridtype == :Fourier
        return compute_derivatives(U₀, B₀, y, Val(:Fourier))

    elseif gridtype == :Chebyshev
        return compute_derivatives(U₀, B₀, y, Dᶻ, D²ᶻ, Val(:Chebyshev))

    else
        error("Unsupported grid type: $gridtype")
    end
end

function compute_derivatives(U₀::Matrix{T}, 
                            B₀::Matrix{T}, 
                            y::AbstractVector{T}, 
                            ::Val{:Fourier}) where T

    ∂ʸU₀,   ∂ʸB₀ = compute_first_derivatives_fourier( U₀, B₀, y)
    ∂ʸʸU₀, ∂ʸʸB₀ = compute_second_derivatives_fourier(U₀, B₀, y)

    return ∂ʸU₀, ∂ʸB₀, ∂ʸʸU₀, ∂ᶻᶻB₀ 
end

function compute_derivatives(U₀::Matrix{T}, 
                            B₀::Matrix{T}, 
                            Dᶻ::AbstractMatrix{T}, 
                            D²ᶻ::AbstractMatrix{T}, 
                            ::Val{:Chebyshev}) where T

    ∂ᶻU₀,  ∂ᶻB₀  = compute_first_derivatives_chebyshev( U₀, B₀, Dᶻ)
    ∂ᶻᶻU₀, ∂ᶻᶻB₀ = compute_second_derivatives_chebyshev(U₀, B₀, D²ᶻ)

    return ∂ᶻU₀, ∂ᶻB₀, ∂ᶻᶻU₀, ∂ᶻᶻB₀ 
end


