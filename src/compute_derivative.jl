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

    ∂ᶻU₀ = similar(U₀)
    ∂ᶻB₀ = similar(U₀)

    mul!(∂ᶻU₀, Dᶻ, U₀)
    mul!(∂ᶻB₀, Dᶻ, B₀)

    return ∂ᶻU₀, ∂ᶻB₀
end

function compute_cross_derivative(∂ʸU₀::Matrix{T}, 
                                  ∂ʸB₀::Matrix{T}, 
                                  Dᶻ::AbstractMatrix{T}) where T

    ∂ʸᶻU₀ = similar(∂ʸU₀)
    ∂ʸᶻB₀ = similar(∂ʸᶻB₀)

    mul!(∂ʸᶻU₀, Dᶻ, ∂ʸU₀)
    mul!(∂ʸᶻB₀, Dᶻ, ∂ʸB₀)

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

    ∂ᶻᶻU₀ = similar(U₀)
    ∂ᶻᶻB₀ = similar(U₀)

    mul!(∂ᶻᶻU₀, D²ᶻ, U₀)
    mul!(∂ᶻᶻB₀, D²ᶻ, B₀)

    return ∂ᶻᶻU₀, ∂ᶻᶻB₀
end

"""
    Derivatives

A struct to wrap all computed derivatives for U₀ and B₀.
"""
Base.@kwdef struct Derivatives{T}
    ∂ʸU₀::Union{Matrix{T}, Nothing} = nothing
    ∂ʸB₀::Union{Matrix{T}, Nothing} = nothing
    ∂ʸʸU₀::Union{Matrix{T}, Nothing} = nothing
    ∂ʸʸB₀::Union{Matrix{T}, Nothing} = nothing
    ∂ʸᶻU₀::Union{Matrix{T}, Nothing} = nothing
    ∂ʸᶻB₀::Union{Matrix{T}, Nothing} = nothing
    ∂ᶻU₀::Union{Matrix{T}, Nothing} = nothing
    ∂ᶻB₀::Union{Matrix{T}, Nothing} = nothing
    ∂ᶻᶻU₀::Union{Matrix{T}, Nothing} = nothing
    ∂ᶻᶻB₀::Union{Matrix{T}, Nothing} = nothing
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
        return compute_derivatives(U₀, B₀, y, Dᶻ,   Val(:Fourier))

    elseif gridtype == :Chebyshev
        return compute_derivatives(U₀, B₀, Dᶻ, D²ᶻ, Val(:Chebyshev))
    
    elseif gridtype == :All
        return compute_derivatives(U₀, B₀, y_or_Dy, Dz, Dzz, Val(:All))

    else
        error("Unsupported grid type: $gridtype")
    end
end

"""
    compute_derivatives(U₀, B₀, y, Dᶻ, D²ᶻ, :Mixed)

Compute derivatives for mixed grids: Fourier in y and Chebyshev in z.
Returns a `Derivatives` struct.
"""
function compute_derivatives(U₀::Matrix{T}, 
                            B₀::Matrix{T}, 
                            y::AbstractVector{T}, 
                            Dᶻ::AbstractMatrix{T}, 
                            D²ᶻ::AbstractMatrix{T}, 
                            ::Val{:All}) where T

    ∂ʸU₀, ∂ʸB₀     = compute_first_derivatives_fourier(U₀, B₀, y)
    ∂ʸʸU₀, ∂ʸʸB₀   = compute_second_derivatives_fourier(U₀, B₀, y)
    ∂ᶻU₀, ∂ᶻB₀     = compute_first_derivatives_chebyshev(U₀, B₀, Dᶻ)
    ∂ᶻᶻU₀, ∂ᶻᶻB₀   = compute_second_derivatives_chebyshev(U₀, B₀, D²ᶻ)
    ∂ʸᶻU₀, ∂ʸᶻB₀   = compute_cross_derivatives(∂ʸU₀, ∂ʸB₀, Dᶻ)

    return Derivatives{T}(
        ∂ʸU₀=∂ʸU₀, ∂ʸB₀=∂ʸB₀,
        ∂ʸʸU₀=∂ʸʸU₀, ∂ʸʸB₀=∂ʸʸB₀,
        ∂ᶻU₀=∂ᶻU₀, ∂ᶻB₀=∂ᶻB₀,
        ∂ᶻᶻU₀=∂ᶻᶻU₀, ∂ᶻᶻB₀=∂ᶻᶻB₀,
        ∂ʸᶻU₀=∂ʸᶻU₀, ∂ʸᶻB₀=∂ʸᶻB₀
    )

function compute_derivatives(U₀::Matrix{T}, 
                            B₀::Matrix{T}, 
                            y::AbstractVector{T}, 
                            Dᶻ::AbstractMatrix{T}, 
                            ::Val{:Fourier}) where T

    ∂ʸU₀, ∂ʸB₀ = compute_first_derivatives_fourier(U₀, B₀, y)
    ∂ʸʸU₀, ∂ʸʸB₀ = compute_second_derivatives_fourier(U₀, B₀, y)
    ∂ʸᶻU₀, ∂ʸᶻB₀ = compute_cross_derivatives(∂ʸU₀, ∂ʸB₀, Dᶻ)

    return Derivatives{T}(
        ∂ʸU₀=∂ʸU₀, ∂ʸB₀=∂ʸB₀,
        ∂ʸʸU₀=∂ʸʸU₀, ∂ʸʸB₀=∂ʸʸB₀,
        ∂ʸᶻU₀=∂ʸᶻU₀, ∂ʸᶻB₀=∂ʸᶻB₀
    )
end

function compute_derivatives(U₀::Matrix{T}, 
                            B₀::Matrix{T}, 
                            Dᶻ::AbstractMatrix{T}, 
                            D²ᶻ::AbstractMatrix{T}, 
                            ::Val{:Chebyshev}) where T

    ∂ᶻU₀, ∂ᶻB₀   = compute_first_derivatives_chebyshev(U₀, B₀, Dᶻ)
    ∂ᶻᶻU₀, ∂ᶻᶻB₀ = compute_second_derivatives_chebyshev(U₀, B₀, D²ᶻ)

    return Derivatives{T}(
        ∂ᶻU₀=∂ᶻU₀, ∂ᶻB₀=∂ᶻB₀,
        ∂ᶻᶻU₀=∂ᶻᶻU₀, ∂ᶻᶻB₀=∂ᶻᶻB₀
    )
end


