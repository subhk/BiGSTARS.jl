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

    if size(U₀)[1] == length(y)
        ∂ʸU₀ = gradient(U₀, y, dims=1, order=1)
        ∂ʸB₀ = gradient(B₀, y, dims=1, order=1)
    else
        ∂ʸU₀ = gradient(U₀, y, dims=2, order=1)
        ∂ʸB₀ = gradient(B₀, y, dims=2, order=1)
    end

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

    if size(U₀)[1] == length(y)
        ∂ʸU₀ = gradient(U₀, y, dims=1, order=2)
        ∂ʸB₀ = gradient(B₀, y, dims=1, order=2)
    else
        ∂ʸU₀ = gradient(U₀, y, dims=2, order=2)
        ∂ʸB₀ = gradient(B₀, y, dims=2, order=2)
    end

    return ∂ʸU₀, ∂ʸB₀
end

"""
    compute_first_derivatives_chebyshev(U₀, B₀, Dᶻ)

Compute first derivatives of `B₀` and `U₀` using Chebyshev differentiation matrices.
Returns: ∂ʸB₀, ∂ᶻB₀, ∂ʸU₀, ∂ᶻU₀
"""
function compute_first_derivatives_chebyshev(U₀::Matrix{T}, 
                                            B₀::Matrix{T}, 
                                            y::AbstractVector{T},
                                            Dᶻ::AbstractMatrix{T}) where T

    ∂ᶻU₀ = similar(U₀)
    ∂ᶻB₀ = similar(U₀)

    if size(U₀)[1] == length(y)
        for it in 1:length(y)
            ∂ᶻU₀[it,:] = Dᶻ * U₀[it,:]
            ∂ᶻB₀[it,:] = Dᶻ * B₀[it,:]
        end
    else
        for it in 1:length(y)
            ∂ᶻU₀[:,it] = Dᶻ * U₀[:,it]
            ∂ᶻB₀[:,it] = Dᶻ * B₀[:,it]
        end
    end

    return ∂ᶻU₀, ∂ᶻB₀
end

function compute_cross_derivatives(∂ʸU₀::Matrix{T}, 
                                  ∂ʸB₀::Matrix{T}, 
                                  y::AbstractVector{T},
                                  Dᶻ::AbstractMatrix{T}) where T

    ∂ʸᶻU₀ = similar(∂ʸU₀)
    ∂ʸᶻB₀ = similar(∂ʸU₀)

    if size(∂ʸU₀)[1] == length(y)
        for it in 1:length(y)
            ∂ʸᶻU₀[it,:] = Dᶻ * ∂ʸU₀[it,:]
            ∂ʸᶻB₀[it,:] = Dᶻ * ∂ʸB₀[it,:]
        end
    else
        for it in 1:length(y)
            ∂ʸᶻU₀[:,it] = Dᶻ * ∂ʸU₀[:,it]
            ∂ʸᶻB₀[:,it] = Dᶻ * ∂ʸB₀[:,it]
        end
    end

    return ∂ʸᶻU₀, ∂ʸᶻB₀
end

"""
    compute_second_derivatives_chebyshev(U₀, Dy, Dz)

Compute second derivatives of `U₀` using Chebyshev differentiation matrices.
Returns: ∂ʸʸU₀, ∂ᶻᶻU₀, ∂ʸᶻU₀
"""
function compute_second_derivatives_chebyshev(U₀::Matrix{T}, 
                                            B₀::Matrix{T}, 
                                            y::AbstractVector{T},
                                            D²ᶻ::AbstractMatrix{T}) where T

    ∂ᶻᶻU₀ = similar(U₀)
    ∂ᶻᶻB₀ = similar(U₀)

    if size(U₀)[1] == length(y)
        for it in 1:length(y)
            ∂ᶻᶻU₀[it,:] = D²ᶻ * U₀[it,:]
            ∂ᶻᶻB₀[it,:] = D²ᶻ * B₀[it,:]
        end
    else
        for it in 1:length(y)
            ∂ᶻᶻU₀[:,it] = D²ᶻ * U₀[:,it]
            ∂ᶻᶻB₀[:,it] = D²ᶻ * B₀[:,it]
        end
    end

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
        return compute_derivatives(U₀, B₀, y, Dᶻ,      Val(:Fourier) )

    elseif gridtype == :Chebyshev
        return compute_derivatives(U₀, B₀, y, Dᶻ, D²ᶻ, Val(:Chebyshev))
    
    elseif gridtype == :All
        return compute_derivatives(U₀, B₀, y, Dᶻ, D²ᶻ, Val(:All)      )

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

    ∂ʸU₀, ∂ʸB₀     = compute_first_derivatives_fourier(   U₀, B₀, y     )
    ∂ʸʸU₀, ∂ʸʸB₀   = compute_second_derivatives_fourier(  U₀, B₀, y     )
    ∂ᶻU₀, ∂ᶻB₀     = compute_first_derivatives_chebyshev( U₀, B₀, y, Dᶻ )
    ∂ᶻᶻU₀, ∂ᶻᶻB₀   = compute_second_derivatives_chebyshev(U₀, B₀, y, D²ᶻ)
    ∂ʸᶻU₀, ∂ʸᶻB₀   = compute_cross_derivatives(∂ʸU₀, ∂ʸB₀, y, Dᶻ)

    return Derivatives{T}(
        ∂ʸU₀=∂ʸU₀, ∂ʸB₀=∂ʸB₀,
        ∂ʸʸU₀=∂ʸʸU₀, ∂ʸʸB₀=∂ʸʸB₀,
        ∂ᶻU₀=∂ᶻU₀, ∂ᶻB₀=∂ᶻB₀,
        ∂ᶻᶻU₀=∂ᶻᶻU₀, ∂ᶻᶻB₀=∂ᶻᶻB₀,
        ∂ʸᶻU₀=∂ʸᶻU₀, ∂ʸᶻB₀=∂ʸᶻB₀
    )
end

function compute_derivatives(U₀::Matrix{T}, 
                            B₀::Matrix{T}, 
                            y::AbstractVector{T}, 
                            Dᶻ::AbstractMatrix{T}, 
                            ::Val{:Fourier}) where T

    ∂ʸU₀, ∂ʸB₀   = compute_first_derivatives_fourier(U₀, B₀, y)
    ∂ʸʸU₀, ∂ʸʸB₀ = compute_second_derivatives_fourier(U₀, B₀, y)
    ∂ʸᶻU₀, ∂ʸᶻB₀ = compute_cross_derivatives(∂ʸU₀, ∂ʸB₀, y, Dᶻ)

    return Derivatives{T}(
        ∂ʸU₀=∂ʸU₀, ∂ʸB₀=∂ʸB₀,
        ∂ʸʸU₀=∂ʸʸU₀, ∂ʸʸB₀=∂ʸʸB₀,
        ∂ʸᶻU₀=∂ʸᶻU₀, ∂ʸᶻB₀=∂ʸᶻB₀
    )
end

function compute_derivatives(U₀::Matrix{T}, 
                            B₀::Matrix{T}, 
                            y::AbstractVector{T},
                            Dᶻ::AbstractMatrix{T}, 
                            D²ᶻ::AbstractMatrix{T}, 
                            ::Val{:Chebyshev}) where T

    ∂ᶻU₀, ∂ᶻB₀   = compute_first_derivatives_chebyshev( U₀, B₀, y, Dᶻ)
    ∂ᶻᶻU₀, ∂ᶻᶻB₀ = compute_second_derivatives_chebyshev(U₀, B₀, y, D²ᶻ)

    return Derivatives{T}(
        ∂ᶻU₀=∂ᶻU₀, ∂ᶻB₀=∂ᶻB₀,
        ∂ᶻᶻU₀=∂ᶻᶻU₀, ∂ᶻᶻB₀=∂ᶻᶻB₀
    )
end


