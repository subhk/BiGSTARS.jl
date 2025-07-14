
"""
    compute_derivatives(B₀, U₀, Dy, Dz)

Compute first and second derivatives of `B₀` and `U₀` using the provided
differentiation matrices `Dy` (∂/∂y) and `Dz` (∂/∂z).

Returns:
    ∂ʸB₀, ∂ᶻB₀, ∂ʸU₀, ∂ᶻU₀, ∂ʸʸU₀, ∂ᶻᶻU₀, ∂ʸᶻU₀
"""
function compute_derivatives(B₀::Matrix, 
                            U₀::Matrix, 
                            Dy::AbstractMatrix, 
                            Dz::AbstractMatrix)
    ∂ʸB₀  = Dy * B₀
    ∂ᶻB₀  = B₀ * Dz'
    ∂ʸU₀  = Dy * U₀
    ∂ᶻU₀  = U₀ * Dz'

    ∂ʸʸU₀ = Dy * Dy * U₀
    ∂ᶻᶻU₀ = U₀ * Dz' * Dz'
    ∂ʸᶻU₀ = Dy * U₀ * Dz'

    return ∂ʸB₀, ∂ᶻB₀, ∂ʸU₀, ∂ᶻU₀, ∂ʸʸU₀, ∂ᶻᶻU₀, ∂ʸᶻU₀
end

# end # module
