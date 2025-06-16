using LinearAlgebra: mul!
using Printf

#@inline diagonal(A::AbstractMatrix, k::Integer=0) = view(A, diagind(A, k))

"""
`cheb_coord_transform transform' : transform the deravtive 
operator from a domain of ζ ∈ [-1, 1] → x ∈ [0, L] via
x = (1.0 + ζ) / 2.0 * L
Input:
    D¹: First-order Chebyshev derivative in ζ
    D²: Second-order Chebyshev derivative in ζ
    d¹: Transformed coefficient 
    d²: Transformed coefficient 
Output:
    Dₓ : First-order Chebyshev derivative in x
    Dₓₓ: Second-order Chebyshev derivative in x
"""
function cheb_coord_transform(D¹, D², d¹, d²)
    Dₓ = zeros(size(D¹, 1), size(D¹, 2))
    mul!(Dₓ, diagm(d¹), D¹)

    Dₓₓ = zeros(size(D², 1), size(D², 2))
    tmp₁ = zeros(size(D², 1), size(D², 2))
    mul!(Dₓₓ,    diagm(d²),   D¹) 
    mul!(tmp₁, diagm(d¹ .^ 2), D²)
    Dₓₓ = Dₓₓ + tmp₁ 

    return Dₓ, Dₓₓ
end


"""
`cheb_coord_transform_ho' : transform the deravtive 
operator from a domain of ζ ∈ [-1, 1] → x ∈ [0, L] via
x = (1.0 + ζ) / 2.0 * L
Input:
    D¹: First-order Chebyshev derivative in ζ
    D²: Second-order Chebyshev derivative in ζ
    D³: First-order Chebyshev derivative in ζ
    D⁴: Second-order Chebyshev derivative in ζ
    d¹: Transformed coefficient 
    d²: Transformed coefficient 
    d³: Transformed coefficient 
    d⁴: Transformed coefficient 
Output:
    Dₓₓₓ : Third-order Chebyshev derivative in x
    Dₓₓₓₓ: Fourth-order Chebyshev derivative in x
"""
function cheb_coord_transform_ho(D¹, D², D³, D⁴, d¹, d², d³, d⁴)

    Dₓₓₓ = zeros(size(D², 1), size(D², 2))
    tmp₁ = zeros(size(D², 1), size(D², 2))
    tmp₂ = zeros(size(D², 1), size(D², 2))
    mul!(Dₓₓₓ, diagm(d³), D¹)
    mul!(tmp₁, diagm(d¹), diagm(d²))
    mul!(tmp₂, diagm(d¹ .^  3), D³)
    Dₓₓₓ = Dₓₓₓ + 3tmp₁ * D² + tmp₂
    
    Dₓₓₓₓ = zeros(size(D², 1), size(D², 2))
    mul!(Dₓₓₓₓ, diagm(d⁴), D¹)
    mul!(tmp₁, diagm(d¹), diagm(d³))
    mul!(tmp₂, diagm(d¹ .^ 2), diagm(d²))
    tmp₁  = 4tmp₁ + 3.0 * diagm(d² .^ 2)
    tmp₂  = 6.0 * diagm(d¹ .^ 2) * diagm(d²)
    Dₓₓₓₓ = Dₓₓₓₓ + tmp₁ * D² + 6tmp₂ * D³ + diagm(d¹ .^  4) * D⁴

    return Dₓₓₓ, Dₓₓₓₓ 
end


#  [-1, 1] ↦ [-L, L]
function MinusLtoPlusL_transform(x, L::Float64)
    z = @. x * L
    Δ1 = @. 1.0 / L + 0.0 * z
    Δ2 = @. 0.0 * z
    return z, Δ1, Δ2
end

# [-1, 1] ↦ [-L, 0]
function MinusLtoZero_transform(x, L::Float64)
    z = @. -1.0 * (1.0 - x) /2.0 * L
    Δ1 = @. 2.0 / L + 0.0 * z
    Δ2 = @. 0.0 * z
    return z, Δ1, Δ2
end

# [-1, 1] ↦ [0, L]
function zerotoL_transform(x, L::Float64)
    z  = @. (1.0 + x) / 2.0 * L
    Δ1 = @. 2.0 / L + 0.0 * z
    Δ2 = @. 0.0 * z
    return z, Δ1, Δ2
end

# [0, 2π] → [0, L]
function transform_02π_to_0L(x, L::Float64)
    z  = @. x/2π * L
    Δ1 = @. 2π / L + 0.0 * z
    Δ2 = @. 0.0 * z
    return z, Δ1, Δ2
end

function zerotoL_transform_ho(x, L::Float64)
    z  = @. (1.0 + x) / 2.0 * L
    Δ1 = @. 2.0 / L + 0.0 * z
    Δ2 = @. 0.0 * z
    Δ3 = @. 0.0 * z
    Δ4 = @. 0.0 * z
    return z, Δ1, Δ2, Δ3, Δ4
end

# [-1, 1] ↦ [0, 1]
function zerotoone_transform(x)
    z  = @. (1.0 + x) / 2.0
    Δ1 = @. 2.0 + 0.0 * z
    Δ2 = @. 0.0 * z
    return z, Δ1, Δ2
end

function chebder_transform(x, D¹, D², fun_transform, kwargs...)
    z, d¹, d² = fun_transform(x, kwargs...)
    Dₓ, Dₓₓ = cheb_coord_transform(D¹, D², d¹, d²)
    return z, Dₓ, Dₓₓ
end

function chebder_transform_ho(x, D¹, D², D³, D⁴, fun_transform, kwargs...)
    z, d¹, d², d³, d⁴ = fun_transform(x, kwargs...)
          Dₓₓₓ, Dₓₓₓₓ = cheb_coord_transform_ho(D¹, D², D³, D⁴, d¹, d², d³, d⁴)
    return z, Dₓₓₓ, Dₓₓₓₓ
end