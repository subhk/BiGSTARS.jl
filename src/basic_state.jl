
@inline function set_diag!(A::AbstractMatrix, v::AbstractVector)
    A[diagind(A)] .= v
    return nothing
end

"""
    struct BasicState

A container holding background fields and their first/second derivatives
projected onto the computational grid.
    
    Fields:
      B₀, U₀              -- buoyancy (temperature) and velocity
      ∂ʸB₀, ∂ᶻB₀          -- first derivatives of B₀
      ∂ʸU₀, ∂ᶻU₀          -- first derivatives of U₀
      ∂ʸʸU₀, ∂ᶻᶻU₀, ∂ʸᶻU₀ -- second derivatives of U₀
"""
BaseStateFields = @NamedTuple begin
    B₀::Matrix{Float64}
    U₀::Matrix{Float64}
    ∂ʸB₀::Matrix{Float64}
    ∂ᶻB₀::Matrix{Float64}
    ∂ʸU₀::Matrix{Float64}
    ∂ᶻU₀::Matrix{Float64}
    ∂ʸʸU₀::Matrix{Float64}
    ∂ᶻᶻU₀::Matrix{Float64}
    ∂ʸᶻU₀::Matrix{Float64}
end

struct BasicState
    fields::BaseStateFields
end

"""
    initialize_basic_state_from_fields(B₀, U₀)

Construct a `BasicState` object with diagonals initialized from
`B₀` and `U₀`. Derivatives can be set later using `initialize_basic_state!`.
"""
function initialize_basic_state_from_fields(B₀, U₀)
    zero_diag = Diagonal(zeros(size(B₀)...))
    fields = BaseStateFields(
        Diagonal(zeros(size(B₀)...)),
        Diagonal(zeros(size(U₀)...)),
        zero_diag, zero_diag,
        zero_diag, zero_diag,
        zero_diag, zero_diag, zero_diag
    )
    bs = BasicState(fields)
    set_diag!(bs.fields.B₀, vec(B₀))
    set_diag!(bs.fields.U₀, vec(U₀))
    return bs
end

"""
    initialize_basic_state!(mf::BasicState, ∂ʸB₀, ∂ᶻB₀, ∂ʸU₀, ∂ᶻU₀, ∂ʸʸU₀, ∂ᶻᶻU₀, ∂ʸᶻU₀)

Assign precomputed first and second derivatives of `B₀` and `U₀`
to the diagonal matrices in `mf.fields`.
"""
function initialize_basic_state!(mf::BasicState, ∂ʸB₀, ∂ᶻB₀, ∂ʸU₀, ∂ᶻU₀, ∂ʸʸU₀, ∂ᶻᶻU₀, ∂ʸᶻU₀)
    set_diag!(mf.fields.∇ʸB₀,   vec(∂ʸB₀))
    set_diag!(mf.fields.∇ᶻB₀,   vec(∂ᶻB₀))
    set_diag!(mf.fields.∇ʸU₀,   vec(∂ʸU₀))
    set_diag!(mf.fields.∇ᶻU₀,   vec(∂ᶻU₀))
    set_diag!(mf.fields.∇ʸʸU₀,  vec(∂ʸʸU₀))
    set_diag!(mf.fields.∇ᶻᶻU₀,  vec(∂ᶻᶻU₀))
    set_diag!(mf.fields.∇ʸᶻU₀,  vec(∂ʸᶻU₀))
    return nothing
end