using SparseArrays

@inline function set_diag!(A::SparseMatrixCSC, v::AbstractVector)
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
    B₀::SparseMatrixCSC{Float64, Int}
    U₀::SparseMatrixCSC{Float64, Int}
    ∂ʸB₀::SparseMatrixCSC{Float64, Int}
    ∂ᶻB₀::SparseMatrixCSC{Float64, Int}
    ∂ʸU₀::SparseMatrixCSC{Float64, Int}
    ∂ᶻU₀::SparseMatrixCSC{Float64, Int}
    ∂ʸʸU₀::SparseMatrixCSC{Float64, Int}
    ∂ᶻᶻU₀::SparseMatrixCSC{Float64, Int}
    ∂ʸᶻU₀::SparseMatrixCSC{Float64, Int}
end

struct BasicState
    fields::BaseStateFields
end

"""
    initialize_basic_state_from_fields(B₀, U₀)

Construct a `BasicState` object with diagonals initialized from
`B₀` and `U₀`. Derivatives can be set later using `initialize_basic_state!`.
"""
function initialize_basic_state_from_fields(B₀::Matrix{Float64}, U₀::Matrix{Float64})
    function zero_diag_sparse(A::Matrix{Float64})
        spdiagm(0 => zeros(Float64, length(A)))
    end

    fields = BaseStateFields(
        spdiagm(0 => vec(B₀)), spdiagm(0 => vec(U₀)),
        zero_diag_sparse(B₀), zero_diag_sparse(B₀),
        zero_diag_sparse(U₀), zero_diag_sparse(U₀),
        zero_diag_sparse(U₀), zero_diag_sparse(U₀), zero_diag_sparse(U₀)
    )
    return BasicState(fields)
end

"""
    initialize_basic_state!(bs::BasicState, ∂ʸB₀, ∂ᶻB₀, ∂ʸU₀, ∂ᶻU₀, ∂ʸʸU₀, ∂ᶻᶻU₀, ∂ʸᶻU₀)

Assign precomputed first and second derivatives of `B₀` and `U₀`
to the diagonal matrices in `bs.fields`.
"""
function initialize_basic_state!(bs::BasicState,
                    ∂ʸB₀::Matrix{Float64}, ∂ᶻB₀::Matrix{Float64},
                    ∂ʸU₀::Matrix{Float64}, ∂ᶻU₀::Matrix{Float64},
                    ∂ʸʸU₀::Matrix{Float64}, ∂ᶻᶻU₀::Matrix{Float64}, 
                    ∂ʸᶻU₀::Matrix{Float64}
        )

    set_diag!(bs.fields.∂ʸB₀,   vec(∂ʸB₀))
    set_diag!(bs.fields.∂ᶻB₀,   vec(∂ᶻB₀))
    set_diag!(bs.fields.∂ʸU₀,   vec(∂ʸU₀))
    set_diag!(bs.fields.∂ᶻU₀,   vec(∂ᶻU₀))
    set_diag!(bs.fields.∂ʸʸU₀,  vec(∂ʸʸU₀))
    set_diag!(bs.fields.∂ᶻᶻU₀,  vec(∂ᶻᶻU₀))
    set_diag!(bs.fields.∂ʸᶻU₀,  vec(∂ʸᶻU₀))

    return nothing
end
