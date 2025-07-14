"""
    GEVPMatrices(TA, TB, N; nblocks=3, labels=nothing)

Create sparse matrices `A` and `B` of size `N × (nblocks × N)` with block views,
labeled using a `NamedTuple` interface for clean access.
"""
struct GEVPMatrices{TA<:Complex, TB<:Real, NA, NB}
    A  :: SparseMatrixCSC{TA, Int}
    B  :: SparseMatrixCSC{TB, Int}
    As :: NA
    Bs :: NB
end

function GEVPMatrices(::Type{TA}, ::Type{TB}, N::Int;
                  nblocks::Int=3,
                  labels::Union{Nothing, 
                  Vector{Symbol}}=nothing) where {TA<:Complex, TB<:Real}

    # Default names like :A1, :A2, ...
    labels = labels === nothing ? Symbol.("A" .* string.(1:nblocks)) : labels
    @assert length(labels) == nblocks "Number of labels must match number of blocks."

    A = spzeros(TA, N, nblocks * N)
    B = spzeros(TB, N, nblocks * N)

    Ab = (; (label => @view(A[:, (i-1)*N+1:i*N]) for (i, label) in enumerate(labels))...)
    Bb = (; (Symbol(replace(string(label), "A" => "B")) => @view(B[:, (i-1)*N+1:i*N]) for (i, label) in enumerate(labels))...)

    return GEVPMatrices{TA, TB, typeof(Ab), typeof(Bb)}(A, B, Ab, Bb)
end
