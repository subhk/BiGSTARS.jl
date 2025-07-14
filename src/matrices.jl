"""
    GEVPMatrices(TA, TB, N; nblocks=3, labels=nothing)

Create sparse matrices `A` and `B` of size `(nblocks × N, nblocks × N)` with block-row views,
labeled using a `NamedTuple` interface for clean access.
"""
struct GEVPMatrices{TA<:Complex, TB<:Real, NA, NB}
    A  :: SparseMatrixCSC{TA, Int}
    B  :: SparseMatrixCSC{TB, Int}
    As :: NA  # NamedTuple of views into row blocks of A
    Bs :: NB  # NamedTuple of views into row blocks of B
end

function GEVPMatrices(::Type{TA}, ::Type{TB}, N::Int;
                      nblocks::Int=3,
                      labels::Union{Nothing, Vector{Symbol}}=nothing) where {TA<:Complex, TB<:Real}

    # Default block names: :A1, :A2, ...
    labels = labels === nothing ? Symbol.("A" .* string.(1:nblocks)) : labels
    @assert length(labels) == nblocks "Number of labels must match number of blocks."

    total_size = nblocks * N
    A = spzeros(TA, total_size, total_size)
    B = spzeros(TB, total_size, total_size)

    # Row-block views
    Ab = NamedTuple{Tuple(labels)}((@view A[(i-1)*N+1 : i*N, :] for i in 1:nblocks)...)
    Bb = NamedTuple{Tuple(labels)}((@view B[(i-1)*N+1 : i*N, :] for i in 1:nblocks)...)

    return GEVPMatrices{TA, TB, typeof(Ab), typeof(Bb)}(A, B, Ab, Bb)
end
