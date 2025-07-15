"""
    GEVPMatrices(TA, TB, blocksA, blocksB; labels=nothing)

    Low-level constructor: builds full sparse A and B from vectors of
    blocks via fast sparse×sparse kron on the diagonal.
"""
struct GEVPMatrices{TA<:Complex, TB<:Real}
    A  :: SparseMatrixCSC{TA,Int}
    B  :: SparseMatrixCSC{TB,Int}
    As :: NamedTuple
    Bs :: NamedTuple
end

"""
    GEVPMatrices(TA, TB, Ablocks::NamedTuple, Bblocks::NamedTuple)

    High-level constructor that accepts `NamedTuple`s of 3-tuples of
    small N×N blocks, keyed by labels (e.g. `:w`, `:ζ`, `:b`).
    It flattens and delegates to the fast bulk constructor.
"""
function GEVPMatrices(::Type{TA}, ::Type{TB},
                      Ablocks::NamedTuple,
                      Bblocks::NamedTuple;
                      labels::Union{Nothing,Vector{Symbol}}=nothing
                     ) where {TA<:Complex, TB<:Real, L}

    # # Preserve insertion order of labels
    # labels = collect(keys(Ablocks))

    # use provided labels or default to the NamedTuple keys
    labels = labels === nothing ? collect(keys(Ablocks)) : labels
    @assert length(labels) == length(keys(Ablocks))

    # Flatten each 3-tuple in order
    blocksA = [ blk for l in labels for blk in Ablocks[l] ]
    blocksB = [ blk for l in labels for blk in Bblocks[l] ]

    # Delegate to main constructor
    return GEVPMatrices(TA, TB, blocksA, blocksB; labels=labels)
end


function GEVPMatrices(::Type{TA}, ::Type{TB},
                      blocksA::Vector{<:AbstractMatrix{TA}},
                      blocksB::Vector{<:AbstractMatrix{TB}};
                      labels::Union{Nothing,Vector{Symbol}}=nothing
                     ) where {TA<:Complex, TB<:Real}

    nblocks = length(blocksA)
    @assert length(blocksB) == nblocks
    N = size(blocksA[1],1)
    labels = labels === nothing ? Symbol.([string(l) for l in 1:nblocks]) : labels
    @assert length(labels)==nblocks

    total = nblocks * N
    A = spzeros(TA, total, total)
    B = spzeros(TB, total, total)

    # Slot blocks onto diagonal via sparse kron
    for i in 1:nblocks
        Ei = sparse([i],[i],[one(Int)], nblocks, nblocks)
        A += kron(Ei, sparse(blocksA[i]))
        B += kron(Ei, sparse(blocksB[i]))
    end

    # # Build NamedTuple views
    As = (; map(i -> labels[i] => view(A, (i-1)*N+1:i*N, :), 1:nblocks)...)
    Bs = (; map(i -> labels[i] => view(B, (i-1)*N+1:i*N, :), 1:nblocks)...)

    # build the NamedTuples of block-row views
    # As = (; (labels[i] => view(A, (i-1)*N+1:i*N, :)) for i in 1:nblocks )
    # Bs = (; (labels[i] => view(B, (i-1)*N+1:i*N, :)) for i in 1:nblocks )

    return GEVPMatrices{TA,TB}(A, B, As, Bs)
end

