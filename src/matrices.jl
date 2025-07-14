"""
    GEVPMatrices(TA, TB, N; nblocks=3, labels=nothing)

Create square sparse matrices `A` and `B` of size `(nblocks × N, nblocks × N)`.
Provides block-row views via `NamedTuple` access: `As` and `Bs`, where each entry 
(e.g. `As.w`) is a view into a row-block of `A`.

# Arguments
- `TA`, `TB`: element types for `A` and `B`, respectively.
- `N`: size of each block row.
- `nblocks`: number of block rows (and columns).
- `labels`: optional `Vector{Symbol}` for naming each block-row. Defaults to `:A1`, `:A2`, ...

# Returns
- `GEVPMatrices` struct with `.A`, `.B`, `.As`, `.Bs` fields.
"""
struct GEVPMatrices{TA<:Complex, TB<:Real, NA, NB}
    A  :: SparseMatrixCSC{TA, Int}
    B  :: SparseMatrixCSC{TB, Int}
    As :: NA  # NamedTuple of views into block-rows of A
    Bs :: NB  # NamedTuple of views into block-rows of B
end

function GEVPMatrices(::Type{TA}, ::Type{TB}, N::Int;
                      nblocks::Int=3,
                      labels::Union{Nothing, Vector{Symbol}}=nothing) where {TA<:Complex, TB<:Real}

    labels = labels === nothing ? Symbol.("A" .* string.(1:nblocks)) : labels
    @assert length(labels) == nblocks "Number of labels must match number of blocks."

    total_size = nblocks * N
    A = spzeros(TA, total_size, total_size)
    B = spzeros(TB, total_size, total_size)

    # As = (; (label => @view A[(i-1)*N+1:i*N, :] for (i, label) in enumerate(labels))...)
    # Bs = (; (label => @view B[(i-1)*N+1:i*N, :] for (i, label) in enumerate(labels))...)

    As = (; ((label => @view A[(i-1)*N+1:i*N, :]) for (i, label) in enumerate(labels))...)
    Bs = (; ((label => @view B[(i-1)*N+1:i*N, :]) for (i, label) in enumerate(labels))...)

    return GEVPMatrices{TA, TB, typeof(As), typeof(Bs)}(A, B, As, Bs)
end
