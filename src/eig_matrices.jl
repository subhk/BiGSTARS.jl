"""
    GEVPMatrices(Ablocks::NamedTuple, Bblocks::NamedTuple)

    Parametric `GEVPMatrices{TA,TB}` struct with an inner constructor:
        - Infers element types `TA<:Complex` and `TB<:Real` from the first blocks
        - Detects an n×m block grid and block size N
        - Assembles full sparse `A` and `B` via fast `kron`
        - Provides `.As` and `.Bs` NamedTuples of block-row views
"""
struct GEVPMatrices{TA<:Complex, TB<:Real}
    A  :: SparseMatrixCSC{TA,Int}
    B  :: SparseMatrixCSC{TB,Int}
    As :: NamedTuple
    Bs :: NamedTuple

    function GEVPMatrices(
        Ablocks::NamedTuple,
        Bblocks::NamedTuple
    )
        # 1) Grid shape & block size
        row_labels = collect(keys(Ablocks))
        n = length(row_labels)
        first_row = Ablocks[row_labels[1]]
        m = length(first_row)
        @assert length(Bblocks[row_labels[1]]) == m "Ablocks/Bblocks mismatch"
        N = size(first_row[1], 1)

        # 2) Infer TA, TB
        TA_inf = eltype(first_row[1]); @assert TA_inf <: Complex
        TB_inf = eltype(Bblocks[row_labels[1]][1]); @assert TB_inf <: Real

        # 3) Allocate A, B
        A = spzeros(TA_inf, n*N, m*N)
        B = spzeros(TB_inf, n*N, m*N)

        # 4) Assemble blocks
        for (i, lbl) in enumerate(row_labels)
            Ai = Ablocks[lbl]; Bi = Bblocks[lbl]
            @assert length(Ai)==m && length(Bi)==m
            for j in 1:m
                Eij = sparse([i],[j],[one(Int)], n, m)
                A  += kron(Eij, sparse(Ai[j]))
                B  += kron(Eij, sparse(Bi[j]))
            end
        end

        # 5) Block-row views
        As = (; map(i -> row_labels[i] => view(A, (i-1)*N+1:i*N, :), 1:n)... )
        Bs = (; map(i -> row_labels[i] => view(B, (i-1)*N+1:i*N, :), 1:n)... )

        # 6) Construct
        return new{TA_inf,TB_inf}(A, B, As, Bs)
    end
end

