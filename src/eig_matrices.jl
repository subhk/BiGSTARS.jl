# """
#     GEVPMatrices(Ablocks::NamedTuple, Bblocks::NamedTuple)

#     Parametric `GEVPMatrices{TA,TB}` struct with an inner constructor:
#         - Infers element types `TA<:Number` and `TB<:Number` from the first blocks
#         - Detects an n×m block grid and block size N
#         - Assembles full sparse `A` and `B` via fast `kron`
#         - Provides `.As` and `.Bs` NamedTuples of block-row views
# """
# struct GEVPMatrices{TA<:Number, TB<:Number}
#     A  :: SparseMatrixCSC{TA,Int}
#     B  :: SparseMatrixCSC{TB,Int}
#     As :: NamedTuple
#     Bs :: NamedTuple

#     function GEVPMatrices(
#             Ablocks::NamedTuple,
#             Bblocks::NamedTuple
#     )
#         # 1) Grid shape & block size
#         # now first_row is guaranteed to be a tuple
#         row_labels = collect(keys(Ablocks))
#         n = length(row_labels)
#         first_row = Ablocks[row_labels[1]]

#         m = length(first_row)
#         @assert length(Bblocks[row_labels[1]]) == m "Ablocks/Bblocks mismatch"
#         N = size(first_row[1], 1)

#         # 2) Infer TA, TB
#         TA_inf = eltype(first_row[1]); 
#         TB_inf = eltype(Bblocks[row_labels[1]][1]); 

#         # 3) Allocate A, B
#         A = spzeros(TA_inf, n*N, m*N)
#         B = spzeros(TB_inf, n*N, m*N)

#         # 4) Assemble blocks
#         for (i, lbl) in enumerate(row_labels)
#             Ai = Ablocks[lbl]; Bi = Bblocks[lbl]
#             @assert length(Ai)==m && length(Bi)==m
#             for j in 1:m
#                 Eij = sparse([i],[j],[one(Int)], n, m)
#                 A  += kron(Eij, sparse(Ai[j]))
#                 B  += kron(Eij, sparse(Bi[j]))
#             end
#         end

#         # 5) Block-row views
#         As = (; map(i -> row_labels[i] => view(A, (i-1)*N+1:i*N, :), 1:n)... )
#         Bs = (; map(i -> row_labels[i] => view(B, (i-1)*N+1:i*N, :), 1:n)... )

#         # 6) Construct
#         return new{TA_inf,TB_inf}(A, B, As, Bs)
#     end
# end



"""
    GEVPMatrices(Ablocks::NamedTuple, Bblocks::NamedTuple)

    Parametric `GEVPMatrices{TA,TB}` struct with an inner constructor:
        - Infers element types `TA<:Number` and `TB<:Number` from the first blocks
        - Detects an n×m block grid and block size N
        - Assembles full sparse `A` and `B` via fast `kron`
        - Provides `.As` and `.Bs` NamedTuples of block-row views
        - Automatically wraps single matrices into 1-tuples
"""
struct GEVPMatrices{TA<:Number, TB<:Number}
    A  :: SparseMatrixCSC{TA,Int}
    B  :: SparseMatrixCSC{TB,Int}
    As :: NamedTuple
    Bs :: NamedTuple

    function GEVPMatrices(
        Ablocks::NamedTuple,
        Bblocks::NamedTuple
    )
                # 1) Normalize each row-value to be a tuple of matrices
        row_labels = collect(keys(Ablocks))
        # start with empty NamedTuples
        Ablocks2 = (;)
        Bblocks2 = (;)
        for lbl in row_labels
            Ai = Ablocks[lbl]
            Bi = Bblocks[lbl]
            Ai_tup = isa(Ai, AbstractMatrix) ? (Ai,) : Tuple(Ai)
            Bi_tup = isa(Bi, AbstractMatrix) ? (Bi,) : Tuple(Bi)
            Ablocks2 = merge(Ablocks2, (lbl => Ai_tup,))
            Bblocks2 = merge(Bblocks2, (lbl => Bi_tup,))
        end
        Ablocks = Ablocks2; Bblocks = Bblocks2

        # 2) Grid shape & block size
        row_labels = collect(keys(Ablocks))  # update labels
        n = length(row_labels)
        first_row = Ablocks[row_labels[1]]
        m = length(first_row)
        @assert length(Bblocks[row_labels[1]]) == m "Ablocks/Bblocks mismatch"
        N = size(first_row[1], 1)

        # 3) Infer TA, TB
        TA_inf = eltype(first_row[1])
        TB_inf = eltype(Bblocks[row_labels[1]][1])

        # 4) Allocate A, B
        A = spzeros(TA_inf, n*N, m*N)
        B = spzeros(TB_inf, n*N, m*N)

        # 5) Assemble blocks via sparse-kron
        for (i, lbl) in enumerate(row_labels)
            Ai_row = Ablocks[lbl]; Bi_row = Bblocks[lbl]
            for j in 1:m
                Eij = sparse([i], [j], [one(Int)], n, m)
                blockA = Ai_row[j]
                blockB = Bi_row[j]
                A_block = blockA isa SparseMatrixCSC ? blockA : sparse(blockA)
                B_block = blockB isa SparseMatrixCSC ? blockB : sparse(blockB)
                A += kron(Eij, A_block)
                B += kron(Eij, B_block)
            end
        end

        # 6) Block-row views
        row_labels = collect(keys(Ablocks))  # ensure correct order
        As = (; map(i -> row_labels[i] => view(A, (i-1)*N+1:i*N, :), 1:n)...)
        Bs = (; map(i -> row_labels[i] => view(B, (i-1)*N+1:i*N, :), 1:n)...)

        # 7) Construct instance
        return new{TA_inf,TB_inf}(A, B, As, Bs)
    end
end
