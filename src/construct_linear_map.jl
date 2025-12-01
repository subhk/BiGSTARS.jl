# --- Shift-and-invert operator ---
struct ShiftAndInvert{TA,TB,TT}
    A_lu::TA
    B::TB
    temp::TT
end

function (M::ShiftAndInvert)(y, x)
    mul!(M.temp, M.B, x)
    ldiv!(y, M.A_lu, M.temp)
end

# Default construction (kept for compatibility)
function construct_linear_map(A, B)
    temp = Vector{eltype(A)}(undef, size(A, 1))
    construct_linear_map(factorize(A), B, temp)
end

# New overload that reuses an existing factorization and workspace vector.
function construct_linear_map(A_fact, B, temp::AbstractVector)
    ShiftAndInvert(A_fact, B, temp) |>
    M -> LinearMap{eltype(temp)}(M, length(temp), ismutating=true)
end
