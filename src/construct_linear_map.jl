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

function construct_linear_map(A, B)
    ShiftAndInvert(factorize(A), B, Vector{eltype(A)}(undef, size(A,1))) |>
    M -> LinearMap{eltype(A)}(M, size(A,1), ismutating=true)
end