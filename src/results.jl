# ==============================================================================
# results.jl — eigenproblem result types + spectral post-processing shared by the
# SLEPc solve path and by reconstruction. (Was eig_solver.jl; the serial
# machinery was removed when SLEPc/PETSc became the sole eigensolver.)
# ==============================================================================

"""
    ConvergenceHistory

Tracks adaptive-shift convergence information during an eigenvalue solve.
"""
mutable struct ConvergenceHistory
    attempts::Vector{Float64}           # shift values attempted
    converged::Vector{Bool}             # whether each attempt converged
    eigenvalues::Vector{ComplexF64}     # λ₁ found at each attempt
    errors::Vector{String}              # message for failed attempts
    final_shift::Float64                # final accepted shift
    total_iterations::Int               # reserved (kept for layout compatibility)

    ConvergenceHistory() = new(Float64[], Bool[], ComplexF64[], String[], 0.0, 0)
end

"""
    SolverResults

Computed eigenpairs plus metadata for one eigenproblem.

# Fields
- `eigenvalues::Vector{ComplexF64}`
- `eigenvectors::Matrix{ComplexF64}`
- `converged::Bool`
- `method_used::Symbol`
- `final_shift::Float64`
- `iterations::Int`
- `solve_time::Float64`
- `history::ConvergenceHistory`
"""
struct SolverResults
    eigenvalues::Vector{ComplexF64}
    eigenvectors::Matrix{ComplexF64}
    converged::Bool
    method_used::Symbol
    final_shift::Float64
    iterations::Int
    solve_time::Float64
    history::ConvergenceHistory
end

"""
    sort_eigenvalues!(λ, Χ, by; rev=true, σ=nothing) -> (λ, Χ)

Return reordered copies. `:nearest` sorts by ascending `|λ - σ|` (requires `σ`),
the mode shift-and-invert targets, so `[1]` is the mode at the shift. `:R`/`:I`/`:M`
sort by real/imaginary part / magnitude.
"""
function sort_eigenvalues!(λ::Vector, Χ::Matrix, by::Symbol; rev::Bool=true,
                           σ::Union{Real,Nothing}=nothing)
    if by === :nearest
        σ === nothing && throw(ArgumentError("sortby=:nearest requires the shift σ"))
        idx = sortperm(λ, by = x -> abs(x - σ))      # ascending: nearest the shift first
    else
        sortfun = by == :R ? real : by == :I ? imag : abs
        idx = sortperm(λ, by=sortfun, rev=rev)
    end
    return λ[idx], Χ[:, idx]
end

"""
    _filter_physical_modes(λ, Χ, B; rtol=1e-6) -> (λ, Χ)

Drop infinite/spurious modes of a generalized pencil `A x = λ B x` with a singular
`B` (descriptor / augmented-derived systems): physical modes have `‖Bχ‖` of `O(1)`,
spurious ones ≈ 0. No-op for a non-singular `B`. Never returns an empty set (falls
back to the inputs), so callers always get a usable result.
"""
function _filter_physical_modes(λ::AbstractVector, Χ::AbstractMatrix, B; rtol::Float64=1e-6)
    (isempty(λ) || size(Χ, 2) == 0) && return λ, Χ
    masses = Vector{Float64}(undef, size(Χ, 2))
    @inbounds for i in 1:size(Χ, 2)
        χ = view(Χ, :, i)
        masses[i] = norm(B * χ) / max(norm(χ), eps())   # norm(χ) computed once
    end
    scale = maximum(masses)
    scale == 0.0 && return λ, Χ
    keep = findall(>(rtol * scale), masses)
    isempty(keep) && return λ, Χ
    return λ[keep], Χ[:, keep]
end

"""
    _keep_by_mass(masses; rtol=1e-6) -> Vector{Int}

Indices of physical modes from precomputed per-mode masses `‖Bχᵢ‖/‖χᵢ‖`: keep
those above `rtol · maximum(masses)` (drop singular-B infinite modes, mass ≈ 0).
The keep-rule of `_filter_physical_modes`, but on masses computed distributedly.
Keeps everything when the set is empty / all-zero / nothing would survive.
"""
function _keep_by_mass(masses::AbstractVector{<:Real}; rtol::Float64=1e-6)
    isempty(masses) && return Int[]
    scale = maximum(masses)
    scale == 0.0 && return collect(eachindex(masses))
    keep = findall(>(rtol * scale), masses)
    isempty(keep) && return collect(eachindex(masses))
    return keep
end

"""
    print_summary(r::SolverResults)

Print a compact summary of one solve result.
"""
function print_summary(r::SolverResults)
    println("Eigenvalue Solver Results")
    println("   Method: $(r.method_used)")
    println("   Converged: $(r.converged ? "✅ Yes" : "❌ No")")
    println("   Final shift: $(r.final_shift)")
    @printf("   Total time: %.3fs\n", r.solve_time)
    println("   Attempts: $(length(r.history.attempts))")
    if r.converged && !isempty(r.eigenvalues)
        for (i, λ) in enumerate(r.eigenvalues)
            @printf("   [%d] % .6f %+.6fi\n", i, real(λ), imag(λ))
        end
    end
    return nothing
end
