

"""
    sort_evals_(λ, Χ, by::Symbol; rev::Bool)

Sort eigenvalues `λ` and corresponding eigenvectors `Χ` by:
- `:R` → real part
- `:I` → imaginary part
- `:M` → magnitude (abs)

Set `rev=true` for descending (default), `false` for ascending.
"""
function sort_evals_(λ::Vector, Χ::Matrix, by::Symbol; rev::Bool=true)
    sortfun = by == :R ? real : by == :I ? imag : abs
    idx = sortperm(λ, by=sortfun, rev=rev)
    return λ[idx], Χ[:, idx]
end

"""
    Eigs_Arpack(𝓛, ℳ; σ, which=:LM, nev=1, maxiter=300, tol=1e-8, sortby=:M)

Compute eigenvalues of the generalized eigenproblem 𝓛 x = λ ℳ x
using ARPACK with shift-and-invert at shift `σ`.

- `which`: ARPACK keyword (e.g., :LM, :LR)
- `sortby`: :R, :I, or :M (real, imag, magnitude)
"""
function Eigs_Arpack(𝓛, ℳ;
                     σ::Float64,
                     which::Symbol = :LM,
                     nev::Int = 1,
                     maxiter::Int = 300,
                     tol::Float64 = 1e-8,
                     sortby::Symbol = :M)

    λ, Χ, info = Arpack.eigs(𝓛, ℳ;
                        nev=nev, 
                        sigma=σ, 
                        which=which,
                        maxiter=maxiter, 
                        tol=tol, 
                        check=0)

    λ, Χ = sort_evals_(λ, Χ, sortby; rev=true)

    return λ, Χ
end

"""
    EigSolver_shift_invert_arpack(𝓛, ℳ; σ₀, which, sortby, ...)

Robust ARPACK-based shift-and-invert eigenvalue solver with:
- Adaptive retry around σ₀
- Convergence check on successive eigenvalues
- Sorting of eigenvalues by specified criterion

Returns: (λ, Χ, σ_used)
"""
function solver_shift_invert_arpack(𝓛, ℳ;
                        σ₀::Float64,
                        which::Symbol = :LM,
                        sortby::Symbol = :M,
                        nev::Int = 1,
                        maxiter::Int = 300,
                        n_tries::Int = 8,
                        Δσ₀::Float64 = 0.1,
                        incre::Float64 = 1.1,
                        ϵ::Float64 = 1e-7
        )

    # Δσ_up = [Δσ₀ * incre^(i-1) * abs(σ₀) for i in 1:n_tries]
    # σ_list = [σ₀ + δ for δ in Δσ_up] ∪ [σ₀ - δ for δ in Δσ_up]

    Δσs_up = [ Δσ₀ * incre^(i-1) * abs(σ₀) for i in 1:n_tries ]
    Δσs_dn = [-δ for δ in Δσs_up]
    σ_list = [σ₀ + δ for δ in vcat(Δσs_up, Δσs_dn)]

    λ_prev = nothing

    for (i, σ) in enumerate(σ_list)
        @printf("(attempt %2d) trying σ = %.6f\n", i, σ)
        try
            λ, Χ = Eigs_Arpack(𝓛, ℳ; σ=σ, which=which, nev=nev, maxiter=maxiter, sortby=sortby)
            @printf(" → converged: λ = %.6f + i%.6f (σ = %.6f)\n", real(λ[1]), imag(λ[1]), σ)

            if λ_prev !== nothing && abs(λ[1] - λ_prev) < ϵ
                @printf(" ✓ converged by Δλ = %.2e < %.2e\n", abs(λ[1] - λ_prev), ϵ)
                return λ, Χ #, σ
            end

            λ_prev = λ[1]
        catch err
            @warn "ARPACK failed at σ = $σ: $(err.msg)"
        end
    end

    error("❌ ARPACK failed to converge after $(2n_tries) attempts around σ₀ = $σ₀")
end
