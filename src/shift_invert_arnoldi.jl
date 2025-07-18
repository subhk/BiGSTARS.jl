#using LinearMaps

#using ArnoldiMethod: partialschur, partialeigen, LM, LR, LI, SR, SI

# --- Generalized Arnoldi eigensolver ---
function Eigs_Arnoldi(𝓛, ℳ;
                      σ::Float64,
                      which = :LR,
                      nev::Int=1,
                      maxiter::Int=100,
                      tol::Float64=1e-12, 
                      sortby::Symbol = :M)
    
    op = construct_linear_map(𝓛 - σ * ℳ, ℳ)

    # # Construct operator
    # op = which in (:LR, :SR) ? construct_linear_map(𝓛 - σ*ℳ, ℳ) :
    #      which in (:LM, :SM) ? construct_linear_map(𝓛, ℳ) :
    #      throw(ArgumentError("Unsupported `which`: $which"))

    decomp, history = partialschur(op;
                            nev=nev,
                            tol=tol,
                            restarts=maxiter,
                            which=which
    )

    μ, Χ = partialeigen(decomp)
    λ = @. 1.0 / μ + σ

    λ, Χ = sort_evals_(λ, Χ, sortby; rev=true)

    return λ, Χ, history
end

# --- Retry wrapper with adaptive σ and convergence checking ---
function solve_shift_invert_arnoldi(𝓛, ℳ;
                                        σ₀::Float64,
                                        which = :LM,
                                        sortby::Symbol = :M,
                                        nev::Int=1,
                                        maxiter::Int=100,
                                        n_tries::Int=8,
                                        Δσ₀::Float64=0.2,
                                        incre::Float64=1.2,
                                        ϵ::Float64=1e-5)

    Δσs_up = [Δσ₀ * incre^(i-1) * abs(σ₀) for i in 1:n_tries]
    Δσs_dn = [-δ for δ in Δσs_up]
    σ_attempts = [σ₀ + δ for δ in vcat(Δσs_up, Δσs_dn)]

    λ_prev = nothing

    for (i, σ) in enumerate(σ_attempts)
        @printf "(attempt %2d) trying σ = %f\n" i real(σ)
        try
            λ, Χ, hist = Eigs_Arnoldi(𝓛, ℳ; σ=σ, which=which, nev=nev, maxiter=maxiter)

            @printf "Converged: first λ = %f + i %f (σ = %f)\n" real(λ[1]) imag(λ[1]) σ

            if λ_prev !== nothing && abs(λ[1] - λ_prev) < ϵ
                @printf "Successive eigenvalues converged: |Δλ| = %.2e < %.2e\n" abs(λ[1] - λ_prev) ϵ
                return λ, Χ #, σ
            end

            λ_prev = λ[1]
        catch err
            @warn "Arnoldi failed at σ = $σ: $err"
        end
    end

    error("Shift-and-invert Arnoldi failed to converge near σ₀ = $σ₀ after $n_tries×2 attempts.")
end
