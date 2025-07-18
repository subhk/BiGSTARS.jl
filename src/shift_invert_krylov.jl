# using Serialization
# #using Pardiso
# using Arpack
# using LinearMaps
# using KrylovKit

# Wrappers
# --------
using VectorInterface: MinimalSVec, MinimalMVec, MinimalVec
# dispatch on val is necessary for type stability

function wrapvec(v, ::Val{mode}) where {mode}
    return mode === :vector ? v :
           mode === :inplace ? MinimalMVec(v) :
           mode === :outplace ? MinimalSVec(v) :
           mode === :mixed ? MinimalSVec(v) :
           throw(ArgumentError("invalid mode ($mode)"))
end

unwrapvec(v::MinimalVec) = v.vec
unwrapvec(v) = v

#if VERSION < v"1.9"
stack(f, itr) = mapreduce(f, hcat, itr)
stack(itr)    = reduce(hcat, itr)
#end


function nearestval_idx(a, x)
    idx::Int = 0
    for it in eachindex(a)
        if a[it] == x
            idx = it
        end
    end
    return idx
end


function Eigs_Krylov(
                    𝓛, ℳ;
                    σ::Float64,
                    which::Symbol = :LR,
                    maxiter::Int = 200,
                    krylovdim::Int = 100,
                    sortby::Symbol = :M)

    op = construct_linear_map(𝓛 - σ * ℳ, ℳ)

    # # Construct operator
    # op = which in (:LR, :SR) ? construct_linear_map(𝓛 - σ*ℳ, ℳ) :
    #      which in (:LM, :SM) ? construct_linear_map(𝓛, ℳ) :
    #      throw(ArgumentError("Unsupported `which`: $which"))

    λinv, Χ, info = eigsolve(op, 
                            rand(ComplexF64, size(𝓛,1)), 
                            1, 
                            which;
                            maxiter=maxiter, 
                            krylovdim=krylovdim, 
                            verbosity=0)

    if which == :LR
        λ = @. 1.0 / λinv + σ
    else
        λ = @. 1.0 / λinv
    end

    #return λ, stack(unwrapvec, Χ)

    return sort_evals_(λ, stack(unwrapvec, Χ), sortby; rev=true)

end


function solve_shift_invert_krylov(
                    𝓛, ℳ;
                    σ₀::Float64,
                    which::Symbol = :LR,
                    sortby::Symbol = :M,
                    maxiter::Int = 200,
                    krylovdim::Int = 100,
                    n_tries::Int = 8,
                    Δσ₀::Float64 = 0.2,
                    incre::Float64 = 1.2,
                    ϵ::Float64 = 1e-5,)

    Δσs_up = [ Δσ₀ * incre^(i-1) * abs(σ₀) for i in 1:n_tries ]
    Δσs_dn = [-δ for δ in Δσs_up]
    σ_attempts = [σ₀ + δ for δ in vcat(Δσs_up, Δσs_dn)]

    λ_prev = nothing

    for (i, σ) in enumerate(σ_attempts)
        @printf "(attempt %2d) trying σ = %f\n" i real(σ)
        try
            λ, Χ = Eigs_Krylov(𝓛, ℳ; 
                                σ=σ, 
                                which=which, 
                                maxiter=maxiter, 
                                krylovdim=krylovdim)
            
            if isempty(λ) || any(isnan, λ)
                @warn "Empty or NaN eigenvalue for σ = $σ"
                continue
            end

            @printf "KrylovKit converged: λ = %f + i %f (σ = %f)\n" real(λ[1]) imag(λ[1]) real(σ)

            if λ_prev !== nothing && abs(λ[1] - λ_prev) < ϵ
                @printf "Successive eigenvalues converged: |Δλ| = %.2e < %.2e\n" abs(λ[1] - λ_prev) ϵ
                return λ, Χ
            end

            λ_prev = λ[1]
        catch err
            @warn "Failure at σ = $σ: $err"
        end
    end

    error("KrylovKit failed to converge with successive eigenvalue tolerance ϵ = $ϵ.")
end

