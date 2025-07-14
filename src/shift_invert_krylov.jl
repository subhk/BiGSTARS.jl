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
stack(itr) = reduce(hcat, itr)
#end

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
    a = ShiftAndInvert( factorize(A), B, Vector{eltype(A)}(undef, size(A,1)) )
    LinearMap{eltype(A)}(a, size(A,1), ismutating=true)
end

function nearestval_idx(a, x)
    idx::Int = 0
    for it in eachindex(a)
        if a[it] == x?
            idx = it
        end
    end
    return idx
end


function Eigs_Krylov(
                𝓛, ℳ;
                σ::Float64,
                which::Symbol = :LR,
                maxiter::Int = 100,
                krylovdim::Int = 300
    )

    # Construct operator
    op = which in (:LR, :SR) ? construct_linear_map(𝓛 - σ*ℳ, ℳ) :
         which in (:LM, :SM) ? construct_linear_map(𝓛, ℳ) :
         throw(ArgumentError("Unsupported `which`: $which"))

    λinv, Χ, info = eigsolve(op, 
                            rand(ComplexF64, size(𝓛,1)), 
                            1, 
                            which;
                            maxiter=maxiter, 
                            krylovdim=krylovdim, 
                            verbosity=0)

    λ = which == :LR ? @. 1.0 / λinv + σ : @. 1.0 / λinv

    return λ, stack(unwrapvec, Χ)
end


function solve_shift_invert_krylov(
                    𝓛, ℳ;
                    σ₀::Float64,
                    which::Symbol = :LR,
                    maxiter::Int = 100,
                    krylovdim::Int = 300,
                    n_tries::Int = 8,
                    Δσ₀::Float64 = 0.02,
                    decay::Float64 = 0.8,
                    ϵ::Float64 = 1e-4,
    )

    Δσs_up = [ Δσ₀ * decay^(i-1) * abs(σ₀) for i in 1:n_tries ]
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
            @warn "Failure at σ = $σ: $(err.msg)"
        end
    end

    error("KrylovKit failed to converge with successive eigenvalue tolerance ϵ = $ϵ.")
end

