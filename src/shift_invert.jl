# using LazyGrids
# using BlockArrays
# using Printf
# using StaticArrays
# #using Interpolations
# using SparseArrays
# using SparseMatrixDicts
# using SpecialFunctions
# using FillArrays
# using Parameters
# using Test
# using MAT
# using BenchmarkTools

# using Serialization
# #using Pardiso
# using Arpack
# using LinearMaps
# using LinearAlgebra

# using Pardiso: PardisoSolver, pardisoinit, set_phase!, set_nprocs!, 
#     set_matrixtype!, fix_iparm!, get_matrix, pardiso, set_msglvl!, Pardiso

function construct_linear_map_pardiso(H, S, num_thread=40)
    ps = PardisoSolver()
    set_matrixtype!(ps, Pardiso.COMPLEX_NONSYM)
    pardisoinit(ps)
    fix_iparm!(ps, :N)
    H_pardiso = get_matrix(ps, H, :N)
    b = rand(Float64, size(H, 1))
    set_phase!(ps, Pardiso.ANALYSIS)
    #set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    set_nprocs!(ps, num_thread) 
    pardiso(ps, H_pardiso, b)
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, H_pardiso, b)
    return (LinearMap{eltype(H)}(
            (y, x) -> begin
                set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
                pardiso(ps, y, H_pardiso, S * x)
            end,
            size(H, 1);
            ismutating=true), ps)
end


function sort_evals_(λₛ, Χ, which, sorting="lm")
    @assert which ∈ ["M", "I", "R"]

    if sorting == "lm"
        if which == "I"
            idx = sortperm(λₛ, by=imag, rev=true) 
        end
        if which == "R"
            idx = sortperm(λₛ, by=real, rev=true) 
        end
        if which == "M"
            idx = sortperm(λₛ, by=abs, rev=true) 
        end
    else
        if which == "I"
            idx = sortperm(λₛ, by=imag, rev=false) 
        end
        if which == "R"
            idx = sortperm(λₛ, by=real, rev=false) 
        end
        if which == "M"
            idx = sortperm(λₛ, by=abs, rev=false) 
        end
    end
    return λₛ[idx], Χ[:, idx]
end

# struct ShiftAndInvert{TA,TB,TT}
#     A_lu::TA
#     B::TB
#     temp::TT
# end

# function (M::ShiftAndInvert)(y, x)
#     mul!(M.temp, M.B, x)
#     ldiv!(y, M.A_lu, M.temp)
# end

# function construct_linear_map(A, B)
#     a = ShiftAndInvert( factorize(A), B, Vector{eltype(A)}(undef, size(A,1)) )
#     LinearMap{eltype(A)}(a, size(A,1), ismutating=true)
# end

function Eigs_Arpack(𝓛, ℳ; σ::Float64, maxiter::Int, which)

    if which == :LR
        
        λₛ, Χ, info = Arpack.eigs(𝓛, ℳ, nev=1, 
                                        tol=1e-7, 
                                        maxiter=maxiter, 
                                        which, 
                                        sigma=σ,
                                        check=0)
        #λₛ = @. 1.0 / λₛ + σ

    end

    if which == :LM

        @printf "Arpack eigs: %s \n" which

        # lm, ps  = construct_linear_map_pardiso(𝓛, ℳ)
        # λₛ⁻¹, Χ = Arpack.eigs(lm, tol=0.0, maxiter=maxiter, nev=20, which=:LM)
        # # Release all internal memory for all matrices
        # set_phase!(ps, Pardiso.RELEASE_ALL) 
        # pardiso(ps)

        # # Eigenvalues have to be inverted to find 
        # # the eigenvalues of the non-inverted problem.
        # λₛ = @. 1.0 / λₛ⁻¹ #* -1.0*im

        λₛ, Χ = Arpack.eigs( #construct_linear_map(𝓛, ℳ),
                            𝓛, ℳ,
                            nev=200, 
                            tol=1e-8, 
                            maxiter=maxiter, 
                            which, 
                            check=0)

        λₛ, Χ  = sort_evals_(λₛ, Χ, which, "lm")
    end 

    return λₛ[1], Χ[:,1]
end


# function EigSolver_shift_invert_arpack_checking(𝓛, ℳ; σ₀::ComplexF64, α::Float64)
#     converged = true
#     λₛ = []
#     count::Int = -1
#     λₛ₀ = zeros(ComplexF64, 1)
#     λₛ₀[1] = σ₀
#     try 
#         push!(λₛ, λₛ₀[1])
#         while converged
#             if count > -1; push!(λₛ, λₛ₀[1]); end
#             λₛₜ = λₛ₀[1].re + α * λₛ₀[1].re
#             @printf "target eigenvalue λ: %f \n" λₛₜ
#             λₛ₀, info = Eigs(𝓛, ℳ; σ=λₛₜ, maxiter=20)
#             count += 1
#         end
#     catch error
#         λₛ = Array(λₛ)
#         if length(λₛ) > 1
#             λₛ = sort_evals_(λₛ, "R")
#         end
#         #λₛ, info = Eigs(𝓛, ℳ; σ=0.99λₛ[1].re, maxiter=20)
#         @printf "found eigenvalue (α=%0.02f): %f + im %f \n" α λₛ[1].re λₛ[1].im
#     end
#     return λₛ[1]
# end

function EigSolver_shift_invert_arpack(𝓛, ℳ; σ₀::Float64, maxiter, which)
    #maxiter::Int = 20
    try 
        σ = 1.20σ₀
        @printf "sigma: %f \n" σ.re
        λₛ, Χ = Eigs_Arpack(𝓛, ℳ; σ=σ, maxiter=maxiter, which)
        @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
        return λₛ, Χ
    catch error
        try 
            σ = 1.10σ₀
            @printf "(first didn't work) sigma: %f \n" real(σ) 
            λₛ, Χ = Eigs_Arpack(𝓛, ℳ; σ=σ, maxiter=maxiter, which)
            @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
            return λₛ, Χ
        catch error
            try 
                σ = 1.05σ₀
                @printf "(second didn't work) sigma: %f \n" real(σ) 
                λₛ, Χ = Eigs_Arpack(𝓛, ℳ; σ=σ, maxiter=maxiter, which)
                @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                return λₛ, Χ
            catch error
                try
                    σ = 0.99σ₀
                    @printf "(third didn't work) sigma: %f \n" real(σ) 
                    λₛ, Χ = Eigs_Arpack(𝓛, ℳ; σ=σ, maxiter=maxiter, which)
                    @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                    return λₛ, Χ
                catch error
                    try
                        σ = 0.95σ₀
                        @printf "(fourth didn't work) sigma: %f \n" real(σ) 
                        λₛ, Χ = Eigs_Arpack(𝓛, ℳ; σ=σ, maxiter=maxiter, which)
                        @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                        return λₛ, Χ
                    catch error
                        try
                            σ = 0.90σ₀
                            @printf "(fifth didn't work) sigma: %f \n" real(σ) 
                            λₛ, Χ = Eigs_Arpack(𝓛, ℳ; σ=σ, maxiter=maxiter, which)
                            @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                            return λₛ, Χ   
                        catch error
                            try
                                σ = 0.85σ₀
                                @printf "(sixth didn't work) sigma: %f \n" real(σ) 
                                λₛ, Χ = Eigs_Arpack(𝓛, ℳ; σ=σ, maxiter=maxiter, which)
                                @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                return λₛ, Χ  
                            catch error
                                σ = 0.80σ₀
                                @printf "(seventh didn't work) sigma: %f \n" real(σ)
                                λₛ, Χ = Eigs_Arpack(𝓛, ℳ; σ=σ, maxiter=maxiter, which)
                                @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                return λₛ, Χ
                            end
                        end            
                    end
                end
            end
        end    
    end
end

# function EigSolver_shift_invert(𝓛, ℳ; σ₀::Float64)
#     maxiter::Int = 20
#     try 
#         σ = 1.10σ₀
#         @printf "sigma: %f \n" σ
#         λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#         @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#         return λₛ #, Χ
#     catch error
#         try
#             σ = 1.05σ₀
#             @printf "(first didn't work) sigma: %f \n" real(σ) 
#             λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#             @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#             return λₛ #, Χ
#         catch error
#             try
#                 σ = 0.95σ₀
#                 @printf "(second didn't work) sigma: %f \n" real(σ)
#                 λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                 @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                 return λₛ #, Χ
#             catch error
#                 try
#                     σ = 0.90σ₀
#                     @printf "(third didn't work) sigma: %f \n" real(σ)
#                     λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                     @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                     return λₛ #, Χ
#                 catch error
#                     try
#                         σ = 0.87σ₀
#                         @printf "(fourth didn't work) sigma: %f \n" real(σ)
#                         λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                         @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                         return λₛ #, Χ 
#                     catch error
#                         try
#                             σ = 0.85σ₀
#                             @printf "(fifth didn't work) sigma: %f \n" real(σ)
#                             λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                             @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                             return λₛ #, Χ
#                         catch error
#                             try
#                                 σ = 0.80σ₀
#                                 @printf "(sixth didn't work) sigma: %f \n" real(σ)
#                                 λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                 @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                                 return λₛ #, Χ
#                             catch error
#                                 try
#                                     σ = 0.70σ₀
#                                     @printf "(seventh didn't work) sigma: %f \n" real(σ) 
#                                     λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                     @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                                     return λₛ #, Χ
#                                 catch error
#                                     try
#                                         σ = 0.65σ₀
#                                         @printf "(eighth didn't work) sigma: %f \n" real(σ) 
#                                         λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                         @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                                         return λₛ #, Χ
#                                     catch error
#                                         try
#                                             σ = 0.60σ₀
#                                             @printf "(ninth didn't work) sigma: %f \n" real(σ) 
#                                             λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                             @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                                             return λₛ #, Χ
#                                         catch error
#                                             try
#                                                 σ = 0.55σ₀
#                                                 @printf "(tenth didn't work) sigma: %f \n" real(σ) 
#                                                 λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                                 @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                                                 return λₛ #, Χ
#                                             catch error
#                                                 σ = 0.50σ₀
#                                                 @printf "(eleventh didn't work) sigma: %f \n" real(σ)
#                                                 λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                                 @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                                                 return λₛ #, Χ
#                                             end    
#                                         end   
#                                     end
#                                 end    
#                             end
#                         end                    
#                     end          
#                 end    
#             end
#         end
#     end
# end

# function EigSolver_shift_invert_2(𝓛, ℳ; σ₀::Float64)
#     maxiter::Int = 20
#     try 
#         σ = 0.90σ₀
#         @printf "sigma: %f \n" σ
#         λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#         @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#         return λₛ #, Χ
#     catch error
#         try
#             σ = 0.87σ₀
#             @printf "(first didn't work) sigma: %f \n" real(σ) 
#             λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#             @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#             return λₛ #, Χ
#         catch error
#             try
#                 σ = 0.84σ₀
#                 @printf "(second didn't work) sigma: %f \n" real(σ)
#                 λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                 @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                 return λₛ #, Χ
#             catch error
#                 try
#                     σ = 0.81σ₀
#                     @printf "(third didn't work) sigma: %f \n" real(σ)
#                     λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                     @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                     return λₛ #, Χ
#                 catch error
#                     try
#                         σ = 0.78σ₀
#                         @printf "(fourth didn't work) sigma: %f \n" real(σ)
#                         λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                         @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                         return λₛ #, Χ 
#                     catch error
#                         try
#                             σ = 0.75σ₀
#                             @printf "(fifth didn't work) sigma: %f \n" real(σ)
#                             λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                             @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                             return λₛ #, Χ
#                         catch error
#                             try
#                                 σ = 0.70σ₀
#                                 @printf "(sixth didn't work) sigma: %f \n" real(σ)
#                                 λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                 @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                                 return λₛ #, Χ
#                             catch error
#                                 try
#                                     σ = 0.65σ₀
#                                     @printf "(seventh didn't work) sigma: %f \n" real(σ) 
#                                     λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                     @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                                     return λₛ #, Χ
#                                 catch error
#                                     try
#                                         σ = 0.60σ₀
#                                         @printf "(eighth didn't work) sigma: %f \n" real(σ) 
#                                         λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                         @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                                         return λₛ #, Χ
#                                     catch error
#                                         try
#                                             σ = 0.55σ₀
#                                             @printf "(ninth didn't work) sigma: %f \n" real(σ) 
#                                             λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                             @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                                             return λₛ #, Χ
#                                         catch error
#                                             try
#                                                 σ = 0.50σ₀
#                                                 @printf "(tenth didn't work) sigma: %f \n" real(σ) 
#                                                 λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                                 @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                                                 return λₛ #, Χ
#                                             catch error
#                                                 σ = 0.45σ₀
#                                                 @printf "(eleventh didn't work) sigma: %f \n" real(σ)
#                                                 λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                                 @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
#                                                 return λₛ #, Χ
#                                             end    
#                                         end   
#                                     end
#                                 end    
#                             end
#                         end                    
#                     end          
#                 end    
#             end
#         end
#     end
# end