using LinearAlgebra
using Printf
using Tullio
using Dierckx #: Spline1D, derivative, evaluate
#using BasicInterpolators: BicubicInterpolator

function myfindall(condition, x)
    results = Int[]
    for i in 1:length(x)
        if condition(x[i])
            push!(results, i)
        end
    end
    return results
end

# print the eigenvalues
function print_evals(λs, n)
    @printf "%i largest eigenvalues: \n" n
    for p in n:-1:1
        if imag(λs[p]) >= 0
            @printf "%i: %1.4e+%1.4eim\n" p real(λs[p]) imag(λs[p])
        end
        if imag(λs[p]) < 0
            @printf "%i: %1.4e%1.4eim\n" p real(λs[p]) imag(λs[p])
        end
    end
end

# sort the eigenvalues
function sort_evals(λs, χ, which, sorting="lm")
    @assert which ∈ ["M", "I", "R"]

    if sorting == "lm"
        if which == "I"
            idx = sortperm(λs, by=imag, rev=true) 
        end
        if which == "R"
            idx = sortperm(λs, by=real, rev=true) 
        end
        if which == "M"
            idx = sortperm(λs, by=abs, rev=true) 
        end
    else
        if which == "I"
            idx = sortperm(λs, by=imag, rev=false) 
        end
        if which == "R"
            idx = sortperm(λs, by=real, rev=false) 
        end
        if which == "M"
            idx = sortperm(λs, by=abs, rev=false) 
        end
    end

    return λs[idx], χ[:,idx]
end


function remove_evals(λs, χ, lower, higher, which)
    
    @assert which ∈ ["M", "I", "R"]

    if which == "I" # imaginary part
        arg = findall( (lower .≤ imag(λs)) .& (imag(λs) .≤ higher) )
    end

    if which == "R" # real part
        arg = findall( (lower .≤ real(λs)) .& (real(λs) .≤ higher) )
    end
    
    if which == "M" # absolute magnitude 
        arg = findall( abs.(λs) .≤ higher )
    end
    
    χ  = χ[:,arg]
    λs = λs[arg]

    return λs, χ

    #return nothing
end

function remove_spurious(λₛ, X)
    #p = findall(x->x>=abs(item), abs.(real(λₛ)))  
    deleteat!(λₛ, 1)
    X₁ = X[:, setdiff(1:end, 1)]
    return λₛ, X₁
end

function inverse_Lap_hor(∇ₕ²)
    Qm, Rm = qr(∇ₕ²)
    invR   = inv(Rm) 
    # by sparsing the matrix speeds up matrix-matrix multiplication 
    Qm     = sparse(Qm) 
    Qᵀ     = transpose(Qm)
    H      = (invR * Qᵀ)
    return H
end

function ∇f(f, x)
    @assert ndims(f) == ndims(x)
    @tullio dx[i] := x[i+1] - x[i]
    @assert std(dx) ≤ 1.0e-6
    N = length(x); #Assume >= 3
    ∂f_∂x = similar(f);
    ∂f_∂x .= 0.0;
    Δx    = x[2]-x[1]; #assuming evenly spaced points

    c₄₊ = (-25.0/12.0, 4.0, -3.0, 4.0/3.0, -1.0/4.0);
    c₄₋ = @. -1.0 * c₄₊;
    c₈  = (1.0/280.0, -4.0/105.0, 1.0/5.0, -4.0/5.0, 0.0, 
        4.0/5.0, -1.0/5.0, 4.0/105.0, -1.0/280.0);
    for k ∈ 1:4
        ∂f_∂x[k] = c₄₊[1]*f[k] + c₄₊[2]*f[k+1] + c₄₊[3]*f[k+2] + c₄₊[4]*f[k+3] + c₄₊[5]*f[k+4];
    end
    for k ∈ 5:N-4
        ∂f_∂x[k]  = c₈[1]*f[k-4] + c₈[2]*f[k-3] + c₈[3]*f[k-2] + c₈[4]*f[k-1];
        ∂f_∂x[k] += c₈[5]*f[k];
        ∂f_∂x[k] += c₈[6]*f[k+1] + c₈[7]*f[k+2] + c₈[8]*f[k+3] + c₈[9]*f[k+4];
    end
    for k ∈ N-3:N
        ∂f_∂x[k] = c₄₋[1]*f[k] + c₄₋[2]*f[k-1] + c₄₋[3]*f[k-2] + c₄₋[4]*f[k-3] + c₄₋[5]*f[k-4];
    end
    return ∂f_∂x ./Δx
end

# function ∇f(f, x)
#     @assert x[2] - x[1] ≈ x[3] - x[2]
#     @assert ndims(f) == ndims(x)
#     N = length(x); #Assume >= 3
#     ∂f_∂x = similar(x);
#     Δx = x[2]-x[1]; #assuming evenly spaced points
#     ∂f_∂x[1] = (-3.0f[1] + 4.0f[2] - f[3]) / (2.0Δx);
#     ∂f_∂x[2] = (-3.0f[2] + 4.0f[3] - f[4]) / (2.0Δx);
#     for k ∈ 3:N-2
#         ∂f_∂x[k] = (1.0/12.0*f[k-2] - 2.0/3.0*f[k-1] + 2.0/3.0*f[k+1] - 1.0/12.0*f[k+2]) / Δx;
#     end
#     ∂f_∂x[N-1] = (3.0f[N-1] - 4.0f[N-2] + f[N-3]) / (2.0Δx);
#     ∂f_∂x[N]   = (3.0f[N]   - 4.0f[N-1] + f[N-2]) / (2.0Δx);
#     return ∂f_∂x
# end

# function gradient(f, x; dims::Int=1)
#     n   = size(f)
#     sol = similar(f)
#     if std(diff(x)) ≤ 1e-6
#         if ndims(f) == 1
#             sol = ∇f(f, x)
#         end
#         if ndims(f) == 2
#             @assert ndims(f) ≥ dims 
#             if dims==1
#                 for it ∈ 1:n[dims+1]
#                     sol[:,it] = ∇f(f[:,it], x)
#                 end
#             else
#                 for it ∈ 1:n[dims-1]
#                     sol[it,:] = ∇f(f[it,:], x)
#                 end
#             end
#         end
#         if ndims(f) == 3
#             @assert ndims(f) ≥ dims 
#             if dims==1
#                 for it ∈ 1:n[dims+1], jt ∈ 1:n[dims+2]
#                     sol[:,it,jt] = ∇f(f[:,it,jt], x)
#                 end
#             elseif dims==2
#                 for it ∈ 1:n[dims-1], jt ∈ 1:n[dims+1]
#                     sol[it,:,jt] = ∇f(f[it,:,jt], x)
#                 end
#             else
#                 for it ∈ 1:n[dims-2], jt ∈ 1:n[dims-1]
#                     sol[it,jt,:] = ∇f(f[it,jt,:], x)
#                 end    
#             end
#         end
#     else
#         @printf "grid is nonuniform \n"
#         if ndims(f) == 1
#             itp = Spline1D(x, f, bc="nearest") 
#             sol = [derivative(itp,  xᵢ) for xᵢ in x]
#         end
#         if ndims(f) == 2
#             @assert ndims(f) ≥ dims 
#             if dims==1
#                 for it ∈ 1:n[dims+1]
#                     itp = Spline1D(x, f[:,it], bc="nearest") 
#                     sol[:,it] = [derivative(itp,  xᵢ) for xᵢ in x]
#                 end
#             else
#                 for it ∈ 1:n[dims-1]
#                     itp = Spline1D(x, f[it,:], bc="nearest") 
#                     sol[it,:] = [derivative(itp,  xᵢ) for xᵢ in x]
#                 end
#             end
#         end
#         if ndims(f) == 3
#             @assert ndims(f) ≥ dims 
#             if dims==1
#                 for it ∈ 1:n[dims+1], jt ∈ 1:n[dims+2]
#                     itp = Spline1D(x, f[:,it,jt], bc="nearest") 
#                     sol[:,it,jt] = [derivative(itp,  xᵢ) for xᵢ in x]
#                 end
#             elseif dims==2
#                 for it ∈ 1:n[dims-1], jt ∈ 1:n[dims+1]
#                     itp = Spline1D(x, f[it,:,jt], bc="nearest") 
#                     sol[it,:,jt] = [derivative(itp,  xᵢ) for xᵢ in x]
#                 end
#             else
#                 for it ∈ 1:n[dims-2], jt ∈ 1:n[dims-1]
#                     itp = Spline1D(x, f[it,jt,:], bc="nearest")
#                     sol[it,jt,:] = [derivative(itp,  xᵢ) for xᵢ in x]
#                 end    
#             end
#         end
#     end
#     return sol
# end


function gradient(f, x; dims::Int=1)
    n   = size(f)
    sol = similar(f)
    x₀  = range(minimum(x), maximum(x), 3length(x)) 
    if ndims(f) == 1
        itp  = Spline1D(x, f, bc="nearest")
        f₀   = [Dierckx.evaluate(itp, xᵢ) for xᵢ in x₀]
        itp₀ = Spline1D(x₀, f₀, bc="nearest") 
        sol  = [derivative(itp, xᵢ; nu=1) for xᵢ in x]
    end
    if ndims(f) == 2
        @assert ndims(f) ≥ dims 
        if dims==1
            for it ∈ 1:n[dims+1]
                itp  = Spline1D(x, f[:,it], bc="nearest") 
                # f₀   = [Dierckx.evaluate(itp, xᵢ) for xᵢ in x₀]
                # itp₀ = Spline1D(x₀, f₀, bc="nearest") 
                sol[:,it] = [derivative(itp, xᵢ; nu=1) for xᵢ in x]
            end
        else
            for it ∈ 1:n[dims-1]
                itp  = Spline1D(x, f[it,:], bc="nearest") 
                # f₀   = [Dierckx.evaluate(itp, xᵢ) for xᵢ in x₀]
                # itp₀ = Spline1D(x₀, f₀, bc="nearest") 
                sol[it,:] = [derivative(itp, xᵢ; nu=1) for xᵢ in x]
            end
        end
    end
    if ndims(f) == 3
        @assert ndims(f) ≥ dims 
        if dims==1
            for it ∈ 1:n[dims+1], jt ∈ 1:n[dims+2]
                itp  = Spline1D(x, f[:,it,jt], bc="nearest")
                f₀   = [Dierckx.evaluate(itp, xᵢ) for xᵢ in x₀]
                itp₀ = Spline1D(x₀, f₀, bc="nearest")  
                sol[:,it,jt] = [derivative(itp, xᵢ; nu=1) for xᵢ in x]
            end
        elseif dims==2
            for it ∈ 1:n[dims-1], jt ∈ 1:n[dims+1]
                itp  = Spline1D(x, f[it,:,jt], bc="nearest")
                f₀   = [Dierckx.evaluate(itp, xᵢ) for xᵢ in x₀]
                itp₀ = Spline1D(x₀, f₀, bc="nearest")  
                sol[it,:,jt] = [derivative(itp, xᵢ; nu=1) for xᵢ in x]
            end
        else
            for it ∈ 1:n[dims-2], jt ∈ 1:n[dims-1]
                itp  = Spline1D(x, f[it,jt,:], bc="nearest")
                f₀   = [Dierckx.evaluate(itp, xᵢ) for xᵢ in x₀]
                itp₀ = Spline1D(x₀, f₀, bc="nearest") 
                sol[it,jt,:] = [derivative(itp, xᵢ; nu=1) for xᵢ in x]
            end    
        end
    end
    return sol
end


function gradient2(f, x; dims::Int=1)
    n   = size(f)
    sol = similar(f)
    x₀  = range(minimum(x), maximum(x), 3length(x)) 
    if ndims(f) == 1
        itp  = Spline1D(x, f, bc="nearest")
        f₀   = [Dierckx.evaluate(itp, xᵢ) for xᵢ in x₀]
        itp₀ = Spline1D(x₀, f₀, bc="nearest")
        sol  = [derivative(itp₀, xᵢ; nu=2) for xᵢ in x]
    end
    if ndims(f) == 2
        @assert ndims(f) ≥ dims 
        if dims==1
            for it ∈ 1:n[dims+1]
                itp  = Spline1D(x, f[:,it], bc="nearest")
                # f₀   = [Dierckx.evaluate(itp, xᵢ) for xᵢ in x₀] 
                # itp₀ = Spline1D(x₀, f₀, bc="nearest")
                sol[:,it] = [derivative(itp, xᵢ; nu=2) for xᵢ in x]
            end
        else
            for it ∈ 1:n[dims-1]
                itp  = Spline1D(x, f[it,:], bc="nearest") 
                # f₀   = [Dierckx.evaluate(itp, xᵢ) for xᵢ in x₀] 
                # itp₀ = Spline1D(x₀, f₀, bc="nearest")
                sol[it,:] = [derivative(itp, xᵢ; nu=2) for xᵢ in x]
            end
        end
    end
    if ndims(f) == 3
        @assert ndims(f) ≥ dims 
        if dims==1
            for it ∈ 1:n[dims+1], jt ∈ 1:n[dims+2]
                itp  = Spline1D(x, f[:,it,jt], bc="nearest")
                f₀   = [Dierckx.evaluate(itp, xᵢ) for xᵢ in x₀] 
                itp₀ = Spline1D(x₀, f₀, bc="nearest") 
                sol[:,it,jt] = [derivative(itp₀, xᵢ; nu=2) for xᵢ in x]
            end
        elseif dims==2
            for it ∈ 1:n[dims-1], jt ∈ 1:n[dims+1]
                itp = Spline1D(x, f[it,:,jt], bc="nearest") 
                f₀   = [Dierckx.evaluate(itp, xᵢ) for xᵢ in x₀] 
                itp₀ = Spline1D(x₀, f₀, bc="nearest") 
                sol[it,:,jt] = [derivative(itp₀, xᵢ; nu=2) for xᵢ in x]
            end
        else
            for it ∈ 1:n[dims-2], jt ∈ 1:n[dims-1]
                itp  = Spline1D(x, f[it,jt,:], bc="nearest")
                f₀   = [Dierckx.evaluate(itp, xᵢ) for xᵢ in x₀] 
                itp₀ = Spline1D(x₀, f₀, bc="nearest") 
                sol[it,jt,:] = [derivative(itp₀, xᵢ; nu=2) for xᵢ in x]
            end    
        end
    end
    return sol
end

# function Interp2D_eigenFun(yn, zn, An, y0, z0)
#     itp = BicubicInterpolator(yn, zn, transpose(An))
#     A₀ = zeros(Float64, length(y0), length(z0))
#     A₀ = [itp(yᵢ, zᵢ) for yᵢ ∈ y0, zᵢ ∈ z0]
#     return A₀
# end

# function twoDContour(r, z, u, v, filename, it)

#     uᵣ = reshape( u, (length(z), length(r)) )
#     w  = reshape( v, (length(z), length(r)) )

#     #U  = reshape( U,  (length(z), length(r)) )
#     #B  = reshape( B,  (length(z), length(r)) )

#     r_interp = collect(LinRange(minimum(r), maximum(r), 5000))
#     z_interp = collect(LinRange(minimum(z), maximum(z), 500) )

#     #U_interp = Interp2D_eigenFun(r, z, U, r_interp, z_interp)
#     #B_interp = Interp2D_eigenFun(r, z, B, r_interp, z_interp)

#     fig = Figure(fontsize=30, size=(1800, 500), )

#     ax1 = Axis(fig[1, 1], xlabel=L"$y$", xlabelsize=30, ylabel=L"$z$", ylabelsize=30)

#     interp_  = Interp2D_eigenFun(r, z, uᵣ, r_interp, z_interp)
#     max_val = maximum(abs.(interp_))
#     levels = range(-0.7max_val, 0.7max_val, length=16)
#     co = contourf!(r_interp, z_interp, interp_, colormap=cgrad(:RdBu, rev=false),
#         levels=levels, extendlow = :auto, extendhigh = :auto )

#     # levels = range(minimum(U), maximum(U), length=8)
#     # contour!(r_interp, z_interp, U_interp, levels=levels, linestyle=:dash, color=:black, linewidth=2) 

#     # contour!(rn, zn, AmS, levels=levels₋, linestyle=:dash,  color=:black, linewidth=2) 
#     # contour!(rn, zn, AmS, levels=levels₊, linestyle=:solid, color=:black, linewidth=2) 

#     tightlimits!(ax1)
#     cbar = Colorbar(fig[1, 2], co)
#     xlims!(minimum(r), maximum(r))
#     ylims!(minimum(z), maximum(z))

#     ax2 = Axis(fig[1, 3], xlabel=L"$y$", xlabelsize=30, ylabel=L"$z$", ylabelsize=30)

#     interp_ = Interp2D_eigenFun(r, z, w, r_interp, z_interp)
#     max_val = maximum(abs.(interp_))
#     levels = range(-0.7max_val, 0.7max_val, length=16)
#     co = contourf!(r_interp, z_interp, interp_, colormap=cgrad(:RdBu, rev=false),
#         levels=levels, extendlow = :auto, extendhigh = :auto )

#     # levels = range(minimum(U), maximum(U), length=8)
#     # contour!(r_interp, z_interp, U_interp, levels=levels, linestyle=:dash, color=:black, linewidth=2) 
        
#     # contour!(rn, zn, AmS, levels=levels₋, linestyle=:dash,  color=:black, linewidth=2) 
#     # contour!(rn, zn, AmS, levels=levels₊, linestyle=:solid, color=:black, linewidth=2) 

#     tightlimits!(ax2)
#     cbar = Colorbar(fig[1, 4], co)
#     xlims!(minimum(r), maximum(r))
#     ylims!(minimum(z), maximum(z))

#     # ax1.title = L"$\mathfrak{R}(\hat{u}_r)$"
#     # ax2.title = L"$\mathfrak{R}(\hat{w})$"

#     fig
#     filename = filename * "_" * string(it) * ".png"
#     save(filename, fig, px_per_unit=4)
# end
