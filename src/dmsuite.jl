using ToeplitzMatrices
using LinearAlgebra
using Printf
using FFTW

#@inline diagonal(A::AbstractMatrix, k::Integer=0) = view(A, diagind(A, k))

# function toeplitz(x::AbstractVector{T}) where T
#     n = length(x)
#     A = zeros(T, n, n)
#     for i = 1:n
#         for j = 1:n-i+1
#             A[i,i+j-1] = x[j]
#         end
#         for j = n-i+2:n
#             A[i, j-(n-i+1)] = x[j]
#         end
#     end
#     return A
# end


function cheb(N)
    @assert N > 0
    x = @. cos(π*(0:N)/N)' / 2 + 0.5; 
    c = @. [2; ones(N-1, 1); 2] * (-1)^(0:N)';
    X = repeat(x, N+1, 1)';
    dX = @. X - X';                  
    
    #D = (c .* (1.0 ./ c)') ./ (dX .+ Eye{Float64}(N+1));      # off-diagonal entries

    D = (c .* (1.0 ./ c)') ./ (dX .+ sparse(Matrix(1.0I, N+1, N+1))); 

    L  = similar(D); fill!(L, 0.0); 
    L[diagind(L)] = dropdims(sum(D, dims=2), dims=2);
    D  = @. D - L;                                              # diagonal entries
    #reverse!(x)
    #reverse!(D)
    # x = transpose(x);
    return x[1,:], D
end

"""
Calculate differentiation matrices using Chebyshev collocation.
Returns the differentiation matrices D1, D2, .. Dmder corresponding to the
mder-th derivative of the function f, at the ncheb Chebyshev nodes in the
interval [-1,1].
Parameters
----------
ncheb : int, polynomial order. ncheb + 1 collocation points
mder   : int
      maximum order of the derivative, 0 < mder <= ncheb - 1
Returns
-------
x  : ndarray
     (ncheb + 1) x 1 array of Chebyshev points
DM : ndarray
     mder x ncheb x ncheb  array of differentiation matrices
Notes
-----
This function returns  mder differentiation matrices corresponding to the
1st, 2nd, ... mder-th derivates on a Chebyshev grid of ncheb points. The
matrices are constructed by differentiating ncheb-th order Chebyshev
interpolants.
The mder-th derivative of the grid function f is obtained by the matrix-
vector multiplication
.. math::
f^{(m)}_i = D^{(m)}_{ij}f_j
The code implements two strategies for enhanced accuracy suggested by
W. Don and S. Solomonoff :
(a) the use of trigonometric  identities to avoid the computation of
differences x(k)-x(j)
(b) the use of the "flipping trick"  which is necessary since sin t can
be computed to high relative precision when t is small whereas sin (pi-t)
cannot.
It may, in fact, be slightly better not to implement the strategies
(a) and (b). Please consult [3] for details.
This function is based on code by Nikola Mirkov
http://code.google.com/p/another-chebpy
References
----------
..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.
..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix
Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519
..[3] R. Baltensperger and M. R. Trummer, Spectral Differencing With A
Twist, SIAM Journal on Scientific Computing 24, (2002) : 1465-1487
Examples
--------
The derivatives of functions is obtained by multiplying the vector of
function values by the differentiation matrix.
"""
function chebdif(ncheb, mder)
    if mder >= ncheb + 1
        throw("number of nodes must be greater than mder")
    end

    if mder <= 0
        throw("derivative order must be at least 1")
    end
    
    ℒ  = Matrix(I, ncheb, ncheb);       # Logical identity matrix.

    # indices used for flipping trick
    nn1 = Int32(floor(ncheb/2));
    nn2 = Int32(ceil(ncheb/2));
    k = 0:1:ncheb-1 #|> collect

    # compute theta vector
    θ = k * π / (ncheb-1);

    # Compute the Chebyshev points
    x = sin.(π * (ncheb-1 .- 2 * range(ncheb-1, 0, length=ncheb)) / (2 * (ncheb-1)));

    #x = @. sin(π*[ncheb-1:-2:1-cheb]'/(2*(ncheb-1)));
    #reverse!(x);

    # Assemble the differentiation matrices
    T = repeat(θ/2, 1, ncheb);
    # trigonometric identity
    Dₓ = 2 * sin.(T' + T) .* sin.(T' - T);
    
    # flipping trick
    Dₓ[nn1+1:end, 1:end] = -reverse(reverse(Dₓ[1:nn2, 1:end], dims=2), dims=1);
        
    # diagonals of Dₓ
    Dₓ[ℒ] .= ones(ncheb);

    # matrix with entries c(k)/c(j)
    #C = toeplitz( (-1.0) .^ k);
    C = Toeplitz( (-1.0) .^ k, (-1.0) .^ k );
    C = Array(C)
    C[1, :] = C[1, :] .* 2.0; C[end, :] = C[end, :] .* 2.0;
    C[:, 1] = C[:, 1] .* 0.5; C[:, end] = C[:, end] .* 0.5;

    # Z contains entries 1/(x(k)-x(j))
    Z = 1.0 ./ Dₓ;
    # with zeros on the diagonal.
    Z[ℒ] .= zeros(ncheb);

    # initialize differentiation matrices.
    D = Matrix(1.0I, ncheb, ncheb);

    Dᵐ = zeros(Float64, mder, ncheb, ncheb);

    for ell = 1:mder
        # off-diagonals
        D = ell * Z .* (C .* repeat(diag(D), 1, ncheb) .- D); # Off-diagonals
        # negative sum trick
        D[ℒ] .= -sum(D, dims=2);                              # Correct main diagonal of D
        # store current D in Dᵐ
        Dᵐ[ell, 1:end, 1:end] = D;                            # Store current D in DM
    end

    # return only one differntiation matrix
    D = Dᵐ[mder, 1:end, 1:end];
    
    # mirror x [1,-1] to x [-1,1] (for convenience)
    #reverse!(x);
    
    reverse!(D);
    return x, D
end

"""
    Fourier spectral differentiation.
    Spectral differentiation matrix on a grid with nfou equispaced points in [0, 2π)
    INPUT
    -----
    nfou: Size of differentiation matrix.
    mder: Derivative required (non-negative integer)
    OUTPUT
    -------
    xxt: Equispaced points 0, 2pi/nfou, 4pi/nfou, ... , (nfou-1)2pi/nfou
    ddm: mder'th order differentiation matrix
    Explicit formulas are used to compute the matrices for m=1 and 2.
    A discrete Fouier approach is employed for m>2. The program
    computes the first column and first row and then uses the
    toeplitz command to create the matrix.
    For mder=1 and 2 the code implements a "flipping trick" to
    improve accuracy suggested by W. Don and A. Solomonoff in
    SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
    The flipping trick is necesary since sin t can be computed to high
    relative precision when t is small whereas sin (pi-t) cannot.
    S.C. Reddy, J.A.C. Weideman 1998.  Corrected for MATLAB R13
    by JACW, April 2003.
"""
function FourierDiff(nfou, mder)
    # grid points
    range_ = 0:1:nfou-1 |> collect
    x₀ = 2π * range_ / nfou

    # grid spacing
    dx₀ = 2π/nfou

    # indices used for flipping trick
    nn1 = Int32(floor( (nfou-1)/2 ));
    nn2 = Int32(ceil(  (nfou-1)/2 ));

    if mder == 0
        # compute first column of zeroth derivative matrix, which is identity
        col1 = zeros(nfou)
        col1[1] = 1
        row1 = deepcopy(col1)

    elseif mder == 1
        # compute first column of 1st derivative matrix
        col0 = 0.5 * [(-1)^k for k ∈ 1:nfou-1]
        if nfou % 2 == 0
            topc = @. 1.0 / tan( (1:nn2) * 0.5dx₀ )
            col2 = col0 .* vcat( topc, -reverse(topc[1:nn1]) )
            col1 = vcat( 0, col2 )
        else
            topc = @. 1.0 / sin( (1:nn2) * 0.5dx₀ )
            col1 = vcat( 0, col0 .* vcat( topc, reverse(topc[1:nn1]) ) )
        end
        # first row
        row1 = @. -1.0 * col1

    elseif mder == 2
        # compute first column of 1st derivative matrix
        col0 = -0.5 * [(-1)^k for k ∈ 1:nfou-1]   
        if nfou % 2 == 0 # corresponds to even number of grid points
            topc = @. 1.0 / sin( (1:nn2) * 0.5dx₀ )^2
            col2 = col0 .* vcat( topc, reverse(topc[1:nn1]) )
            col1 = vcat( -π^2 / 3.0 / dx₀^2 - 1.0/6.0, col2 )
        else  # corresponds to odd number of grid points
            topc = @. ( 1.0/ tan((1:nn2) * 0.5dx₀)/ sin((1:nn2) * 0.5dx₀) )
            col2 = col0 .* vcat( topc, -reverse(topc[1:nn1]) )
            col1 = vcat( -π^2 / 3 / dx₀^2 + 1/12, col2 ) 
        end
        # first row
        row1 = 1.0 .* col1
    else
        # employ FFT to compute 1st column of matrix for mder > 2
        nfo1  = floor((nfou-1)/2);
        nfo2  = @. -nfou/2 * (rem(mder,2)==0) * ones(rem(nfou,2)==0)
        mwave = 1.0im .* vcat(0:1:nfo1, nfo2, -nfo1:-1)
        col1  = real( ifft( 
                    mwave.^mder .* fft( vcat(1, zeros(nfou-1)) ) 
                    ) 
                )
        if mder % 2 == 0
            row1 = 1.0 .* col1
        else
            col1 = vcat(0, col1[2:nfou])
            row1 = -1.0 .* col1
        end
    end

    𝒟 = Toeplitz(col1, row1)
    return x₀, 𝒟
end


# """
# compute first derivative using 4th order centred 
# difference scheme.z is evenly spaced.
# """
# function ddz(z; order_accuracy::Int)
#     Δz = z[2] - z[1]
#     N  = length(z)
#     D  = zeros(Float64, N, N)
#     if order_accuracy == 4
#         for k=3:N-2
#             D[k,k-2] = 1/12;
#             D[k,k-1] = -2/3;
#             D[k,k+1] = 2/3;
#             D[k,k+2] = -1/12;
#         end
#         D[1,1] = -3/2; D[1,2] = 2; D[1,3] = -1/2;
#         D[2,2] = -3/2; D[2,3] = 2; D[2,4] = -1/2;
        
#         D[N,N]     = 3/2; D[N,N-1]   = -2;  D[N,N-2]   = 1/2;
#         D[N-1,N-1] = 3/2; D[N-1,N-2] = -2;  D[N-1,N-3] = 1/2;
#         return D./Δz
#     elseif order_accuracy == 2
#         for k=2:N-1
#             D[k,k-1] = -1.0;
#             D[k,k+1] = 1.0;
#         end
#         D[1,1] = -3.0; D[1,2]  = 4.0;  D[1,3]   = -1.0;
#         D[N,N] = 3.0; D[N,N-1] = -4.0; D[N,N-2] = 1.0
#         return D./2Δz
#     else
#         error("Invalid order of accuracy")
#     end
# end

# """
# compute second derivative using 4th order centred 
# difference scheme. z is evenly spaced.
# """
# function ddz2(z; order_accuracy::Int)
#     Δz = z[2] - z[1]
#     N  = length(z)
#     D  = zeros(Float64, N,N);
#     if order_accuracy == 4
#         for k=3:N-2
#             D[k,k-2] = -1/12;
#             D[k,k-1] = 4/3;
#             D[k,k]   = -5/2;
#             D[k,k+1] = 4/3;
#             D[k,k+2] = -1/12;
#         end 
#         D[1,1] = 2.0; D[1,2] = -5.0; D[1,3] = 4.0; D[1,4] = -1.0;
#         D[2,2] = 2.0; D[2,3] = -5.0; D[2,4] = 4.0; D[2,5] = -1.0;

#         D[N,N]     = 2.0;  D[N,N-1]   = -5.0;  D[N,N-2]   = 4.0; D[N,N-3]   = -1.0;
#         D[N-1,N-1] = 2.0;  D[N-1,N-2] = -5.0;  D[N-1,N-3] = 4.0; D[N-1,N-4] = -1.0;
#         return D./Δz^2 
#     elseif order_accuracy == 2
#         for k=2:N-1
#             D[k,k-1] = 1.0;
#             D[k,k]   = -2.0; 
#             D[k,k+1] = 1.0;
#         end
#         D[1,1] = 2.0; D[1,2]   = -5.0; D[1,3]   = 4.0; D[1,4]   = -1.0;
#         D[N,N] = 2.0; D[N,N-1] = -5.0; D[N,N-2] = 4.0; D[N,N-3] = -1.0;
#         return D./Δz^2 
#     else
#         error("Invalid order of accuracy")
#     end
# end

# """
# compute fourth derivative using 4th order centred 
# difference scheme. z is evenly spaced.
# """
# function ddz4(z; order_accuracy::Int)
#     Δz = z[2] - z[1];
#     N  = length(z);
#     D  = zeros(Float64, N,N);
#     if order_accuracy == 4
#         for k=4:N-3
#             D[k,k-3] = -1/6;
#             D[k,k-2] = 2;
#             D[k,k-1] = -13/2;
#             D[k,k]   = 28/3;
#             D[k,k+1] = -13/2; 
#             D[k,k+2] = 2;
#             D[k,k+3] = -1/6;
#         end 
#         D[1,1] = 3; D[1,2] = -14;  D[1,3] = 26;  D[1,4] = -24;  D[1,5] = 11;  D[1,6] = -2;
#         D[2,2] = 3; D[2,3] = -14;  D[2,4] = 26;  D[2,5] = -24;  D[2,6] = 11;  D[2,7] = -2;
#         D[3,3] = 3; D[3,4] = -14;  D[3,5] = 26;  D[3,6] = -24;  D[3,7] = 11;  D[3,8] = -2;
        
#         D[N,N]     = 3; D[N,N-1]   = -14; D[N,N-2]   = 26; D[N,N-3]   = -24; D[N,N-4]   = 11; D[N,N-5]   = -2;
#         D[N-1,N-1] = 3; D[N-1,N-2] = -14; D[N-1,N-3] = 26; D[N-1,N-4] = -24; D[N-1,N-5] = 11; D[N-1,N-6] = -2;
#         D[N-2,N-2] = 3; D[N-2,N-3] = -14; D[N-2,N-4] = 26; D[N-2,N-5] = -24; D[N-2,N-6] = 11; D[N-2,N-7] = -2;
#         return D./Δz^4
#     elseif order_accuracy == 2
#         for k=3:N-2
#             D[k,k-2] = 1.0;
#             D[k,k-1] = -4.0;
#             D[k,k]   = 6.0;
#             D[k,k+1] = -4.0; 
#             D[k,k+2] = 1.0;
#         end 
#         D[1,1] = 3.0;  D[1,2] = -14.0; D[1,3] = 26.0; D[1,4] = -24.0; D[1,5] = 11.0; D[1,6] = -2.0;
#         D[2,2] = 3.0;  D[2,3] = -14.0; D[2,4] = 26.0; D[2,5] = -24.0; D[2,6] = 11.0; D[2,7] = -2.0;

#         D[N,N]   = 3.0;  D[N,N-1] = -14.0; D[N,N-2] = 26.0; D[N,N-3] = -24.0;  
#         D[N,N-4] = 11.0; D[N,N-5] = -2.0;

#         D[N-1,N-1] = 3.0;  D[N-1,N-2] = -14.0; D[N-1,N-3] = 26.0; D[N-1,N-4] = -24.0;  
#         D[N-1,N-5] = 11.0; D[N-1,N-6] = -2.0;
#         return D./Δz^4
#     else
#         error("Invalid order of accuracy")
#     end
# end

"""
Periodic differentiation matrix constructed using
finite difference method (FDM) with 2nd order accurate
"""
function FourierDiff_fdm(Nz, order_derivate::Int)
    # grid points
    range_ = 0:1:Nz-1
    z = collect(2π * range_/Nz)
    # grid spacing
    del = 2π/Nz
    N = length(z)
    D = zeros(N, N)
    if order_derivate == 1
        for k in 2:N-1
            D[k,k-1] = -0.5;
            D[k,k+1] = 0.5;
        end
        D[1,2]   = 0.5;  D[1, N] = -0.5;
        D[N,N-1] = -0.5; D[N,1]  = 0.5;
        return z, D./del
    elseif order_derivate == 2
        for k in 2:N-1
            D[k,k-1] = 1;
            D[k,k]   = -2;
            D[k,k+1] = 1;
        end
        D[1,1] = -2; D[1,2]   = 1; D[1,N] = 1;
        D[N,N] = -2; D[N,N-1] = 1; D[N,1] = 1;
        return z, D./del^2
    elseif order_derivate == 4
        for k in 3:N-2
            D[k,k-2] = 1;  D[k,k-1] = -4;
            D[k,k]   = 6;
            D[k,k+1] = -4; D[k,k+2] = 1;
        end
        D[1,1] = 6;  D[1,2] = -4; D[1,3] = 1;  D[1,N-1] = 1; D[1,N] = -4;
        D[2,1] = -4; D[2,2] = 6;  D[2,3] = -4; D[2,4]   = 1; D[2,N] = 1;
    
        D[N-1,N] = -4; D[N-1,N-1] = 6;  D[N-1,N-2] = -4; D[N-1,N-3] = 1; D[N-1,1] = 1;
        D[N,N]   = 6;  D[N,N-1]   = -4; D[N,N-2]   = 1;  D[N,2]     = 1; D[N,1] = -4;
        return z, D./del^4
    
    else
        error("Invalid order of derivate")
    end
end


# function FourierDiff_fdm_4th(Nz, order_derivate::Int)
#     # grid points
#     range_ = 0:1:Nz-1
#     z = 2π * range_/Nz |> collect
#     # grid spacing
#     del = 2π/Nz
#     N = length(z)
#     D = zeros(N, N)
#     if order_derivate == 1
#         for k in 3:N-2
#             D[k,k-2] = 1/12; D[k,k-1] = -2/3;
#             D[k,k+0] = 0.0;
#             D[k,k+1] = 2/3;  D[k,k+2] = -1/12;
#         end
#         D[1,2] = 2/3;  D[1,3]  = -1/12; D[1,N] = -2/3;  D[1,N-1] = 1/12;    
#         D[2,1] = -2/3; D[2,3]  = 2/3;   D[2,4] = -1/12; D[2,N]   = 1/12;

#         D[N,N-1]   = -2/3; D[N,N-2]   = 1/12; D[N,1]   = 2/3; D[N,2]   = -1/12;
#         D[N-1,N-2] = -2/3; D[N-1,N-3] = 1/12; D[N-1,N] = 2/3; D[N-1,1] = -1/12;
#         return z, D./del
#     elseif order_derivate == 2
#         for k in 3:N-2
#             D[k,k-2] = -1/12; D[k,k-1] = 4/3;
#             D[k,k+0] = -2.5;
#             D[k,k+1] = 4/3;  D[k,k+2] = -1/12;
#         end
#         D[1,1] = -2.5;  D[1,2] = 4/3;  D[1,3] = -1/12; D[1,N] = 4/3;   D[1,N-1] = -1/12;
#         D[2,1] = 4/3;   D[2,2] = -2.5; D[2,3] = 4/3;   D[2,4] = -1/12; D[2,N] = -1/12; 
#         D[3,1] = -1/12; D[3,2] = 4/3;  D[3,3] = -2.5;  D[3,4] = 4/3;   D[3,5] = -1/12;

#         D[N,N]   = -2.5;  D[N,N-1]   = 4/3;  D[N,N-2]   = -1/12; D[N,1]     = 4/3;   D[N,2]   = -1/12;
#         D[N-1,N] = 4/3;   D[N-1,N-1] = -2.5; D[N-1,N-2] = 4/3;   D[N-1,N-3] = -1/12; D[N-1,1] = -1/12;
#         D[N-2,N] = -1/12; D[N-2,N-1] = 4/3;  D[N-2,N-2] = -2.5;  D[N-2,N-3] = 4/3;   D[N-2,N-4] = -1/12;
#         return z, D./del^2
#     elseif order_derivate == 4
#         for k in 4:N-3
#             D[k,k-3] = -1/6;  D[k,k-2] = 2; D[k,k-1] = -13/2;
#             D[k,k+0] = 28/3;
#             D[k,k+1] = -13/2; D[k,k+2] = 2; D[k,k+3] = -1/6;
#         end
#         D[1,1] = 28/3;  D[1,2] = -13/2; D[1,3] = 2;     D[1,4] = -1/6;  D[1,N] = -13/2; D[1,N-1] = 2;  D[1,N-2] = -1/6;
#         D[2,1] = -13/2; D[2,2] = 28/3;  D[2,3] = -13/2; D[2,4] = 2;     D[2,5] = -1/6;  D[2,N]   = 2;  D[2,N-1] = -1/6;
#         D[3,1] = 2;     D[3,2] = -13/2; D[3,3] = 28/3;  D[3,4] = -13/2; D[3,5] = 2;     D[3,6] = -1/6; D[3,N]   = -1/6;
#         D[4,1] = -1/6;  D[4,2] = 2;     D[4,3] = -13/2; D[4,4] = 28/3;  D[4,5] = -13/2; D[4,6] = 2;    D[4,7]   = -1/6;    
    
#         D[N,N]   = 28/3;  D[N,N-1] = -13/2; D[N,N-2] = 2;  D[N,N-3]   = -1/6; 
#         D[N,1]   = -13/2; D[N,2]   = 2;     D[N,3]   = -1/6;
        
#         D[N-1,N]   = -13/2; D[N-1,N-1] = 28/3; D[N-1,N-2] = -13/2; D[N-1,N-3] = 2;    
#         D[N-1,N-4] = -1/6;  D[N-1,1]   = 2;    D[N-1,2]   = -1/6; 

#         D[N-2,N]   = 2;  D[N-2,N-1] = -13/2; D[N-2,N-2] = 28/3; D[N-2,N-3] = -13/2;    
#         D[N-2,N-4] = 2;  D[N-2,N-5] = -1/6;  D[N-2,1]   = -1/6; 

#         D[N-3,N]   = -1/6;  D[N-3,N-1] = 2; D[N-3,N-2] = -13/2; D[N-3,N-3] = 28/3;    
#         D[N-3,N-4] = -13/2; D[N-3,N-5] = 2; D[N-3,1]   = -1/6; 

#         return z, D./del^4
#     else
#         error("Invalid order of derivate")
#     end
# end