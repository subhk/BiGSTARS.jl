using Test
using SparseArrays
using LinearAlgebra
using BiGSTARS: conversion_operator, differentiation_operator, multiplication_operator,
    chebyshev_points, chebyshev_coefficients

@testset "Ultraspherical Operators" begin

    #──────────────────────────────────────────────────────────────────────────#
    #  Test 1: S_0 structure (Chebyshev T -> C^(1) conversion)                 #
    #──────────────────────────────────────────────────────────────────────────#
    @testset "S_0 structure" begin
        N = 8
        S0 = conversion_operator(0, N)

        # Must be sparse
        @test S0 isa SparseMatrixCSC

        # (1,1) = 1
        @test S0[1, 1] == 1.0

        # Diagonal entries: S_0[n,n] = 1/2 for n >= 2
        for n in 2:N
            @test S0[n, n] == 0.5
        end

        # Super-diagonal offset +2: S_0[n-2,n] = -1/2 for n >= 3
        for n in 3:N
            @test S0[n - 2, n] == -0.5
        end

        # All other entries should be zero
        for i in 1:N, j in 1:N
            if (i == 1 && j == 1)
                continue
            elseif (i == j && i >= 2)
                continue
            elseif (i == j - 2 && j >= 3)
                continue
            else
                @test S0[i, j] == 0.0
            end
        end

        # Check sparsity: should have exactly 1 + (N-1) + (N-2) = 2N-2 nonzeros
        @test nnz(S0) == 2 * N - 2
    end

    #──────────────────────────────────────────────────────────────────────────#
    #  Test 2: S_p for p >= 1 structure                                        #
    #──────────────────────────────────────────────────────────────────────────#
    @testset "S_p structure (p >= 1)" begin
        N = 8

        for p in [1, 2, 3]
            Sp = conversion_operator(p, N)

            @test Sp isa SparseMatrixCSC
            @test Sp[1, 1] == 1.0

            for n in 2:N
                expected = p / (n - 1 + p)
                @test Sp[n, n] ≈ expected atol = 1e-14
            end

            for n in 3:N
                expected = -p / (n - 1 + p)
                @test Sp[n - 2, n] ≈ expected atol = 1e-14
            end

            # Should have same sparsity pattern as S_0
            @test nnz(Sp) == 2 * N - 2
        end
    end

    #──────────────────────────────────────────────────────────────────────────#
    #  Test 3: D_0 structure (differentiation T -> C^(1))                      #
    #──────────────────────────────────────────────────────────────────────────#
    @testset "D_0 structure" begin
        N = 8
        D0 = differentiation_operator(0, N)

        @test D0 isa SparseMatrixCSC

        # D_0[n,n+1] = n for all n >= 1 (from d/dx T_n = n*C^(1)_{n-1})
        for n in 1:(N - 1)
            @test D0[n, n + 1] == Float64(n)
        end

        # Last row should be all zeros
        for j in 1:N
            @test D0[N, j] == 0.0
        end

        # Verify with T_3(x) = 4x^3 - 3x: derivative = 12x^2 - 3 = 3*U_2 = 3*C^(1)_2
        # D_0[3,4] = 3, so D_0 * e_4 = 3*e_3
        c_T3 = zeros(N)
        c_T3[4] = 1.0  # T_3
        dc = D0 * c_T3
        @test dc[3] == 3.0
        @test all(dc[i] == 0.0 for i in 1:N if i != 3)
    end

    #──────────────────────────────────────────────────────────────────────────#
    #  Test 4: D_p for p >= 1 structure                                        #
    #──────────────────────────────────────────────────────────────────────────#
    @testset "D_p structure (p >= 1)" begin
        N = 8

        for p in [1, 2, 3]
            Dp = differentiation_operator(p, N)

            @test Dp isa SparseMatrixCSC

            # D_p[n,n+1] = 2p for all n >= 1
            for n in 1:(N - 1)
                @test Dp[n, n + 1] == 2.0 * p
            end

            # Last row is all zeros
            for j in 1:N
                @test Dp[N, j] == 0.0
            end

            # Exactly N-1 nonzero entries (superdiagonal)
            @test nnz(Dp) == N - 1
        end
    end

    #──────────────────────────────────────────────────────────────────────────#
    #  Test 5: Multiplication operator M_f with f(x) = x                       #
    #──────────────────────────────────────────────────────────────────────────#
    @testset "Multiplication operator (f(x) = x)" begin
        N = 8
        # f(x) = x has Chebyshev coefficients [0, 1] since x = T_1(x)
        f_coeffs = [0.0, 1.0]
        Mx = multiplication_operator(f_coeffs, N)

        @test Mx isa SparseMatrixCSC

        # x * T_0 = T_1:  column 1 should give e_2 (1-indexed)
        e1 = zeros(N); e1[1] = 1.0
        result = Mx * e1
        expected = zeros(N); expected[2] = 1.0
        @test result ≈ expected atol = 1e-14

        # x * T_1 = (T_0 + T_2)/2:  column 2 should give 0.5*e_1 + 0.5*e_3
        e2 = zeros(N); e2[2] = 1.0
        result = Mx * e2
        expected = zeros(N); expected[1] = 0.5; expected[3] = 0.5
        @test result ≈ expected atol = 1e-14

        # x * T_2 = (T_1 + T_3)/2:  column 3 should give 0.5*e_2 + 0.5*e_4
        e3 = zeros(N); e3[3] = 1.0
        result = Mx * e3
        expected = zeros(N); expected[2] = 0.5; expected[4] = 0.5
        @test result ≈ expected atol = 1e-14

        # Structure: x*T_0 = T_1 makes M[2,1]=1 (special: two T_1 terms coincide)
        # All other sub/super-diagonal entries are 0.5
        @test Mx[2, 1] ≈ 1.0 atol = 1e-14   # special: x*T_0 = T_1
        @test Mx[1, 2] ≈ 0.5 atol = 1e-14   # from x*T_1 = (T_0+T_2)/2
        for n in 2:(N - 1)
            @test Mx[n, n + 1] ≈ 0.5 atol = 1e-14
        end
        for n in 3:N
            @test Mx[n, n - 1] ≈ 0.5 atol = 1e-14
        end
    end

    #──────────────────────────────────────────────────────────────────────────#
    #  Test 6: Domain scaling (derivative of x^2 on [0, 1])                    #
    #──────────────────────────────────────────────────────────────────────────#
    @testset "Domain scaling: derivative of x^2 on [0,1]" begin
        N = 16

        # Get Chebyshev points on [0, 1]
        x = chebyshev_points(N, 0.0, 1.0)

        # f(x) = x^2, values at Chebyshev points
        f_vals = x .^ 2

        # Get Chebyshev coefficients of f on [0, 1]
        # We need to work on [-1, 1] internally. Map x in [0,1] to t in [-1,1]:
        # t = 2x - 1, so x = (t+1)/2, x^2 = (t+1)^2/4 = (t^2+2t+1)/4
        # = (1/4)*(T_0 + T_2)/2 + (2/4)*T_1 + (1/4)*T_0  ... let's just compute.

        # Get reference points on [-1, 1]
        t = chebyshev_points(N)
        # f(x(t)) = ((t+1)/2)^2 = (t+1)^2/4
        f_mapped = ((t .+ 1.0) ./ 2.0) .^ 2
        coeffs = chebyshev_coefficients(f_mapped)

        # Differentiate in coefficient space:
        # D_0 gives d/dt in C^(1) basis, but d/dx = (dt/dx) * d/dt = 2 * d/dt
        # since t = 2x - 1 => dt/dx = 2
        D0 = differentiation_operator(0, N)
        S0 = conversion_operator(0, N)

        # d/dt coefficients in C^(1) basis
        dcoeffs_dt = D0 * coeffs

        # Convert to T basis for comparison (solve S0 * c_T = c_C1)
        # Or just verify numerically: evaluate derivative at Chebyshev points
        # and compare with exact derivative 2x = t + 1.

        # For the exact check: d(x^2)/dx = 2x. At each Chebyshev point in [0,1]:
        exact_deriv = 2.0 .* x

        # Numerical derivative: use the collocation approach for comparison.
        # Build the second-kind collocation matrix for C^(1) at the CGL points.
        # Actually, let's verify using the spectral coefficients directly.
        #
        # d/dx f = (2/(b-a)) * d/dt f, where (b-a) = 1, so scaling = 2.
        # The D_0 operator gives d/dt coefficients. We need to scale by 2.
        #
        # d/dx coefficients in C^(1) basis = 2 * D_0 * coeffs_T
        dcoeffs_dx = 2.0 .* dcoeffs_dt

        # Exact: d(x^2)/dx = 2x = t + 1 = T_0(t) + T_1(t)
        # In C^(1) basis (Legendre = U polynomials scaled):
        # C^(1)_0 = 1, C^(1)_1 = 2t
        # t + 1 = 1 + t = C^(1)_0 + (1/2)*C^(1)_1
        # So expected C^(1) coefficients: [1, 0.5, 0, 0, ...]
        @test abs(dcoeffs_dx[1] - 1.0) < 1e-10
        @test abs(dcoeffs_dx[2] - 0.5) < 1e-10
        for k in 3:N
            @test abs(dcoeffs_dx[k]) < 1e-10
        end
    end

    #──────────────────────────────────────────────────────────────────────────#
    #  Test 7: End-to-end eigenvalue test                                      #
    #          -u'' = lambda * u, u(-1) = u(1) = 0                             #
    #          Exact eigenvalues: lambda_k = (k*pi/2)^2, k = 1, 2, 3, ...     #
    #──────────────────────────────────────────────────────────────────────────#
    @testset "End-to-end eigenvalue: -u'' = lambda*u" begin
        N = 32

        # Build operators
        D0 = differentiation_operator(0, N)
        D1 = differentiation_operator(1, N)
        S0 = conversion_operator(0, N)
        S1 = conversion_operator(1, N)

        # Second derivative operator: D1 * D0 maps T -> C^(2)
        # -u'' = lambda * u becomes: -(D1*D0)*c = lambda * (S1*S0)*c
        # where c are the Chebyshev T coefficients.
        A = -D1 * D0          # -d^2/dx^2 in C^(2) basis
        B = S1 * S0           # identity converted to C^(2) basis

        # Convert to dense for boundary bordering
        A_full = Matrix(A)
        B_full = Matrix(B)

        # Replace last 2 rows with boundary conditions (boundary bordering)
        # u(-1) = sum_{n=0}^{N-1} c_n * T_n(-1) = sum c_n * (-1)^n = 0
        # u(1)  = sum_{n=0}^{N-1} c_n * T_n(1)  = sum c_n * 1      = 0

        # Row N-1: u(-1) = 0
        for j in 1:N
            A_full[N - 1, j] = (-1.0)^(j - 1)
            B_full[N - 1, j] = 0.0
        end

        # Row N: u(1) = 0
        for j in 1:N
            A_full[N, j] = 1.0
            B_full[N, j] = 0.0
        end

        # Solve generalized eigenvalue problem
        evals = eigvals(A_full, B_full)

        # Filter for finite, positive, real eigenvalues
        real_evals = Float64[]
        for ev in evals
            if isfinite(ev) && abs(imag(ev)) < 1e-6 && real(ev) > 0.1
                push!(real_evals, real(ev))
            end
        end
        sort!(real_evals)

        # First eigenvalue should be (pi/2)^2 ≈ 2.467...
        lambda1_exact = (pi / 2)^2
        @test length(real_evals) >= 1
        @test abs(real_evals[1] - lambda1_exact) / lambda1_exact < 1e-10

        # Second eigenvalue: (2*pi/2)^2 = pi^2
        if length(real_evals) >= 2
            lambda2_exact = pi^2
            @test abs(real_evals[2] - lambda2_exact) / lambda2_exact < 1e-8
        end

        # Third eigenvalue: (3*pi/2)^2
        if length(real_evals) >= 3
            lambda3_exact = (3 * pi / 2)^2
            @test abs(real_evals[3] - lambda3_exact) / lambda3_exact < 1e-6
        end
    end

    #──────────────────────────────────────────────────────────────────────────#
    #  Bonus: chebyshev_points and chebyshev_coefficients consistency           #
    #──────────────────────────────────────────────────────────────────────────#
    @testset "Chebyshev points and coefficients" begin
        N = 16

        # Points on [-1, 1]
        x = chebyshev_points(N)
        @test length(x) == N
        @test x[1] ≈ 1.0 atol = 1e-14   # cos(0) = 1
        @test x[N] ≈ -1.0 atol = 1e-14  # cos(pi) = -1

        # T_3(x) = 4x^3 - 3x -> coefficients should be [0, 0, 0, 1, 0, ...]
        f_vals = 4.0 .* x .^ 3 .- 3.0 .* x
        coeffs = chebyshev_coefficients(f_vals)

        @test abs(coeffs[4] - 1.0) < 1e-12  # T_3 coefficient
        for k in 1:N
            if k != 4
                @test abs(coeffs[k]) < 1e-12
            end
        end

        # Test with a polynomial: 3*T_0 - 2*T_2 + T_5
        # T_0 = 1, T_2 = 2x^2-1, T_5 = 16x^5 - 20x^3 + 5x
        f_vals = 3.0 .* ones(N) .+ (-2.0) .* (2.0 .* x .^ 2 .- 1.0) .+
                 1.0 .* (16.0 .* x .^ 5 .- 20.0 .* x .^ 3 .+ 5.0 .* x)
        coeffs = chebyshev_coefficients(f_vals)

        @test abs(coeffs[1] - 3.0) < 1e-11
        @test abs(coeffs[2]) < 1e-11
        @test abs(coeffs[3] - (-2.0)) < 1e-11
        @test abs(coeffs[4]) < 1e-11
        @test abs(coeffs[5]) < 1e-11
        @test abs(coeffs[6] - 1.0) < 1e-11
        for k in 7:N
            @test abs(coeffs[k]) < 1e-11
        end
    end

end  # @testset "Ultraspherical Operators"
