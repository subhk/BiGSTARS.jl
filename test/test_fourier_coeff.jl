using Test
using SparseArrays
using FFTW
using BiGSTARS: fourier_diff_operator, fourier_multiply_operator

@testset "Fourier Coefficient-Space Operators" begin

    @testset "Differentiation (diagonal)" begin
        N = 16
        L = 2π
        D = fourier_diff_operator(N, L, 1)

        @test size(D) == (N, N)
        @test D isa SparseMatrixCSC

        wavenumbers = fftfreq(N, N)
        for j in 1:N
            @test D[j, j] ≈ im * wavenumbers[j] * 2π / L
        end

        # 2nd derivative
        D2 = fourier_diff_operator(N, L, 2)
        for j in 1:N
            @test D2[j, j] ≈ (im * wavenumbers[j] * 2π / L)^2
        end
    end

    @testset "Derivative of sin(x)" begin
        N = 32
        L = 2π
        x = fourier_points(N, L)

        f_hat = fft(sin.(x)) / N
        D = fourier_diff_operator(N, L, 1)
        df_hat = D * f_hat
        df = real.(ifft(df_hat * N))

        @test maximum(abs.(df - cos.(x))) < 1e-10
    end

    @testset "Second derivative of sin(x)" begin
        N = 32
        L = 2π
        x = fourier_points(N, L)

        f_hat = fft(sin.(x)) / N
        D2 = fourier_diff_operator(N, L, 2)
        d2f_hat = D2 * f_hat
        d2f = real.(ifft(d2f_hat * N))

        # d²/dx² sin(x) = -sin(x)
        @test maximum(abs.(d2f - (-sin.(x)))) < 1e-10
    end

    @testset "Multiplication operator (circulant)" begin
        N = 32
        L = 2π
        x = fourier_points(N, L)

        f = cos.(x)
        g = sin.(x)
        fg_expected = sin.(x) .* cos.(x)  # = sin(2x)/2

        f_hat = fft(f) / N
        g_hat = fft(g) / N

        Mf = fourier_multiply_operator(f_hat, N)
        fg_hat = Mf * g_hat
        fg_computed = real.(ifft(fg_hat * N))

        @test maximum(abs.(fg_computed - fg_expected)) < 1e-10
    end

    @testset "Fourier points" begin
        N = 16
        L = 2π
        x = fourier_points(N, L)

        @test length(x) == N
        @test x[1] ≈ 0.0
        @test x[end] ≈ L * (N - 1) / N
        # Spacing should be uniform
        dx = diff(x)
        @test maximum(abs.(dx .- L / N)) < 1e-14
    end

end
