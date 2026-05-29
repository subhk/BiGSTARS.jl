using Test

# This test file is meant to be run via the full module (runtests.jl)
# where BiGSTARS is loaded. It tests the end-to-end DSL workflow.

@testset "End-to-End Integration" begin

    @testset "1D eigenvalue: -u'' = sigma*u" begin
        domain = Domain(z = Chebyshev(N=32, lower=-1.0, upper=1.0))
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)

        @equation prob sigma * u == -dz(dz(u))
        @bc prob left(u) == 0
        @bc prob right(u) == 0

        cache = discretize(prob)
        A, B = assemble(cache, 0.0)

        # Direct solve for verification
        lambdas = eigvals(Matrix(A), Matrix(B))
        real_pos = sort(filter(l -> isfinite(l) && abs(imag(l)) < 1e-6 && real(l) > 0.1, lambdas) .|> real)

        @test length(real_pos) >= 1
        @test abs(real_pos[1] - (π / 2)^2) < 0.01
    end

    @testset "Wavenumber sweep: sigma*u = -dx(dx(u)) - dz(dz(u))" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=32, lower=-1.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)

        @equation prob sigma * u == -dx(dx(u)) - dz(dz(u))
        @bc prob left(u) == 0
        @bc prob right(u) == 0

        cache = discretize(prob)

        # For each k, smallest eigenvalue should be (pi/2)^2 + k^2
        for k in [0.0, 1.0, 2.0]
            A, B = assemble(cache, k)
            lambdas = eigvals(Matrix(A), Matrix(B))
            real_pos = sort(filter(l -> isfinite(l) && abs(imag(l)) < 1e-6 && real(l) > 0.1, lambdas) .|> real)

            expected = (π / 2)^2 + k^2
            @test length(real_pos) >= 1
            @test abs(real_pos[1] - expected) / expected < 0.01
        end
    end

    @testset "Substitution in equation" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=16, lower=-1.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)

        @substitution prob D2(A) = dz(dz(A))
        @equation prob sigma * u == -D2(u)
        @bc prob left(u) == 0
        @bc prob right(u) == 0

        cache = discretize(prob)
        A, B = assemble(cache, 0.0)

        lambdas = eigvals(Matrix(A), Matrix(B))
        real_pos = sort(filter(l -> isfinite(l) && abs(imag(l)) < 1e-6 && real(l) > 0.1, lambdas) .|> real)

        @test length(real_pos) >= 1
        @test abs(real_pos[1] - (π / 2)^2) < 0.1
    end

    @testset "Sparse solve path matches dense path" begin
        domain = Domain(
            x = FourierTransformed(),
            z = Chebyshev(N=24, lower=-1.0, upper=1.0)
        )
        prob = EVP(domain, variables=[:u], eigenvalue=:sigma)
        @equation prob sigma * u == -dx(dx(u)) - dz(dz(u))
        @bc prob left(u) == 0
        @bc prob right(u) == 0
        cache = discretize(prob)
        ks = [0.5, 1.5]

        rd = solve(cache, ks; sigma_0=3.0, method=:Arnoldi, sparse=false)
        rs = solve(cache, ks; sigma_0=3.0, method=:Arnoldi, sparse=true)

        for i in eachindex(ks)
            @test rd[i].converged && rs[i].converged
            @test abs(rd[i].eigenvalues[1] - rs[i].eigenvalues[1]) < 1e-6
        end
    end

    @testset "Variable-coefficient field multiply stays banded (Olver–Townsend)" begin
        mkc(N) = begin
            d = Domain(x=FourierTransformed(), z=Chebyshev(N=N, lower=-1.0, upper=1.0))
            p = EVP(d, variables=[:u], eigenvalue=:sigma)
            z = gridpoints(d, :z); p[:U] = @. tanh(4z)
            @equation p sigma * u == U * dz(u) - dz(dz(u))
            @bc p left(u) == 0
            @bc p right(u) == 0
            discretize(p)
        end

        # Banded C^(λ) multiplication: assembled fill falls with N and routes
        # sparse at high N. The old dense S·M·S⁻¹ form plateaued near 0.26.
        @test BiGSTARS._assembled_density(mkc(768)) < 0.12

        # Physics preserved: smallest positive-real eigenvalue is unchanged.
        A, B = assemble(mkc(128), 1.0)
        ev = eigvals(Matrix(A), Matrix(B))
        λ = sort(filter(e -> isfinite(e) && real(e) > 1e-3, ev), by=abs)[1]
        @test abs(λ - 1.6880521668) < 1e-6
    end

    @testset "Density gate: sparse for banded, dense for filled" begin
        # constant-coefficient 2D Laplacian → banded/block → very sparse
        db = Domain(x=FourierTransformed(), y=Fourier(16, [0, 1.0]), z=Chebyshev(12, [0, 1.0]))
        pb = EVP(db, variables=[:u], eigenvalue=:sigma)
        @equation pb sigma * u == -dx(dx(u)) - dy(dy(u)) - dz(dz(u))
        @bc pb left(u) == 0
        @bc pb right(u) == 0
        cb = discretize(pb)
        @test BiGSTARS._assembled_density(cb) < 0.10        # → auto-selects sparse

        # rich variable coefficient → fills in → above threshold
        df = Domain(x=FourierTransformed(), z=Chebyshev(N=64, lower=-1.0, upper=1.0))
        pf = EVP(df, variables=[:u], eigenvalue=:sigma)
        z = gridpoints(df, :z); pf[:U] = @. tanh(4z)
        @equation pf sigma * u == U * dz(u) - dz(dz(u))
        @bc pf left(u) == 0
        @bc pf right(u) == 0
        cf = discretize(pf)
        @test BiGSTARS._assembled_density(cf) > 0.10        # → auto-selects dense
    end

    @testset "Derived-var augmentation matches substitution path" begin
        # Legacy (no derived variables): sigma*psi = -dz(dz(psi)) - dx(dx(psi))
        # Eigenvalues: (n*pi)^2 + k^2 (positive).
        function make_legacy()
            domain = Domain(x=FourierTransformed(), z=Chebyshev(N=24, lower=0.0, upper=1.0))
            prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)
            @equation prob sigma * psi == -dz(dz(psi)) - dx(dx(psi))
            @bc prob left(psi) == 0
            @bc prob right(psi) == 0
            discretize(prob)
        end
        # Augmented: introduce v = -dz(dz(psi)) - dx(dx(psi)) as a derived variable.
        # @derive v v = rhs means Op(v) = v = rhs, i.e. v is an alias for the rhs expression.
        # The constraint assembled is: 0 == v - (-dz(dz(psi)) - dx(dx(psi))).
        function make_augmented()
            domain = Domain(x=FourierTransformed(), z=Chebyshev(N=24, lower=0.0, upper=1.0))
            prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)
            @derive prob v v = -dz(dz(psi)) - dx(dx(psi))
            @equation prob sigma * psi == v
            @bc prob left(psi) == 0
            @bc prob right(psi) == 0
            discretize(prob; augment_derived=true)
        end
        function smallest(cache, k)
            A, B = assemble(cache, k)
            ev = eigvals(Matrix(A), Matrix(B))
            sort(filter(e -> isfinite(e) && real(e) > 0.1, ev), by=real)[1]
        end
        cache_leg = make_legacy()
        cache_aug = make_augmented()
        @test cache_aug.derived_var_order == [:v]
        @test cache_aug.N_vars == 2
        @test cache_leg.N_vars == 1
        for k in (0.5, 1.5)
            @test abs(smallest(cache_leg, k) - smallest(cache_aug, k)) < 1e-8
        end
    end

    @testset "Augmented derived var with a real operator (analytic)" begin
        # Genuine operator-defined derived variable that the legacy inverse path
        # CANNOT handle (the 1D _sparse_block_inverse ignores BCs, leaving dz²
        # singular). Augmentation keeps v as an unknown with its BCs as real rows.
        #   σψ = v,  dz²v = ψ on z∈[0,1],  v(0)=v(1)=0, ψ(0)=ψ(1)=0
        #   ⇒ σ·dz²ψ = ψ,  ψ=sin(nπz)  ⇒  σ = -1/(nπ)².  Smallest |σ| = -1/π².
        domain = Domain(z=Chebyshev(N=32, lower=0.0, upper=1.0))
        prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)
        @derive prob v dz(dz(v)) = psi
        @derive_bc prob v left(v) == 0
        @derive_bc prob v right(v) == 0
        @equation prob sigma * psi == v
        @bc prob left(psi) == 0
        @bc prob right(psi) == 0

        cache = discretize(prob; augment_derived=true)
        @test cache.N_vars == 2
        @test cache.derived_var_order == [:v]

        A, B = assemble(cache, 0.0)
        ev = eigvals(Matrix(A), Matrix(B))
        finite_real = real.(filter(e -> isfinite(e) && abs(imag(e)) < 1e-6, ev))
        # The analytic n=1,2,3 eigenvalues must each appear in the spectrum.
        for n in 1:3
            target = -1 / (n * π)^2
            @test minimum(abs.(finite_real .- target)) < 1e-4
        end
    end

    @testset "Reconstruct augmented derived variable from eigenvector" begin
        domain = Domain(z=Chebyshev(N=24, lower=0.0, upper=1.0))
        prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)
        @derive prob v dz(dz(v)) = psi
        @derive_bc prob v left(v) == 0
        @derive_bc prob v right(v) == 0
        @equation prob sigma * psi == v
        @bc prob left(psi) == 0
        @bc prob right(psi) == 0

        cache = discretize(prob; augment_derived=true)
        A, B = assemble(cache, 0.0)
        F = eigen(Matrix(A), Matrix(B))
        # pick the eigenvector of the eigenvalue closest to the physical n=1 mode -1/π²
        target = -1 / π^2
        idx = argmin([isfinite(λ) ? abs(λ - target) : Inf for λ in F.values])
        vec = F.vectors[:, idx]

        Np = cache.N_per_var
        v_block = ComplexF64.(vec[Np+1:2Np])            # v is the 2nd variable block
        v_recon = reconstruct(cache, prob, vec, 0.0, :v)
        @test length(v_recon) == Np
        @test norm(v_recon - v_block) / max(norm(v_block), eps()) < 1e-10
    end

    @testset "Augmented derived problem assembles sparse" begin
        # The augmented operator-defined derived problem assembles a banded
        # (sparse) operator — no dense Op⁻¹. (The legacy inverse path cannot
        # build this case at all, so there is no dense baseline to compare.)
        function mkc(N)
            domain = Domain(z=Chebyshev(N=N, lower=0.0, upper=1.0))
            prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)
            @derive prob v dz(dz(v)) = psi
            @derive_bc prob v left(v) == 0
            @derive_bc prob v right(v) == 0
            @equation prob sigma * psi == v
            @bc prob left(psi) == 0
            @bc prob right(psi) == 0
            discretize(prob; augment_derived=true)
        end
        A, _ = assemble(mkc(64), 0.0)
        @test issparse(A)
        @test nnz(A) / length(A) < 0.10        # banded, far from dense
    end

    @testset "Spurious modes filtered from descriptor (augmented) solve" begin
        # The augmented system has a singular B (zero rows for the constraint and
        # boundary equations) → infinite eigenvalues. The solver must return only
        # physical modes, for BOTH methods. Physical: σ=-1/(nπ)²; nearest -0.1 is -1/π².
        domain = Domain(z=Chebyshev(N=48, lower=0.0, upper=1.0))
        prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)
        @derive prob v dz(dz(v)) = psi
        @derive_bc prob v left(v) == 0
        @derive_bc prob v right(v) == 0
        @equation prob sigma * psi == v
        @bc prob left(psi) == 0
        @bc prob right(psi) == 0
        cache = discretize(prob; augment_derived=true)

        for method in (:Arnoldi, :Krylov)
            res = solve(cache; sigma_0=-0.1, method=method, nev=4, n_tries=2)
            @test res[1].converged
            # no huge spurious (infinite) modes survive the filter
            @test all(e -> abs(e) < 0.5, res[1].eigenvalues)
            # the physical n=1 mode is present
            @test minimum(abs.(res[1].eigenvalues .- (-1/π^2))) < 1e-4
        end
    end

    @testset "Derived variables are augmented by default" begin
        domain = Domain(z=Chebyshev(N=16, lower=0.0, upper=1.0))
        prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)
        @derive prob v dz(dz(v)) = psi
        @derive_bc prob v left(v) == 0
        @derive_bc prob v right(v) == 0
        @equation prob sigma * psi == v
        @bc prob left(psi) == 0
        @bc prob right(psi) == 0

        # Default (no kwarg) now augments → v is a real variable block.
        cache = discretize(prob)
        @test cache.derived_var_order == [:v]
        @test cache.N_vars == 2

        # Opt-out restores the legacy eliminate-via-inverse path.
        cache_legacy = discretize(prob; augment_derived=false)
        @test isempty(cache_legacy.derived_var_order)
        @test cache_legacy.N_vars == 1
    end

end
