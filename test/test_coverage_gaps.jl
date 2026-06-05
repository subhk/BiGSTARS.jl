using Test
using BiGSTARS
using BiGSTARS: VarNode, ParamNode, ConstNode, EigenvalueNode, WavenumberNode,
                DerivNode, BinaryOpNode, UnaryOpNode, SubstitutionNode
using LinearAlgebra
using SparseArrays

# ──────────────────────────────────────────────────────────────────────────────
# Targeted tests for branches the main suite leaves uncovered (see coverage audit).
# Grouped by source module.
# ──────────────────────────────────────────────────────────────────────────────

@testset "Coverage gaps" begin

    @testset "utils: remove_evals by imaginary and magnitude" begin
        λ = ComplexF64[3.0 + 0im, 1.0 + 2im, 2.0 - 1im, 0.5 + 5im]
        Χ = Matrix{ComplexF64}(I, 4, 4)

        # "I" — keep imag ∈ [-1.5, 2.5]: drops 0.5+5im
        lI, χI = remove_evals(λ, Χ, -1.5, 2.5, "I")
        @test all(e -> -1.5 ≤ imag(e) ≤ 2.5, lI)
        @test length(lI) == 3 && size(χI, 2) == 3
        @test length(remove_evals(λ, Χ, -1.5, 2.5, :I)[1]) == 3   # Symbol convenience

        # "M" — keep magnitude ∈ [0, 3]: drops 0.5+5im (|·|≈5)
        lM, _ = remove_evals(λ, Χ, 0.0, 3.0, "M")
        @test all(e -> abs(e) ≤ 3.0, lM)
        @test length(remove_evals(λ, Χ, 0.0, 3.0, :M)[1]) == length(lM)
    end

    @testset "eig_solver: sort_eigenvalues! non-nearest criteria" begin
        λ = ComplexF64[3.0 + 0im, 1.0 + 4im, 2.0 - 2im]
        Χ = Matrix{ComplexF64}(I, 3, 3)

        lR, _ = BiGSTARS.sort_eigenvalues!(λ, Χ, :R; rev=true)
        @test real(lR[1]) == 3.0                       # largest real first
        lI, _ = BiGSTARS.sort_eigenvalues!(λ, Χ, :I; rev=true)
        @test imag(lI[1]) == 4.0                       # largest imag first
        lM, _ = BiGSTARS.sort_eigenvalues!(λ, Χ, :M; rev=true)
        @test abs(lM[1]) ≈ maximum(abs.(λ))            # largest magnitude first
    end

    @testset "expr: show methods for node types" begin
        @test sprint(show, EigenvalueNode(:sigma)) == "sigma"
        @test sprint(show, UnaryOpNode(:-, VarNode(:u))) == "-(u)"
        @test sprint(show, SubstitutionNode(:Lap, ExprNode[VarNode(:u), VarNode(:w)])) ==
              "Lap(u, w)"
    end

    @testset "transforms: error paths and chebyshev round-trip" begin
        # FourierTransformed direction has no grid → cannot differentiate
        d = Domain(x = FourierTransformed(), z = Chebyshev(N=8, lower=0.0, upper=1.0))
        @test_throws ErrorException differentiate(zeros(8), d, :x)

        # Unknown spectral filter
        dy = Domain(y = Fourier(8, [0.0, 1.0]))
        @test_throws ErrorException differentiate(zeros(8), dy, :y; filter=:bogus)

        # to_coefficients dispatch
        c = to_coefficients(Float64.(1:8), :chebyshev)
        @test length(c) == 8
        @test_throws ErrorException to_coefficients(zeros(8), :bogus)

        # to_physical chebyshev requires x; round-trips through chebyshev_evaluate
        @test_throws ErrorException to_physical(c, :chebyshev)              # missing x
        xpts = chebyshev_points(8)
        @test to_physical(c, :chebyshev; x=xpts) ≈ Float64.(1:8)
        @test_throws ErrorException to_physical(c, :bogus)
    end

    @testset "domain: FourierTransformed / Fourier operator paths" begin
        d = Domain(x = FourierTransformed(),
                   y = Fourier(8, [0.0, 1.0]),
                   z = Chebyshev(N=6, lower=0.0, upper=1.0))

        @test_throws ErrorException gridpoints(d, :x)                  # no grid
        @test_throws ErrorException BiGSTARS.get_diff_operator(d, :x, 1)
        @test_throws ErrorException BiGSTARS.get_N(d, :x)

        Dy = BiGSTARS.get_diff_operator(d, :y, 1)                      # Fourier operator
        @test size(Dy) == (8, 8)
    end

    @testset "boundary: interior eval, deriv error, deriv-order on op nodes" begin
        @test BiGSTARS.chebyshev_T(2, 0.5) ≈ cos(2 * acos(0.5))        # interior |x|<1
        @test_throws ErrorException BiGSTARS.chebyshev_T_deriv(3, 1, 0.5)  # not a boundary

        # count_bc_deriv_order: max over binary, descends through unary
        e = BinaryOpNode(:+, DerivNode(DerivNode(VarNode(:u), :z), :z), VarNode(:w))
        @test BiGSTARS.count_bc_deriv_order(e) == 2
        @test BiGSTARS.count_bc_deriv_order(UnaryOpNode(:-, DerivNode(VarNode(:u), :z))) == 1
    end

    @testset "k_separation: distribute_products + extract_k branches" begin
        # (a + b) * c → a*c + b*c   (left distributes)
        left_sum = BinaryOpNode(:*, BinaryOpNode(:+, VarNode(:a), VarNode(:b)), VarNode(:c))
        dl = BiGSTARS.distribute_products(left_sum)
        @test dl isa BinaryOpNode && dl.op == :+

        # a * (b - c) → a*b - a*c   (right distributes)
        right_sum = BinaryOpNode(:*, VarNode(:a), BinaryOpNode(:-, VarNode(:b), VarNode(:c)))
        dr = BiGSTARS.distribute_products(right_sum)
        @test dr isa BinaryOpNode && dr.op == :-

        # d(a + b) → da + db   (derivative is linear)
        dsum = BiGSTARS.distribute_products(DerivNode(BinaryOpNode(:+, VarNode(:a), VarNode(:b)), :z))
        @test dsum isa BinaryOpNode && dsum.op == :+

        # extract_k_power through a unary and a derivative wrapping k
        kw = WavenumberNode(:k_x)
        pu, _ = BiGSTARS.extract_k_power(UnaryOpNode(:-, kw))
        @test pu == 1
        pd, _ = BiGSTARS.extract_k_power(DerivNode(kw, :z))
        @test pd == 1

        # two constant factors merge
        pc, rc = BiGSTARS.extract_k_power(BinaryOpNode(:*, ConstNode(2.0), ConstNode(3.0)))
        @test rc isa ConstNode && rc.value == 6.0 && pc == 0

        # const · k → keep the const, count the k (right-operand-is-k branch)
        pck, rck = BiGSTARS.extract_k_power(BinaryOpNode(:*, ConstNode(2.0), kw))
        @test pck == 1 && rck == ConstNode(2.0)

        # extract_k_powers (per-symbol variant) also merges two constants
        _, rcp = BiGSTARS.extract_k_powers(BinaryOpNode(:*, ConstNode(2.0), ConstNode(3.0)))
        @test rcp isa ConstNode && rcp.value == 6.0
    end

    @testset "substitutions: substitute_args unary + passthrough" begin
        body = UnaryOpNode(:-, VarNode(:A))
        out = BiGSTARS.substitute_args(body, [:A], ExprNode[VarNode(:u)])
        @test out == UnaryOpNode(:-, VarNode(:u))

        # ConstNode is passthrough (the `else` branch)
        @test BiGSTARS.substitute_args(ConstNode(2.0), [:A], ExprNode[VarNode(:u)]) ==
              ConstNode(2.0)
    end

    @testset "lowering: collect_var_names descends into substitution args" begin
        node = SubstitutionNode(:Lap, ExprNode[VarNode(:u), DerivNode(VarNode(:w), :z)])
        @test BiGSTARS.collect_var_names(node) == Set([:u, :w])
    end

    @testset "evp: first_chebyshev_coord returns nothing without Chebyshev" begin
        d = Domain(y = Fourier(8, [0.0, 1.0]))
        prob = EVP(d, variables=[:w], eigenvalue=:sigma)
        @test BiGSTARS.first_chebyshev_coord(prob) === nothing
    end

    @testset "macros: resolve_symbol branches" begin
        d = Domain(z = Chebyshev(N=8, lower=0.0, upper=1.0))
        prob = EVP(d, variables=[:u], eigenvalue=:sigma)
        prob[:U] = ones(8)
        @substitution prob Lap(A) = dz(dz(A))

        @test BiGSTARS.resolve_symbol(prob, :Lap) == ParamNode(:Lap)   # substitution name
        @test BiGSTARS.resolve_symbol(prob, :U) == ParamNode(:U)
        @test_throws ErrorException BiGSTARS.resolve_symbol(prob, :nope)
    end

    @testset "macros: parse_expr_ast literal + multi-arg fold" begin
        d = Domain(z = Chebyshev(N=8, lower=0.0, upper=1.0))
        prob = EVP(d, variables=[:u], eigenvalue=:sigma)
        # numeric literal coefficient → ConstNode; u + u + u → multi-arg + fold
        @equation prob sigma * u == 2.0 * u + u + u
        @test length(prob.equations) == 1
        @test prob.equations[1].rhs isa BinaryOpNode
    end

    @testset "macros: substitution body shapes" begin
        d = Domain(z = Chebyshev(N=8, lower=0.0, upper=1.0))
        prob = EVP(d, variables=[:w], eigenvalue=:sigma)
        prob[:U] = ones(8)

        @substitution prob WithParam(A) = U * A          # bare param → resolve_substitution_symbol
        @substitution prob WithVar(A) = w + A            # variable symbol → resolve_symbol branch
        @substitution prob Neg(A) = -dz(A)               # unary minus body
        @substitution prob Tri(A) = A + A + A            # multi-arg fold
        @substitution prob Nested(A) = Neg(A) + A        # nested SubstitutionNode in body

        @test haskey(prob.substitutions, :WithParam)
        @test prob.substitutions[:WithVar].body.left == VarNode(:w)
        @test prob.substitutions[:Neg].body isa UnaryOpNode
        @test prob.substitutions[:Tri].body isa BinaryOpNode
        @test prob.substitutions[:Nested].body.left isa SubstitutionNode
    end

    @testset "macros: malformed inputs throw" begin
        me(e) = macroexpand(@__MODULE__, e)
        @test_throws Exception me(:(@substitution a b c))   # >2 args / not Name(A)=expr
        @test_throws Exception me(:(@equation a b c))       # not [prob] lhs=rhs
        @test_throws Exception me(:(@bc a b c))             # not [prob] left(e)=val
        @test_throws Exception me(:(@derive v dz(v)))       # Form 2 lhs not `lhs = rhs`
        @test_throws Exception BiGSTARS.parse_substitution_body(:([1, 2]), Symbol[])
    end

    @testset "macros: @derive form 1 (Op_var = rhs) and @equation = form" begin
        d = Domain(x = FourierTransformed(), z = Chebyshev(N=8, lower=0.0, upper=1.0))
        prob = EVP(d, variables=[:w], eigenvalue=:sigma)
        @substitution prob Dh(A) = dz(dz(A))

        @derive prob Dh_v = dz(w)                # Form 1, explicit prob (underscore split)
        @test haskey(prob.derived_vars, :v)

        @derive Dh_v2 = dz(w)                    # Form 1, no prob → active-problem path
        @test haskey(prob.derived_vars, :v2)

        # @equation with single `=` (assignment form, not `==`)
        @equation prob sigma * w = -dz(dz(w))
        @test length(prob.equations) == 1
    end

    @testset "macros: @derive_bc with derivative order" begin
        d = Domain(z = Chebyshev(N=8, lower=0.0, upper=1.0))
        prob = EVP(d, variables=[:psi], eigenvalue=:sigma)
        @derive prob v dz(dz(v)) = psi
        @derive_bc prob v left(dz(v)) == 0       # exercises the deriv-order counting loop
        bcs = prob.derived_vars[:v].bcs
        @test length(bcs) == 1
        @test bcs[1].expr isa DerivNode          # deriv_order==1 ⇒ dz(v)
        @test bcs[1].side == :left

        # non-derivative inner expression ⇒ deriv-counting loop breaks at order 0
        @derive_bc prob v right(abs(v)) == 0
        @test length(prob.derived_vars[:v].bcs) == 2
    end

    @testset "macros: @bc with explicit coordinate argument" begin
        d = Domain(z = Chebyshev(N=8, lower=0.0, upper=1.0))
        prob = EVP(d, variables=[:u], eigenvalue=:sigma)
        @bc prob left(u, z) == 0                 # explicit coord (3-arg left/right form)
        @test prob.bcs[1].coord == :z
    end

    @testset "macros: @derive form 2 with scalar literal + parameter in operator" begin
        d = Domain(z = Chebyshev(N=8, lower=0.0, upper=1.0))
        prob = EVP(d, variables=[:psi], eigenvalue=:sigma)
        prob[:alpha] = 2.0
        # lhs operator carries a numeric literal (ConstNode) and a parameter (ParamNode,
        # prob_ref=nothing path); rhs references a variable.
        @derive prob v 3.0 * dz(dz(v)) + alpha * v = psi
        @test haskey(prob.derived_vars, :v)
        @test haskey(prob.substitutions, :_derive_v)
    end

    @testset "discretize: matrix-valued parameter operator" begin
        N = 8
        d = Domain(z = Chebyshev(N=N, lower=0.0, upper=1.0))
        prob = EVP(d, variables=[:u], eigenvalue=:sigma)
        prob[:M] = Matrix{Float64}(I, N, N)              # N_per_var × N_per_var matrix param
        @equation prob sigma * u == M * u - dz(dz(u))
        @bc prob left(u) == 0
        @bc prob right(u) == 0
        cache = discretize(prob)
        A, B = assemble(cache, 0.0)
        @test size(A) == (N, N)
        ev = eigvals(Matrix(A), Matrix(B))
        @test any(e -> isfinite(e) && real(e) > 0.5, ev)
    end

    @testset "reconstruct: matrix-valued parameter operator in @compute" begin
        N = 8
        d = Domain(x = FourierTransformed(), z = Chebyshev(N=N, lower=0.0, upper=1.0))
        prob = EVP(d, variables=[:w], eigenvalue=:sigma)
        prob[:M] = Matrix{Float64}(I, N, N)             # N_per_var × N_per_var matrix param
        @equation prob sigma * w == -dz(dz(w))
        @bc prob left(w) == 0
        @bc prob right(w) == 0
        cache = discretize(prob)

        ev = rand(ComplexF64, cache.N_total)
        Np = cache.N_per_var
        @compute_setup cache ev 0.5
        @compute mw = M * w                              # matrix-parameter operator branch
        @test length(mw) == Np
        @test mw ≈ ev[1:Np]                             # M = I ⇒ operator is identity

        # _evaluate_expr rejects a node it cannot evaluate (e.g. the eigenvalue)
        @test_throws ErrorException BiGSTARS._evaluate_expr(
            EigenvalueNode(:sigma), prob, cache, ev, 0.5)
    end

    @testset "discretize: dynamic (eigenvalue-dependent) boundary condition" begin
        N = 16
        mk(dynamic) = begin
            d = Domain(z = Chebyshev(N=N, lower=0.0, upper=1.0))
            p = EVP(d, variables=[:u], eigenvalue=:sigma)
            @equation p sigma * u == -dz(dz(u))
            @bc p left(u) == 0
            if dynamic
                @bc p right(sigma * u + dz(u)) == 0   # eigenvalue appears on the boundary
            else
                @bc p right(dz(u)) == 0               # static counterpart
            end
            discretize(p)
        end
        Ad, Bd = assemble(mk(true), 0.0)
        As, Bs = assemble(mk(false), 0.0)
        @test size(Ad) == (N, N) && size(Bd) == (N, N)
        @test all(isfinite, nonzeros(Ad)) && all(isfinite, nonzeros(Bd))
        # The eigenvalue term in the dynamic BC lands in B; the static BC's B row is empty.
        @test nnz(Bd) > nnz(Bs)
        @test any(isfinite, eigvals(Matrix(Ad), Matrix(Bd)))
    end

    @testset "discretize: field-coefficient and derivative-of-product BCs" begin
        N = 24
        d = Domain(z = Chebyshev(N=N, lower=0.0, upper=1.0))
        p = EVP(d, variables=[:u], eigenvalue=:sigma)
        p[:U] = ones(N)                       # U ≡ 1 at the boundaries
        @equation p sigma * u == -dz(dz(u))
        @bc p left(U * u) == 0                # field-param coefficient at boundary → u(0)=0
        @bc p right(dz(U * u)) == 0           # derivative of a product → u'(1)=0
        cache = discretize(p)
        A, B = assemble(cache, 0.0)
        ev = eigvals(Matrix(A), Matrix(B))
        pos = sort(real.(filter(e -> isfinite(e) && abs(imag(e)) < 1e-6 && real(e) > 0.1, ev)))
        # σu=-u'', u(0)=0, u'(1)=0 ⇒ σ_n=((n-½)π)²; smallest = (π/2)²
        @test !isempty(pos)
        @test abs(pos[1] - (π / 2)^2) < 1e-3
    end

    @testset "discretize: derivatives of coefficient expressions (T-basis branches)" begin
        # Each additive term is a derivative OF a coefficient expression, routing through
        # discretize_expr_in_T: ConstNode (2u), field ParamNode (U·u), unary (-u),
        # Fourier derivative (dy(u)), and the conservative flux form dz(U·dz(u)).
        d = Domain(y = Fourier(8, [0.0, 1.0]), z = Chebyshev(N=12, lower=0.0, upper=1.0))
        p = EVP(d, variables=[:u], eigenvalue=:sigma)
        _, Z = meshgrid(d, :y, :z); p[:U] = vec(@. 1.0 + 0.3 * Z)
        @equation p sigma * u == -dz(U * dz(u)) + dz(2.0 * u) - dz(U * u) + dz(-u) + dz(dy(u))
        @bc p left(u) == 0
        @bc p right(u) == 0
        cache = discretize(p)
        A, B = assemble(cache, 0.0)
        @test size(A, 1) == cache.N_total
        @test issparse(A)
        @test all(isfinite, nonzeros(A))
    end

    @testset "discretize: flux-form variable coefficient is resolution-stable" begin
        # σu = -d/dz(U du/dz), U(z)=1+½z, u(0)=u(1)=0. Smallest eigenvalue must be
        # stable under refinement (proof the conservative operator assembles correctly).
        function smallest(N)
            d = Domain(z = Chebyshev(N=N, lower=0.0, upper=1.0))
            p = EVP(d, variables=[:u], eigenvalue=:sigma)
            z = gridpoints(d, :z); p[:U] = @. 1.0 + 0.5 * z
            @equation p sigma * u == -dz(U * dz(u))
            @bc p left(u) == 0
            @bc p right(u) == 0
            A, B = assemble(discretize(p), 0.0)
            ev = eigvals(Matrix(A), Matrix(B))
            minimum(real.(filter(e -> isfinite(e) && abs(imag(e)) < 1e-6 && real(e) > 0.1, ev)))
        end
        @test abs(smallest(24) - smallest(40)) < 1e-4
    end

    @testset "discretize: order-0 field multiply (T-basis S·M·S⁻¹ fallback)" begin
        # No Chebyshev derivative anywhere ⇒ highest_cheb_order==0, so the banded
        # C^(λ) multiply is skipped and the field coefficient routes through the
        # T-basis multiply fallback (_apply_field_multiply / _full_conversion_inv).
        d = Domain(y = Fourier(8, [0.0, 1.0]), z = Chebyshev(N=8, lower=0.0, upper=1.0))
        p = EVP(d, variables=[:u], eigenvalue=:sigma)
        _, Z = meshgrid(d, :y, :z); p[:U] = vec(@. 1.0 + 0.2 * Z)
        @equation p sigma * u == U * u - dy(dy(u))     # field × variable, no dz
        cache = discretize(p)
        A, B = assemble(cache, 0.0)
        @test size(A, 1) == cache.N_total
        @test all(isfinite, nonzeros(A))
    end

    # ── Direct unit tests for discretize.jl internal tree/operator helpers ──────
    # These pure functions have many branches that the integration pipeline reaches
    # only for specific (rare) expression shapes; exercising them directly is exact
    # and deterministic.

    @testset "discretize internals: strip_eigenvalue + _try_strip" begin
        s = EigenvalueNode(:sigma)
        @test BiGSTARS.strip_eigenvalue(BinaryOpNode(:*, s, VarNode(:u))) == VarNode(:u)
        @test BiGSTARS.strip_eigenvalue(BinaryOpNode(:*, VarNode(:u), s)) == VarNode(:u)
        # product chain on the left: (σ·U)·u → U·u
        @test BiGSTARS.strip_eigenvalue(
            BinaryOpNode(:*, BinaryOpNode(:*, s, ParamNode(:U)), VarNode(:u))) ==
            BinaryOpNode(:*, ParamNode(:U), VarNode(:u))
        # product chain on the right: u·(σ·U) → u·U
        @test BiGSTARS.strip_eigenvalue(
            BinaryOpNode(:*, VarNode(:u), BinaryOpNode(:*, s, ParamNode(:U)))) ==
            BinaryOpNode(:*, VarNode(:u), ParamNode(:U))
        @test_throws ErrorException BiGSTARS.strip_eigenvalue(VarNode(:u))
    end

    @testset "discretize internals: _strip_eigenvalue_from_term" begin
        s = EigenvalueNode(:sigma)
        f(e) = BiGSTARS._strip_eigenvalue_from_term(e, :sigma)
        @test f(s) == ConstNode(1.0)
        @test f(BinaryOpNode(:*, s, VarNode(:u))) == VarNode(:u)          # left σ
        @test f(BinaryOpNode(:*, VarNode(:u), s)) == VarNode(:u)          # right σ
        # right σ with a coefficient: u·(2σ) → u·2
        @test f(BinaryOpNode(:*, VarNode(:u), BinaryOpNode(:*, ConstNode(2.0), s))) ==
              BinaryOpNode(:*, VarNode(:u), ConstNode(2.0))
        @test f(UnaryOpNode(:-, BinaryOpNode(:*, s, VarNode(:u)))) ==
              UnaryOpNode(:-, VarNode(:u))                                # unary
        @test f(VarNode(:u)) == VarNode(:u)                              # passthrough
        @test f(BinaryOpNode(:*, s, s)) isa BinaryOpNode                 # σ on both sides
    end

    @testset "discretize internals: _replace_var" begin
        g(e) = BiGSTARS._replace_var(e, :a, :b)
        @test g(VarNode(:a)) == VarNode(:b)
        @test g(VarNode(:c)) == VarNode(:c)
        @test g(DerivNode(VarNode(:a), :z)) == DerivNode(VarNode(:b), :z)
        @test g(BinaryOpNode(:+, VarNode(:a), VarNode(:c))) ==
              BinaryOpNode(:+, VarNode(:b), VarNode(:c))
        @test g(UnaryOpNode(:-, VarNode(:a))) == UnaryOpNode(:-, VarNode(:b))
        @test g(SubstitutionNode(:F, ExprNode[VarNode(:a)])) ==
              SubstitutionNode(:F, ExprNode[VarNode(:b)])
        @test g(ConstNode(1.0)) == ConstNode(1.0)                        # passthrough
    end

    @testset "discretize internals: _distribute_deriv (BC derivative distribution)" begin
        h(e) = BiGSTARS._distribute_deriv(e, :z, 1)
        @test h(BinaryOpNode(:+, VarNode(:a), VarNode(:b))) ==
              BinaryOpNode(:+, DerivNode(VarNode(:a), :z), DerivNode(VarNode(:b), :z))
        # scalar/param on the right of a product stays outside the derivative
        @test h(BinaryOpNode(:*, VarNode(:a), ParamNode(:U))) ==
              BinaryOpNode(:*, DerivNode(VarNode(:a), :z), ParamNode(:U))
        @test h(DerivNode(VarNode(:a), :y)) == DerivNode(DerivNode(VarNode(:a), :z), :y)
        @test h(UnaryOpNode(:-, VarNode(:a))) == UnaryOpNode(:-, DerivNode(VarNode(:a), :z))
        @test_throws ErrorException h(BinaryOpNode(:*, VarNode(:a), VarNode(:b)))  # non-scalar product
        @test_throws ErrorException h(EigenvalueNode(:sigma))                      # unsupported
    end

    @testset "discretize internals: operator/expr builders on hand-built nodes" begin
        du = Domain(z = Chebyshev(N=8, lower=0.0, upper=1.0))
        pu = EVP(du, variables=[:u], eigenvalue=:sigma)
        pu[:E] = 2.0
        pu[:U] = collect(1.0:8.0)
        pu[:M] = Matrix{Float64}(I, 8, 8)
        Np = 8

        # _discretize_operator branches (T basis, dummy=nothing → VarNode is identity)
        do_(e) = BiGSTARS._discretize_operator(e, nothing, pu, Np)
        @test do_(BinaryOpNode(:+, VarNode(:u), VarNode(:u))) == 2 * BiGSTARS._full_identity(du)
        @test do_(BinaryOpNode(:-, VarNode(:u), VarNode(:u))) == 0 * BiGSTARS._full_identity(du)
        @test do_(BinaryOpNode(:*, VarNode(:u), ConstNode(3.0))) == 3 * BiGSTARS._full_identity(du)
        @test do_(BinaryOpNode(:*, VarNode(:u), VarNode(:u))) isa AbstractMatrix   # fallback *
        @test do_(ParamNode(:E)) == 2 * BiGSTARS._full_identity(du)                # scalar param
        @test do_(ConstNode(5.0)) == 5 * BiGSTARS._full_identity(du)
        @test do_(WavenumberNode(:k_x)) == BiGSTARS._full_identity(du)
        @test_throws ErrorException do_(EigenvalueNode(:sigma))

        # discretize_expr_in_T branches
        dt(e) = BiGSTARS.discretize_expr_in_T(e, pu, Np)
        @test dt(ParamNode(:E)) == ComplexF64(2.0) * sparse(I, Np, Np)             # scalar
        @test dt(ParamNode(:M)) == ComplexF64.(pu[:M])                             # matrix
        @test dt(BinaryOpNode(:+, VarNode(:u), VarNode(:u))) isa AbstractMatrix    # +
        @test dt(BinaryOpNode(:-, VarNode(:u), VarNode(:u))) isa AbstractMatrix    # -
        @test_throws ErrorException dt(EigenvalueNode(:sigma))                     # unsupported

        # discretize_expr branches (C^(p) basis), ctx=nothing
        de(e) = BiGSTARS.discretize_expr(e, pu, Np, 1, nothing)
        @test de(ParamNode(:E)) isa AbstractMatrix                                 # scalar param (275)
        @test de(ParamNode(:U)) isa AbstractMatrix                                 # field param standalone (280)
        @test de(ConstNode(3.0)) isa AbstractMatrix                                # const standalone (284)
        @test de(BinaryOpNode(:+, VarNode(:u), VarNode(:u))) isa AbstractMatrix    # +
        @test de(BinaryOpNode(:-, VarNode(:u), VarNode(:u))) isa AbstractMatrix    # -
        @test de(BinaryOpNode(:*, ConstNode(2.0), ConstNode(3.0))) isa AbstractMatrix  # scalar·scalar
        @test de(BinaryOpNode(:*, VarNode(:u), ParamNode(:E))) isa AbstractMatrix  # scalar on right
        @test de(BinaryOpNode(:*, DerivNode(VarNode(:u), :z), ParamNode(:U))) isa AbstractMatrix # field on right
        @test de(UnaryOpNode(:-, VarNode(:u))) isa AbstractMatrix                  # unary
        @test_throws ErrorException de(EigenvalueNode(:sigma))                     # cannot discretize

        # _extract_field_param: scalar·field both orders
        @test BiGSTARS._extract_field_param(BinaryOpNode(:*, ConstNode(2.0), ParamNode(:U)), pu)[1] == ParamNode(:U)
        @test BiGSTARS._extract_field_param(BinaryOpNode(:*, ParamNode(:U), ConstNode(2.0)), pu)[2] == ComplexF64(2.0)

        # _converted_field_multiply (standalone field, T-basis path)
        @test BiGSTARS._converted_field_multiply(ParamNode(:U), pu, Np, 0, nothing) isa AbstractMatrix
    end

    @testset "discretize internals: FourierTransformed-direction guards" begin
        dft = Domain(x = FourierTransformed(), z = Chebyshev(N=8, lower=0.0, upper=1.0))
        pft = EVP(dft, variables=[:u], eigenvalue=:sigma)
        # A derivative in a FourierTransformed direction must have been lowered earlier.
        @test_throws ErrorException BiGSTARS.discretize_expr(
            DerivNode(VarNode(:u), :x), pft, 8, 1, nothing)
        @test_throws ErrorException BiGSTARS.discretize_expr_in_T(
            DerivNode(VarNode(:u), :x), pft, 8)
    end

    @testset "discretize internals: BC row builders on hand-built nodes" begin
        du = Domain(z = Chebyshev(N=8, lower=0.0, upper=1.0))
        pu = EVP(du, variables=[:u], eigenvalue=:sigma)

        # _try_bc_scalar: negation and product-of-scalars
        @test BiGSTARS._try_bc_scalar(UnaryOpNode(:-, ConstNode(2.0)), :left, :z, pu) == ComplexF64(-2.0)
        @test BiGSTARS._try_bc_scalar(BinaryOpNode(:*, ConstNode(2.0), ConstNode(3.0)), :left, :z, pu) == ComplexF64(6.0)

        # _build_bc_row_1d!: ConstNode is a no-op; var·scalar (scalar on right); negation
        row = zeros(ComplexF64, 8)
        BiGSTARS._build_bc_row_1d!(row, ConstNode(2.0), :left, :z, pu)
        @test all(==(0), row)                                            # bare const → no row
        BiGSTARS._build_bc_row_1d!(row, BinaryOpNode(:*, VarNode(:u), ConstNode(2.0)), :left, :z, pu)
        @test any(!=(0), row)                                            # scalar-on-right `*` branch
        row2 = zeros(ComplexF64, 8)
        BiGSTARS._build_bc_row_1d!(row2, UnaryOpNode(:-, VarNode(:u)), :left, :z, pu)
        @test any(!=(0), row2)                                           # unary minus

        # error branches
        @test_throws ErrorException BiGSTARS._build_bc_row_1d!(
            zeros(ComplexF64, 8), BinaryOpNode(:*, VarNode(:u), VarNode(:u)), :left, :z, pu)
        @test_throws ErrorException BiGSTARS._build_bc_row_1d!(
            zeros(ComplexF64, 8), EigenvalueNode(:sigma), :left, :z, pu)
        # derivative in the wrong (non-boundary) direction is rejected
        @test_throws ErrorException BiGSTARS._build_bc_row_1d!(
            zeros(ComplexF64, 8), DerivNode(VarNode(:u), :y), :left, :z, pu)
    end

    @testset "discretize internals: _try_strip, chained-deriv-any" begin
        s = EigenvalueNode(:sigma)
        @test BiGSTARS._try_strip(s) == ConstNode(1.0)
        @test BiGSTARS._try_strip(BinaryOpNode(:*, VarNode(:u), s)) == VarNode(:u)   # right σ
        # nested recursion, left and right
        @test BiGSTARS._try_strip(
            BinaryOpNode(:*, BinaryOpNode(:*, s, ParamNode(:U)), VarNode(:u))) ==
            BinaryOpNode(:*, ParamNode(:U), VarNode(:u))
        @test BiGSTARS._try_strip(
            BinaryOpNode(:*, VarNode(:u), BinaryOpNode(:*, s, ParamNode(:U)))) ==
            BinaryOpNode(:*, VarNode(:u), ParamNode(:U))
        @test BiGSTARS._try_strip(VarNode(:u)) === nothing                            # nothing

        # _strip_eigenvalue_from_term: left σ carrying a coefficient (2σ·u → 2·u)
        @test BiGSTARS._strip_eigenvalue_from_term(
            BinaryOpNode(:*, BinaryOpNode(:*, ConstNode(2.0), s), VarNode(:u)), :sigma) ==
            BinaryOpNode(:*, ConstNode(2.0), VarNode(:u))

        # count/unwrap chained derivatives in a non-matching direction
        @test BiGSTARS.count_chained_derivs_any(DerivNode(VarNode(:u), :y), :z) == 0
        @test BiGSTARS.unwrap_chained_derivs_any(DerivNode(VarNode(:u), :y), :z) ==
              DerivNode(VarNode(:u), :y)
    end

    @testset "discretize internals: field-multiply fallback + conversion inverse" begin
        # _full_conversion_inv: Chebyshev (builds S⁻¹ lifted) and Fourier-only (identity)
        dz8 = Domain(z = Chebyshev(N=8, lower=0.0, upper=1.0))
        @test BiGSTARS._full_conversion_inv(dz8, 1, nothing) isa AbstractMatrix
        dyf = Domain(y = Fourier(8, [0.0, 1.0]))
        @test BiGSTARS._full_conversion_inv(dyf, 1, nothing) isa AbstractMatrix     # identity branch

        # _apply_field_multiply: N_per_var-sized M_f → S·M·S⁻¹·G
        Mf = sparse(ComplexF64(1.0) * I, 8, 8); G = sparse(ComplexF64(1.0) * I, 8, 8)
        @test size(BiGSTARS._apply_field_multiply(Mf, G, dz8, :z, 1, 8, nothing, nothing)) == (8, 8)

        # _apply_field_multiply: N_z-sized M_f → lift 1D op to the 2D grid (order ≥ 1)
        d2 = Domain(y = Fourier(4, [0.0, 1.0]), z = Chebyshev(N=4, lower=0.0, upper=1.0))
        Mf1 = sparse(ComplexF64(1.0) * I, 4, 4); G2 = sparse(ComplexF64(1.0) * I, 16, 16)
        @test size(BiGSTARS._apply_field_multiply(Mf1, G2, d2, :z, 1, 16, nothing, nothing)) == (16, 16)
    end

    @testset "discretize internals: _sparse_block_inverse paths" begin
        # 1D (no Fourier dim): direct dense inverse
        dz = Domain(z = Chebyshev(N=8, lower=0.0, upper=1.0))
        @test BiGSTARS._sparse_block_inverse(sparse(ComplexF64(1.0) * I, 8, 8), dz) isa AbstractMatrix

        # Block-diagonal Fourier×Cheb with boundary bordering (BC rows replace last rows)
        d2 = Domain(y = Fourier(4, [0.0, 1.0]), z = Chebyshev(N=6, lower=0.0, upper=1.0))
        opbd = sparse(ComplexF64(1.0) * I, 24, 24)
        bc = BiGSTARS.BoundaryCondition(:left, :z, VarNode(:v), 0.0, false)
        Hbd = BiGSTARS._sparse_block_inverse(opbd, d2; bcs=[bc])
        @test size(Hbd) == (24, 24)

        # Non-block-diagonal operator → dense fallback inverse
        opfull = Matrix(ComplexF64(1.0) * I, 24, 24); opfull[1, 7] = 0.1
        @test BiGSTARS._sparse_block_inverse(sparse(opfull), d2) isa AbstractMatrix
    end

    @testset "discretize: keyword-form assemble dispatch" begin
        d = Domain(x = FourierTransformed(), z = Chebyshev(N=12, lower=-1.0, upper=1.0))
        p = EVP(d, variables=[:u], eigenvalue=:sigma)
        @equation p sigma * u == -dx(dx(u)) - dz(dz(u))
        @bc p left(u) == 0
        @bc p right(u) == 0
        cache = discretize(p)
        Ak, Bk = assemble(cache; k_x=1.5)        # keyword form → single-dim delegate
        Ap, Bp = assemble(cache, 1.5)            # positional form
        @test Ak == Ap && Bk == Bp
    end

end
