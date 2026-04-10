using Test
using BiGSTARS: ExprNode, VarNode, ParamNode, ConstNode, EigenvalueNode,
    WavenumberNode, DerivNode, BinaryOpNode, UnaryOpNode, SubstitutionNode

@testset "Expression Tree" begin

    @testset "Node construction" begin
        v = VarNode(:psi)
        @test v.name == :psi

        p = ParamNode(:U)
        @test p.name == :U

        c = ConstNode(3.14)
        @test c.value == 3.14

        e = EigenvalueNode(:sigma)
        @test e.name == :sigma

        k = WavenumberNode(:k)
        @test k.name == :k

        d = DerivNode(v, :z)
        @test d.coord == :z
        @test d.expr === v

        add = BinaryOpNode(:+, v, p)
        @test add.op == :+
        @test add.left === v
        @test add.right === p

        neg = UnaryOpNode(:-, v)
        @test neg.op == :-
        @test neg.expr === v

        sub = SubstitutionNode(:Lap, [v])
        @test sub.name == :Lap
        @test sub.args == [v]
    end

    @testset "Equality" begin
        a = BinaryOpNode(:+, VarNode(:psi), ConstNode(1.0))
        b = BinaryOpNode(:+, VarNode(:psi), ConstNode(1.0))
        @test a == b

        c = BinaryOpNode(:+, VarNode(:psi), ConstNode(2.0))
        @test a != c

        @test VarNode(:psi) != ParamNode(:psi)
        @test DerivNode(VarNode(:psi), :z) == DerivNode(VarNode(:psi), :z)
        @test DerivNode(VarNode(:psi), :z) != DerivNode(VarNode(:psi), :y)
    end

    @testset "Display" begin
        v = VarNode(:psi)
        p = ParamNode(:U)
        d = DerivNode(v, :z)
        expr = BinaryOpNode(:*, p, d)
        s = sprint(show, expr)
        @test occursin("U", s)
        @test occursin("psi", s)
        @test occursin("dz", s)
    end

    @testset "Nested trees" begin
        # sigma * Lap(psi) where Lap = dx(dx) + dz(dz)
        psi = VarNode(:psi)
        sigma = EigenvalueNode(:sigma)
        lap_psi = BinaryOpNode(:+,
            DerivNode(DerivNode(psi, :x), :x),
            DerivNode(DerivNode(psi, :z), :z)
        )
        lhs = BinaryOpNode(:*, sigma, lap_psi)

        @test lhs.left == sigma
        @test lhs.right == lap_psi
        @test lap_psi.left isa DerivNode
        @test lap_psi.left.expr isa DerivNode
    end

end
