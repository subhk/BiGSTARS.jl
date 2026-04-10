using Test
using BiGSTARS: ExprNode, VarNode, ParamNode, DerivNode, BinaryOpNode, SubstitutionNode,
    Substitution, expand_substitutions, contains_substitution

@testset "Substitution Expansion" begin

    @testset "Simple expansion" begin
        subs = Dict{Symbol, Substitution}()
        subs[:Lap] = Substitution(:Lap, [:A],
            BinaryOpNode(:+,
                DerivNode(DerivNode(VarNode(:A), :x), :x),
                DerivNode(DerivNode(VarNode(:A), :z), :z)
            )
        )

        input = SubstitutionNode(:Lap, [VarNode(:psi)])
        result = expand_substitutions(input, subs)

        expected = BinaryOpNode(:+,
            DerivNode(DerivNode(VarNode(:psi), :x), :x),
            DerivNode(DerivNode(VarNode(:psi), :z), :z)
        )
        @test result == expected
    end

    @testset "Nested expansion" begin
        subs = Dict{Symbol, Substitution}()
        subs[:Lap] = Substitution(:Lap, [:A],
            BinaryOpNode(:+,
                DerivNode(DerivNode(VarNode(:A), :x), :x),
                DerivNode(DerivNode(VarNode(:A), :z), :z)
            )
        )
        subs[:BiLap] = Substitution(:BiLap, [:A],
            SubstitutionNode(:Lap, [SubstitutionNode(:Lap, [VarNode(:A)])])
        )

        input = SubstitutionNode(:BiLap, [VarNode(:psi)])
        result = expand_substitutions(input, subs)

        @test !contains_substitution(result)
    end

    @testset "Expansion inside BinaryOpNode" begin
        subs = Dict{Symbol, Substitution}()
        subs[:Lap] = Substitution(:Lap, [:A],
            BinaryOpNode(:+,
                DerivNode(DerivNode(VarNode(:A), :x), :x),
                DerivNode(DerivNode(VarNode(:A), :z), :z)
            )
        )

        input = BinaryOpNode(:*, ParamNode(:U),
            SubstitutionNode(:Lap, [VarNode(:psi)]))
        result = expand_substitutions(input, subs)

        @test result.left == ParamNode(:U)
        @test result.right isa BinaryOpNode
        @test !contains_substitution(result)
    end

    @testset "Depth limit prevents infinite recursion" begin
        subs = Dict{Symbol, Substitution}()
        subs[:Bad] = Substitution(:Bad, [:A],
            SubstitutionNode(:Bad, [VarNode(:A)])
        )

        input = SubstitutionNode(:Bad, [VarNode(:psi)])
        @test_throws ErrorException expand_substitutions(input, subs; max_depth=10)
    end

end
