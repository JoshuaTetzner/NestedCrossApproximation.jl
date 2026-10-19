using Test
using NestedCrossApproximation
using AdaptiveCrossApproximation

@testset "bottomup.jl" begin
    @testset "BottomUp" begin
        default = @inferred NestedCrossApproximation.BottomUp()
        @test default isa NestedCrossApproximation.BottomUp{<:ACA,Nothing}
        @test default.factorization isa ACA
        @test default.representor === nothing

        factorization = ACA(; convergence=FNormEstimator(1e-6))
        representor = NestedCrossApproximation.RandomRep()
        compressor = @inferred NestedCrossApproximation.BottomUp(;
            factorization=factorization, representor=representor
        )
        @test compressor isa NestedCrossApproximation.BottomUp{
            typeof(factorization),NestedCrossApproximation.RandomRep
        }
        @test compressor.factorization === factorization
        @test compressor.representor === representor
        @test isconcretetype(typeof(compressor))

        positional = @inferred NestedCrossApproximation.BottomUp(factorization, representor)
        @test typeof(positional) === typeof(compressor)
        @test NestedCrossApproximation.BottomUp(; representor=representor).factorization isa
            ACA
        @test NestedCrossApproximation.BottomUp(; factorization=factorization).representor ===
            nothing

        # Mutable, but the field types are fixed by the constructor.
        compressor.factorization = ACA(; convergence=FNormEstimator(1e-6))
        @test compressor.factorization !== factorization
        @test_throws MethodError compressor.representor = nothing
    end
end
