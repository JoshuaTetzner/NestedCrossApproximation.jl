using Test
using NestedCrossApproximation
using AdaptiveCrossApproximation

@testset "topdown.jl" begin
    @testset "TopDown" begin
        default = @inferred NestedCrossApproximation.TopDown()
        @test default isa NestedCrossApproximation.TopDown{<:ACA,Nothing}
        @test default.factorization isa ACA
        @test default.representor === nothing

        factorization = ACA(; convergence=FNormEstimator(1e-6))
        representor = NestedCrossApproximation.RandomRep()
        compressor = @inferred NestedCrossApproximation.TopDown(;
            factorization=factorization, representor=representor
        )
        @test compressor isa NestedCrossApproximation.TopDown{
            typeof(factorization),NestedCrossApproximation.RandomRep
        }
        @test compressor.factorization === factorization
        @test compressor.representor === representor
        @test isconcretetype(typeof(compressor))

        positional = @inferred NestedCrossApproximation.TopDown(factorization, representor)
        @test typeof(positional) === typeof(compressor)
        @test NestedCrossApproximation.TopDown(; representor=representor).factorization isa
            ACA
        @test NestedCrossApproximation.TopDown(; factorization=factorization).representor ===
            nothing

        compressor.factorization = ACA(; convergence=FNormEstimator(1e-6))
        @test compressor.factorization !== factorization
        @test_throws MethodError compressor.representor = nothing
    end
end
