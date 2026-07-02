using Test
using NestedCrossApproximation
using AdaptiveCrossApproximation

@testset "buffer.jl" begin
    testpts, trialpts = randomwellseparatedclouds(40)
    farmatrix = AdaptiveCrossApproximation.AbstractKernelMatrix(
        HelmholtzTestKernel(2.0), testpts, trialpts
    )
    n, m = size(farmatrix)
    maxrank = 7

    @testset "bufferidx alternates by level parity" begin
        @test NestedCrossApproximation.bufferidx(1) == 2
        @test NestedCrossApproximation.bufferidx(2) == 1
        @test NestedCrossApproximation.bufferidx(3) == 2
        @test NestedCrossApproximation.bufferidx(1) != NestedCrossApproximation.bufferidx(2)
    end

    @testset "testbuffer/trialbuffer shapes" begin
        td = NestedCrossApproximation.testbuffer(
            NestedCrossApproximation.TopDown(), farmatrix, maxrank
        )
        @test td isa Tuple{<:AbstractMatrix,<:AbstractMatrix}
        @test size(td[1]) == (n, maxrank)
        @test size(td[2]) == (n, maxrank)

        bu = NestedCrossApproximation.testbuffer(
            NestedCrossApproximation.BottomUp(), farmatrix, maxrank
        )
        @test size(bu) == (n, maxrank)

        tdt = NestedCrossApproximation.trialbuffer(
            NestedCrossApproximation.TopDown(), farmatrix, maxrank
        )
        @test size(tdt[1]) == (maxrank, m)
        @test size(tdt[2]) == (maxrank, m)

        but = NestedCrossApproximation.trialbuffer(
            NestedCrossApproximation.BottomUp(), farmatrix, maxrank
        )
        @test size(but) == (maxrank, m)
    end

    @testset "fartestbuffer/fartrialbuffer shapes depend on factorization type" begin
        @test size(NestedCrossApproximation.fartestbuffer(ACA(), farmatrix, maxrank)) ==
            (maxrank, m)
        @test size(
            NestedCrossApproximation.fartestbuffer(
                iACA(
                    MaximumValue(),
                    MimicryPivoting(testpts, trialpts),
                    FNormExtrapolator(iFNormEstimator(1e-3)),
                ),
                farmatrix,
                maxrank,
            ),
        ) == (maxrank, maxrank)

        @test size(NestedCrossApproximation.fartrialbuffer(ACA(), farmatrix, maxrank)) ==
            (n, maxrank)
        @test size(
            NestedCrossApproximation.fartrialbuffer(
                iACA(
                    MaximumValue(),
                    MimicryPivoting(testpts, trialpts),
                    FNormExtrapolator(iFNormEstimator(1e-3)),
                ),
                farmatrix,
                maxrank,
            ),
        ) == (maxrank, maxrank)
    end
end
