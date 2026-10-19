using Test
using NestedCrossApproximation
using AdaptiveCrossApproximation
using StaticArrays

@testset "buffer.jl" begin
    scalartypes = (Float32, Float64, ComplexF32, ComplexF64)
    n, m, maxrank = 9, 13, 7
    topdown = NestedCrossApproximation.TopDown()
    bottomup = NestedCrossApproximation.BottomUp()
    points = [SVector(1.0 * i, 0.0, 0.0) for i in 1:3]
    iaca = IACA(
        MaximumValue(),
        MimicryPivoting(points, points),
        FNormExtrapolator(FNormEstimator(1e-3)),
    )

    # TopDown keeps two buffers and alternates between them from level to level.
    @testset "bufferidx" begin
        @test [NestedCrossApproximation.bufferidx(level) for level in 1:6] == [2, 1, 2, 1, 2, 1]
    end

    @testset "testbuffer ($T)" for T in scalartypes
        farmatrix = AdaptiveCrossApproximation.PointMatrix{T}((i, j) -> zero(T), 1:n, 1:m)

        buffers = @inferred NestedCrossApproximation.testbuffer(topdown, farmatrix, maxrank)
        @test buffers isa NTuple{2,Matrix{T}}
        @test all(buffer -> size(buffer) == (n, maxrank), buffers)
        @test all(iszero, buffers)
        @test buffers[1] !== buffers[2]

        buffer = @inferred NestedCrossApproximation.testbuffer(bottomup, farmatrix, maxrank)
        @test buffer isa Matrix{T}
        @test size(buffer) == (n, maxrank)
        @test iszero(buffer)
    end

    @testset "trialbuffer ($T)" for T in scalartypes
        farmatrix = AdaptiveCrossApproximation.PointMatrix{T}((i, j) -> zero(T), 1:n, 1:m)

        buffers = @inferred NestedCrossApproximation.trialbuffer(
            topdown, farmatrix, maxrank
        )
        @test buffers isa NTuple{2,Matrix{T}}
        @test all(buffer -> size(buffer) == (maxrank, m), buffers)
        @test all(iszero, buffers)
        @test buffers[1] !== buffers[2]

        buffer = @inferred NestedCrossApproximation.trialbuffer(
            bottomup, farmatrix, maxrank
        )
        @test buffer isa Matrix{T}
        @test size(buffer) == (maxrank, m)
        @test iszero(buffer)
    end

    # The scratch buffer of the pivot search: ACA holds full far-field rows (columns),
    # IACA only the cross matrix.
    @testset "fartestbuffer ($T)" for T in scalartypes
        farmatrix = AdaptiveCrossApproximation.PointMatrix{T}((i, j) -> zero(T), 1:n, 1:m)

        buffer = @inferred NestedCrossApproximation.fartestbuffer(ACA(), farmatrix, maxrank)
        @test buffer isa Matrix{T}
        @test size(buffer) == (maxrank, m)
        @test iszero(buffer)

        buffer = @inferred NestedCrossApproximation.fartestbuffer(iaca, farmatrix, maxrank)
        @test buffer isa Matrix{T}
        @test size(buffer) == (maxrank, maxrank)
        @test iszero(buffer)
    end

    @testset "fartrialbuffer ($T)" for T in scalartypes
        farmatrix = AdaptiveCrossApproximation.PointMatrix{T}((i, j) -> zero(T), 1:n, 1:m)

        buffer = @inferred NestedCrossApproximation.fartrialbuffer(
            ACA(), farmatrix, maxrank
        )
        @test buffer isa Matrix{T}
        @test size(buffer) == (n, maxrank)
        @test iszero(buffer)

        buffer = @inferred NestedCrossApproximation.fartrialbuffer(iaca, farmatrix, maxrank)
        @test buffer isa Matrix{T}
        @test size(buffer) == (maxrank, maxrank)
        @test iszero(buffer)
    end
end
