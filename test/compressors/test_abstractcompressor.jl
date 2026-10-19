using Test
using NestedCrossApproximation

@testset "abstractcompressor.jl" begin
    # Wideband assembly exists for BottomUp only; every other compressor has to say so.
    @testset "_check_wideband_compressor" begin
        @test NestedCrossApproximation._check_wideband_compressor(
            NestedCrossApproximation.BottomUp(), :test
        ) === nothing
        @test NestedCrossApproximation._check_wideband_compressor(
            NestedCrossApproximation.BottomUp(), :trial
        ) === nothing

        for compressor in (
                NestedCrossApproximation.TopDown(),
                NestedCrossApproximation.BatchedBottomUp(;
                    pivotselector=nothing, compressor=nothing
                ),
            ),
            side in (:test, :trial)

            @test_throws ArgumentError NestedCrossApproximation._check_wideband_compressor(
                compressor, side
            )
            message = try
                NestedCrossApproximation._check_wideband_compressor(compressor, side)
            catch err
                err.msg
            end
            @test occursin("only supported with BottomUp", message)
            @test occursin(string(nameof(typeof(compressor))), message)
            @test occursin("$side compressor", message)
        end
    end

    # TopDown and BottomUp compress a plain far field, the directional one only BottomUp.
    @testset "testbases and trialbases" begin
        kernelmatrix = NestedCrossApproximation.AbstractKernelMatrix
        blocktree = NestedCrossApproximation.BlockTree
        topdown = NestedCrossApproximation.TopDown
        bottomup = NestedCrossApproximation.BottomUp
        plain = NestedCrossApproximation.FarData
        directional = NestedCrossApproximation.DirectionalData

        for bases in
            (NestedCrossApproximation.testbases, NestedCrossApproximation.trialbases)
            @test hasmethod(bases, Tuple{topdown,kernelmatrix,blocktree,plain})
            @test hasmethod(bases, Tuple{bottomup,kernelmatrix,blocktree,plain})
            @test hasmethod(bases, Tuple{bottomup,kernelmatrix,blocktree,directional})
            @test !hasmethod(bases, Tuple{topdown,kernelmatrix,blocktree,directional})
        end
    end
end
