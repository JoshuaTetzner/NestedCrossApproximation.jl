using Test
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using StaticArrays
using Random

@testset "batchedbottomup.jl" begin
    Random.seed!(8)
    testpts = [SVector{3,Float64}(rand(3)) for _ in 1:150]
    trialpts = [SVector{3,Float64}(rand(3)) + SVector(3.0, 0.0, 0.0) for _ in 1:200]
    builder = TwoNTreeBuilder(; minhalfsize=1 / 2^5, minvalues=15)
    ttree = TwoNTree(testpts; builder=builder)
    stree = TwoNTree(trialpts; builder=builder)
    pivotselector = TreeMimicryPivoting(testpts, trialpts, stree)
    densecompressor = ACA(; rowpivoting=FullPivoting(), columnpivoting=FullPivoting())

    @testset "BatchedBottomUp" begin
        compressor = @inferred NestedCrossApproximation.BatchedBottomUp(;
            pivotselector=pivotselector, compressor=densecompressor
        )
        @test compressor isa NestedCrossApproximation.BatchedBottomUp{
            typeof(pivotselector),typeof(densecompressor)
        }
        @test compressor.pivotselector === pivotselector
        @test compressor.compressor === densecompressor
        @test isconcretetype(typeof(compressor))

        positional = @inferred NestedCrossApproximation.BatchedBottomUp(
            pivotselector, densecompressor
        )
        @test typeof(positional) === typeof(compressor)

        @test_throws UndefKeywordError NestedCrossApproximation.BatchedBottomUp()
        @test_throws UndefKeywordError NestedCrossApproximation.BatchedBottomUp(;
            pivotselector=pivotselector
        )
        @test_throws UndefKeywordError NestedCrossApproximation.BatchedBottomUp(;
            compressor=densecompressor
        )
    end

    @testset "Sampled residual stopping" begin
        A = zeros(5, 3)
        A[1, 1], A[2, 2], A[3, 3] = 1, 0.009, 0.001
        cfg = ACA(FullPivoting(), FullPivoting(), FNormEstimator(1e-2))
        @test first(cfg(copy(A), 3)) == 1
        @test first(NestedCrossApproximation._compressblock(cfg, copy(A), 3)) == 2
        # Dense compressors without the sampled keyword retain their interface.
        other = (block, ranklimit) -> (ranklimit, [1], [1])
        @test NestedCrossApproximation._compressblock(other, A, 3) == (3, [1], [1])
    end

    # At most `maxrank` points of the far tree, and no more than cluster or far field hold.
    @testset "_selectfarpivots" begin
        cluster = H2Trees.values(ttree, H2Trees.root(ttree))
        farnodes = collect(H2Trees.children(stree, H2Trees.root(stree)))
        farvalues = H2Trees.values(stree, farnodes)

        @testset "Capped subtree counts" begin
            for node in 1:H2Trees.numberofnodes(stree), cap in (0, 1, 7, 30, 300)
                @test NestedCrossApproximation._clustersize(stree, node, cap) ==
                    min(cap, length(H2Trees.values(stree, node)))
            end
            # Several small far clusters must still cap the pivot count at
            # their actual total, rather than assuming maxrank points exist.
            small = [
                n for n in 1:H2Trees.numberofnodes(stree) if
                H2Trees.isleaf(stree, n) && 0 < length(H2Trees.values(stree, n)) <= 7
            ][1:2]
            actual = H2Trees.values(stree, small)
            @test sort(
                NestedCrossApproximation._selectfarpivots(
                    pivotselector, cluster, small, stree, 25
                ),
            ) == sort(actual)
        end

        pivots = @inferred NestedCrossApproximation._selectfarpivots(
            pivotselector, cluster, farnodes, stree, 25
        )
        @test pivots isa Vector{Int}
        @test length(pivots) == 25
        @test allunique(pivots)
        @test issubset(pivots, farvalues)
        @test NestedCrossApproximation._selectfarpivots(
            pivotselector, cluster, farnodes, stree, 25
        ) == pivots

        @test length(
            NestedCrossApproximation._selectfarpivots(
                pivotselector, cluster[1:7], farnodes, stree, 25
            ),
        ) == 7
        leaf = first(n for n in 1:H2Trees.numberofnodes(stree) if H2Trees.isleaf(stree, n))
        leafvalues = H2Trees.values(stree, leaf)
        @test length(leafvalues) < 25
        @test sort(
            NestedCrossApproximation._selectfarpivots(
                pivotselector, cluster, [leaf], stree, 25
            ),
        ) == sort(leafvalues)

        @test isempty(
            NestedCrossApproximation._selectfarpivots(
                pivotselector, Int[], farnodes, stree, 25
            ),
        )
        @test isempty(
            NestedCrossApproximation._selectfarpivots(
                pivotselector, cluster, farnodes, stree, 0
            ),
        )
    end
end
