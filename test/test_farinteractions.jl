using Test
using NestedCrossApproximation
using H2Trees
using ParallelKMeans
using StaticArrays
using Random

# Near (AdaptiveCrossApproximation) and far (farinteractions) must partition the
# full value x value matrix: every pair covered, none covered twice.
function check_near_far_partition(tree::BlockTree, isnear)
    ttree = H2Trees.testtree(tree)
    stree = H2Trees.trialtree(tree)
    n = H2Trees.numberofvalues(ttree)
    covered = zeros(Bool, n, n)

    values, nearvalues = NestedCrossApproximation.nearinteractions(tree; isnear=isnear)
    for i in eachindex(values)
        @test !any(covered[values[i], nearvalues[i]])
        covered[values[i], nearvalues[i]] .= true
    end

    testfardata, _ = NestedCrossApproximation.farinteractions(tree; isnear=isnear)
    for t in 1:H2Trees.numberofnodes(ttree)
        tvals = H2Trees.values(ttree, t)
        for s in NestedCrossApproximation.farnodes(testfardata, t)
            svals = H2Trees.values(stree, s)
            @test !isnear(ttree, stree, t, s)
            @test !any(covered[tvals, svals])
            covered[tvals, svals] .= true
        end
    end

    @test all(covered)
end

@testset "near/far partition" begin
    Random.seed!(1)
    points = [@SVector rand(3) for _ in 1:300]
    isnear = NestedCrossApproximation.isnear()

    @testset "KMeansTree" begin
        ttree = KMeansTree(points, 2; minvalues=10)
        stree = KMeansTree(points, 2; minvalues=10)
        check_near_far_partition(BlockTree(ttree, stree), isnear)
    end

    @testset "TwoNTree" begin
        tree = TwoNTree(points, points, 1 / 2^6; testminvalues=10, trialminvalues=10)
        check_near_far_partition(tree, isnear)
    end
end
