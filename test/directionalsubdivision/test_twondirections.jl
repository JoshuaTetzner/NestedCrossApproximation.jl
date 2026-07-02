using Test
using NestedCrossApproximation
using StaticArrays

@testset "DirectionTree structure" begin
    N = 3
    maxlevel = 2
    tree = NestedCrossApproximation.DirectionTree(Val(N), maxlevel, 5)

    @test length(NestedCrossApproximation.rootnodes(tree)) == 2N
    @test NestedCrossApproximation.numberofnodes(tree) == 2N * (4^maxlevel - 1) ÷ 3

    for root in NestedCrossApproximation.rootnodes(tree)
        @test NestedCrossApproximation.parent(tree, root) == 0
        for child in NestedCrossApproximation.children(tree, root)
            @test NestedCrossApproximation.parent(tree, child) == root
            @test NestedCrossApproximation.isleaf(tree, child)
        end
    end
end

@testset "direction descent" begin
    tree = NestedCrossApproximation.DirectionTree(Val(3), 3, 5)

    # Root order is +x,-x,+y,-y,+z,-z; an x-dominant vector must match root 1.
    @test NestedCrossApproximation.direction(SVector(5.0, 0.1, 0.1), tree, 5) == 1

    # direction(...,clusterlevel) walks the same greedy path as the full
    # descent, just stopped earlier: shallower picks must be ancestors.
    vec = SVector(0.4, 0.3, 0.8)
    shallow = NestedCrossApproximation.direction(vec, tree, 5)  # maxdepth 1
    mid = NestedCrossApproximation.direction(vec, tree, 4)      # maxdepth 2
    full = NestedCrossApproximation.direction(vec, tree)        # maxdepth 3 (tree.level)

    @test NestedCrossApproximation.parent(tree, mid) == shallow
    @test NestedCrossApproximation.parent(tree, full) == mid
end
