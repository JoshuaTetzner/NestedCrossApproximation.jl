using Test
using NestedCrossApproximation
using H2Trees
using ParallelKMeans
using StaticArrays
using Random

@testset "farinteractions.jl" begin
    isnear = NestedCrossApproximation.isnear()

    Random.seed!(1)
    points = [SVector{3,Float64}(rand(3)) for _ in 1:300]
    shifted = [SVector{3,Float64}(rand(3)) + SVector(1.0, 0.0, 0.0) for _ in 1:200]
    distant = [SVector{3,Float64}(rand(3)) + SVector(3.0, 0.0, 0.0) for _ in 1:40]
    kmeansbuilder = KMeansTreeBuilder(; numberofclusters=2, minvalues=10)
    twonbuilder = TwoNTreeBuilder(; minhalfsize=1 / 2^6, minvalues=10)
    singlenode = KMeansTreeBuilder(; numberofclusters=2, minvalues=100)

    # The last two pair distinct trees; in the last one the roots are already far.
    @testset "farinteractions ($name)" for (name, ttree, stree) in (
        (
            "KMeansTree",
            KMeansTree(points; builder=kmeansbuilder),
            KMeansTree(points; builder=kmeansbuilder),
        ),
        (
            "TwoNTree",
            TwoNTree(points; builder=twonbuilder),
            TwoNTree(points; builder=twonbuilder),
        ),
        (
            "asymmetric trees",
            KMeansTree(points[1:200]; builder=kmeansbuilder),
            KMeansTree(
                shifted; builder=KMeansTreeBuilder(; numberofclusters=2, minvalues=25)
            ),
        ),
        (
            "admissible roots",
            KMeansTree(points[1:40]; builder=singlenode),
            KMeansTree(distant; builder=singlenode),
        ),
    )
        tree = H2Trees.BlockTree(ttree, stree)
        nt = H2Trees.numberofnodes(ttree)
        ns = H2Trees.numberofnodes(stree)
        rootsfar = !isnear(ttree, stree, H2Trees.root(ttree), H2Trees.root(stree))
        @test rootsfar == (name == "admissible roots")

        testdata, trialdata = NestedCrossApproximation.farinteractions(tree; isnear=isnear)
        testfars = [NestedCrossApproximation.farnodes(testdata, t) for t in 1:nt]
        trialfars = [NestedCrossApproximation.farnodes(trialdata, s) for s in 1:ns]
        testpairs = [(t, s) for t in 1:nt for s in testfars[t]]
        trialpairs = [(t, s) for s in 1:ns for t in trialfars[s]]

        @test !isempty(testpairs)
        @test allunique(testpairs)
        @test sort(testpairs) == sort(trialpairs)
        @test all(!isnear(ttree, stree, t, s) for (t, s) in testpairs)
        # A far pair's parents are near, otherwise it would be covered one level up.
        @test all(
            isnear(ttree, stree, H2Trees.parent(ttree, t), H2Trees.parent(stree, s)) for
            (t, s) in testpairs if !rootsfar
        )

        # Near and far blocks together cover every matrix entry exactly once.
        covered = zeros(Int, H2Trees.numberofvalues(ttree), H2Trees.numberofvalues(stree))
        values, nearvalues = NestedCrossApproximation.nearinteractions(tree; isnear=isnear)
        for i in eachindex(values)
            covered[values[i], nearvalues[i]] .+= 1
        end
        for (t, s) in testpairs
            covered[H2Trees.values(ttree, t), H2Trees.values(stree, s)] .+= 1
        end
        @test all(==(1), covered)
    end

    # A narrower index type has to give identical answers.
    for I in (Int, Int32)
        data = NestedCrossApproximation.FarData(I[1, 3, 3, 5], I[7, 8, 9, 10])

        @test data isa NestedCrossApproximation.FarData{I}
        @test NestedCrossApproximation.farptr(data) === data.nodetofarnodesptr
        @test NestedCrossApproximation.farnodes(data) === data.farnodes
        @test NestedCrossApproximation.farnodes(data, 1) == [7, 8]
        @test isempty(NestedCrossApproximation.farnodes(data, 2))
        @test NestedCrossApproximation.farnodes(data, 3) == [9, 10]
    end

    # A node's own far nodes followed by those of every ancestor.
    @testset "farfield" begin
        Random.seed!(3)
        pts = [SVector{3,Float64}(rand(3)) for _ in 1:60]
        tree = TwoNTree(pts; builder=TwoNTreeBuilder(; minhalfsize=1 / 2^3, minvalues=4))
        nnodes = H2Trees.numberofnodes(tree)

        leaf = findlast(n -> H2Trees.isleaf(tree, n), 1:nnodes)
        chain = vcat(leaf, collect(H2Trees.ParentUpwardsIterator(tree, leaf)))
        @test length(chain) >= 3

        # One far node per node on the chain, marked so a misplaced entry is visible.
        counts = zeros(Int, nnodes)
        counts[chain] .= 1
        data = NestedCrossApproximation.FarData(
            NestedCrossApproximation.ptrfromcounts(counts), 10 .* sort(chain)
        )

        @test NestedCrossApproximation.farfield(tree, data, leaf) == 10 .* chain
        # A node with no far nodes of its own still sees its ancestors'.
        parent = chain[2]
        counts[leaf] = 0
        nofar = NestedCrossApproximation.FarData(
            NestedCrossApproximation.ptrfromcounts(counts), 10 .* sort(chain[2:end])
        )
        @test isempty(NestedCrossApproximation.farnodes(nofar, leaf))
        @test NestedCrossApproximation.farfield(tree, nofar, leaf) == 10 .* chain[2:end]
        @test NestedCrossApproximation.farfield(tree, nofar, parent) == 10 .* chain[2:end]
    end
end
