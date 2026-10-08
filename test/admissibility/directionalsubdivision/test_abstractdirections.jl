using Test
using H2Trees
using NestedCrossApproximation
using StaticArrays
using Random
using ParallelKMeans

@testset "abstractdirections.jl" begin
    # Ragged per-node input, CSR out. Node 1 holds 3 far nodes split over 2 directions,
    # node 2 none, node 3 two under one direction inherited from 2 parent directions.
    @testset "_finalize_directionaldata" begin
        far = NestedCrossApproximation.FarData([1, 4, 4, 6], [11, 12, 13, 21, 22])
        data = NestedCrossApproximation._finalize_directionaldata(
            far,
            [[5, 7], Int[], [9]],
            [[2, 1], Int[], [2]],
            [Int[], Int[], [1, 2]],
            [0, 0, 2],
            :sentinel,
        )

        @test data.nodetodirsptr == [1, 3, 3, 4]
        @test data.dirs == [5, 7, 9]
        @test data.diridxtofarptr == [1, 3, 4, 6]
        @test data.nodetochdirptr == [1, 1, 1, 3]
        @test data.partochddirmap == [1, 2]

        @test NestedCrossApproximation.fardata(data) === far
        @test NestedCrossApproximation.farptr(data) === NestedCrossApproximation.farptr(far)
        @test NestedCrossApproximation.farnodes(data) ===
            NestedCrossApproximation.farnodes(far)
        @test NestedCrossApproximation.isnear(data) === :sentinel

        @test [NestedCrossApproximation.ndirections(data, n) for n in 1:3] == [2, 0, 1]
        @test [NestedCrossApproximation.dirrange(data, n) for n in 1:3] == [1:2, 3:2, 3:3]
        @test [NestedCrossApproximation.dirs(data, n) for n in 1:3] == [[5, 7], [], [9]]
        @test [NestedCrossApproximation.nfarnodes(data, n) for n in 1:3] == [3, 0, 2]
        @test [NestedCrossApproximation.farrange(data, n) for n in 1:3] == [1:3, 4:3, 4:5]
        @test [NestedCrossApproximation.farnodes(data, n) for n in 1:3] == [[11, 12, 13], [], [21, 22]]
        @test [NestedCrossApproximation.dirmap(data, n) for n in 1:3] == [[], [], [1, 2]]

        # The direction counts have to add up to the far count of the node.
        @test_throws AssertionError NestedCrossApproximation._finalize_directionaldata(
            far,
            [[5, 7], Int[], [9]],
            [[2, 2], Int[], [2]],
            [Int[], Int[], [1, 2]],
            [0, 0, 2],
            :sentinel,
        )
    end

    # One cloud: with two separated clouds no low-frequency node holds a far interaction.
    isnear = NestedCrossApproximation.isnearwideband(2π / 1.5; γ=1.0)
    Random.seed!(3)
    points = [SVector{3,Float64}(rand(3)) for _ in 1:200]
    twonbuilder = TwoNTreeBuilder(; minhalfsize=1 / 2^6, minvalues=2)
    kmeansbuilder = KMeansTreeBuilder(; numberofclusters=2, minvalues=4)
    Random.seed!(11)
    cases = NamedTuple[]
    for (name, ttree, stree) in (
        (
            "TwoNTree",
            TwoNTree(points; builder=twonbuilder),
            TwoNTree(points; builder=twonbuilder),
        ),
        (
            "BoundingBallTree",
            KMeansTree(points; builder=kmeansbuilder),
            KMeansTree(points; builder=kmeansbuilder),
        ),
    )
        tree = H2Trees.BlockTree(ttree, stree)
        testdata, trialdata = NestedCrossApproximation.fardata(tree, isnear)
        push!(cases, (; name, ttree, stree, testdata, trialdata))
    end

    @test NestedCrossApproximation.interaction(2, 3, cases[1].ttree, cases[1].stree) ==
        H2Trees.center(cases[1].ttree, 2) - H2Trees.center(cases[1].stree, 3)

    @testset "fardata ($(c.name))" for c in cases
        nt = H2Trees.numberofnodes(c.ttree)
        ns = H2Trees.numberofnodes(c.stree)
        testpairs = Set(
            (t, s) for t in 1:nt for s in NestedCrossApproximation.farnodes(c.testdata, t)
        )
        trialpairs = Set(
            (t, s) for s in 1:ns for t in NestedCrossApproximation.farnodes(c.trialdata, s)
        )

        @test !isempty(testpairs)
        @test testpairs == trialpairs
        @test NestedCrossApproximation.isnear(c.testdata) === isnear
        alldirs = NestedCrossApproximation.dirs.(Ref(c.testdata), 1:nt)
        @test any(==([0]), alldirs)
        @test any(d -> !isempty(d) && d != [0], alldirs)
    end

    @testset "dirfarnodes ($(c.name))" for c in cases
        for node in 1:H2Trees.numberofnodes(c.ttree)
            split = Int[]
            for dir in NestedCrossApproximation.dirs(c.testdata, node)
                append!(split, NestedCrossApproximation.dirfarnodes(c.testdata, node, dir))
            end
            # Far nodes are stored grouped by direction.
            @test split == collect(NestedCrossApproximation.farnodes(c.testdata, node))
        end

        node = findfirst(
            n -> NestedCrossApproximation.ndirections(c.testdata, n) > 1,
            1:H2Trees.numberofnodes(c.ttree),
        )
        @test_throws ErrorException NestedCrossApproximation.dirfarnodes(
            c.testdata, node, maximum(NestedCrossApproximation.dirs(c.testdata, node)) + 1
        )
    end

    @testset "dirfarfield ($(c.name))" for c in cases
        for node in 1:H2Trees.numberofnodes(c.ttree)
            chain = vcat(node, collect(H2Trees.ParentUpwardsIterator(c.ttree, node)))
            for dir in NestedCrossApproximation.dirs(c.testdata, node)
                own = NestedCrossApproximation.dirfarnodes(c.testdata, node, dir)
                field = NestedCrossApproximation.dirfarfield(c.ttree, c.testdata, node, dir)

                @test field[1:length(own)] == own
                @test allunique(field)
                @test all(
                    f -> any(
                        f in NestedCrossApproximation.farnodes(c.testdata, a) for
                        a in chain
                    ),
                    field,
                )
                dir == 0 || continue

                # The sentinel inherits only while the ancestors stay low frequency.
                lf = Iterators.takewhile(
                    a -> 0 in NestedCrossApproximation.dirs(c.testdata, a), chain
                )
                @test field == reduce(
                    vcat,
                    (NestedCrossApproximation.dirfarnodes(c.testdata, a, 0) for a in lf);
                    init=Int[],
                )
            end
        end

        @test_throws ErrorException NestedCrossApproximation.dirfarfield(
            c.ttree, c.testdata, H2Trees.numberofnodes(c.ttree) + 1, 0
        )
    end

    # The inverse of dirfarnodes.
    @testset "diridxfromlocalfaridx ($(c.name))" for c in cases
        for node in 1:H2Trees.numberofnodes(c.ttree)
            first = Int(NestedCrossApproximation.farptr(c.testdata)[node])
            @test all(1:NestedCrossApproximation.nfarnodes(c.testdata, node)) do i
                diridx = NestedCrossApproximation.diridxfromlocalfaridx(c.testdata, node, i)
                diridx in NestedCrossApproximation.dirrange(c.testdata, node) &&
                    Int(c.testdata.diridxtofarptr[diridx]) <=
                    first + i - 1 <
                    Int(c.testdata.diridxtofarptr[diridx + 1])
            end
        end

        @test_throws AssertionError NestedCrossApproximation.diridxfromlocalfaridx(
            c.testdata, 1, NestedCrossApproximation.nfarnodes(c.testdata, 1) + 1
        )
    end

    @testset "isbasisnode ($(c.name))" for c in cases
        nnodes = H2Trees.numberofnodes(c.ttree)
        leaf = findfirst(
            n ->
                H2Trees.isleaf(c.ttree, n) &&
                !isempty(NestedCrossApproximation.dirs(c.testdata, n)),
            1:nnodes,
        )
        inner = findfirst(
            n ->
                !H2Trees.isleaf(c.ttree, n) &&
                NestedCrossApproximation.dirs(c.testdata, n) == [0],
            1:nnodes,
        )
        @test all(
            NestedCrossApproximation.isbasisnode(c.ttree, c.testdata, leaf, d) for
            d in NestedCrossApproximation.dirs(c.testdata, leaf)
        )
        @test !NestedCrossApproximation.isbasisnode(c.ttree, c.testdata, inner, 0)

        # The parent's local slot has to exist in the dirmap of every child.
        for t in 1:nnodes
            for (local_d, diridx) in
                enumerate(NestedCrossApproximation.dirrange(c.testdata, t))
                dir = Int(c.testdata.dirs[diridx])
                isempty(
                    NestedCrossApproximation.dirfarfield(c.ttree, c.testdata, t, dir)
                ) && continue
                NestedCrossApproximation.isbasisnode(c.ttree, c.testdata, t, dir) &&
                    continue
                for child in H2Trees.children(c.ttree, t)
                    @test NestedCrossApproximation.dirmap(c.testdata, child)[local_d] in
                        1:NestedCrossApproximation.ndirections(c.testdata, child)
                end
            end
        end
    end
end
