using Test
using H2Trees
using NestedCrossApproximation
using LinearAlgebra
using OhMyThreads
using Random
using StaticArrays

@testset "transfermatrices.jl" begin
    scalartypes = (Float32, Float64, ComplexF32, ComplexF64)

    @testset "testtransfer ($T)" for T in scalartypes
        Random.seed!(1)
        buf = rand(T, 12, 6)
        original = copy(buf)
        rows = [9, 2, 6]
        childrows = [4, 2, 11, 7]

        transfer = @inferred NestedCrossApproximation.testtransfer(rows, childrows, buf)
        @test transfer isa Matrix{T}
        @test size(transfer) == (length(childrows), length(rows))
        @test buf == original
        @test transfer * buf[rows, 1:3] ≈ buf[childrows, 1:3]

        # Nestedness: child basis times transfer matrix is the parent basis on the child.
        lowrank = rand(T, 12, 3) * rand(T, 3, 6)
        childvalues = [1, 4, 5, 8, 10]
        childpivots = [4, 10, 1]
        childbasis = NestedCrossApproximation.nestedtestbasis(
            childvalues, childpivots, lowrank
        )
        parentbasis = NestedCrossApproximation.nestedtestbasis(collect(1:12), rows, lowrank)
        transfer = NestedCrossApproximation.testtransfer(rows, childpivots, lowrank)
        @test childbasis * transfer ≈ parentbasis[childvalues, :] rtol =
            100 * sqrt(eps(real(T)))
    end

    @testset "trialtransfer ($T)" for T in scalartypes
        Random.seed!(2)
        buf = rand(T, 6, 12)
        original = copy(buf)
        cols = [9, 2, 6]
        childcols = [4, 2, 11, 7]

        transfer = @inferred NestedCrossApproximation.trialtransfer(cols, childcols, buf)
        @test transfer isa Matrix{T}
        @test size(transfer) == (length(cols), length(childcols))
        @test buf == original
        @test buf[1:3, cols] * transfer ≈ buf[1:3, childcols]

        lowrank = rand(T, 6, 3) * rand(T, 3, 12)
        childvalues = [1, 4, 5, 8, 10]
        childpivots = [4, 10, 1]
        childbasis = NestedCrossApproximation.nestedtrialbasis(
            childvalues, childpivots, lowrank
        )
        parentbasis = NestedCrossApproximation.nestedtrialbasis(
            collect(1:12), cols, lowrank
        )
        transfer = NestedCrossApproximation.trialtransfer(cols, childpivots, lowrank)
        @test transfer * childbasis ≈ parentbasis[:, childvalues] rtol =
            100 * sqrt(eps(real(T)))
    end

    @testset "dependent parent pivots ($T)" for T in scalartypes
        Random.seed!(3)
        buf = rand(T, 8, 2)
        buf[7, :] = buf[3, :]
        @test_throws NestedCrossApproximation.SingularCrossMatrixError NestedCrossApproximation.testtransfer(
            [3, 7], [1, 2], buf
        )
        @test_throws NestedCrossApproximation.SingularCrossMatrixError NestedCrossApproximation.trialtransfer(
            [3, 7], [1, 2], permutedims(buf)
        )
    end

    Random.seed!(4)
    points = [SVector{3,Float64}(rand(3)) for _ in 1:200]
    tree = TwoNTree(points; builder=TwoNTreeBuilder(; minhalfsize=1 / 2^6, minvalues=4))
    nnodes = H2Trees.numberofnodes(tree)
    pivots = [first(H2Trees.values(tree, node), 2) for node in 1:nnodes]
    levelnodes = [
        [
            node for
            node in H2Trees.LevelIterator(tree, level) if !H2Trees.isleaf(tree, node)
        ] for level in H2Trees.levels(tree)
    ]
    nodes = reduce(vcat, levelnodes)

    @testset "transfermatrices! and _build_transfer_storage ($T)" for T in scalartypes
        Random.seed!(5)
        testbuf = rand(T, length(points), 2)
        trialbuf = permutedims(testbuf)

        for (transfermatrices!, transferfunction, buf) in (
            (
                NestedCrossApproximation.testtransfermatrices!,
                NestedCrossApproximation.testtransfer,
                testbuf,
            ),
            (
                NestedCrossApproximation.trialtransfermatrices!,
                NestedCrossApproximation.trialtransfer,
                trialbuf,
            ),
        )
            transfer = Vector{Vector{Matrix{T}}}(undef, nnodes)
            serial = Vector{Vector{Matrix{T}}}(undef, nnodes)
            for level in eachindex(levelnodes)
                transfermatrices!(tree, transfer, levelnodes[level], pivots, buf)
                transfermatrices!(
                    tree,
                    serial,
                    levelnodes[level],
                    pivots,
                    buf;
                    scheduler=SerialScheduler(),
                )
            end

            @test [node for node in 1:nnodes if isassigned(transfer, node)] == sort(nodes)
            @test all(
                transfer[node] == [
                    transferfunction(pivots[node], pivots[child], buf) for
                    child in H2Trees.children(tree, node)
                ] for node in nodes
            )
            @test all(transfer[node] == serial[node] for node in nodes)

            store = @inferred NestedCrossApproximation._build_transfer_storage(
                transfer, levelnodes, tree
            )
            plan = store.plan
            @test store isa NestedCrossApproximation.TransferStore{
                T,NestedCrossApproximation.TransferTraversalPlan
            }
            @test plan.levelidcs == nodes
            @test [
                plan.levelidcs[plan.leveltoidcsptr[l]:(plan.leveltoidcsptr[l + 1] - 1)] for
                l in eachindex(levelnodes)
            ] == levelnodes
            @test all(
                plan.chdsidcs[plan.idxtochdsptr[i]:(plan.idxtochdsptr[i + 1] - 1)] ==
                collect(H2Trees.children(tree, node)) for (i, node) in enumerate(nodes)
            )
            @test all(
                store.blocks[plan.idxtochdsptr[i]:(plan.idxtochdsptr[i + 1] - 1)] ==
                transfer[node] for (i, node) in enumerate(nodes)
            )
            @test length(store.blocks) == length(plan.chdsidcs) == nnodes - 1

            # Levels without transfers only shift the level pointer.
            padded = NestedCrossApproximation._build_transfer_storage(
                transfer, vcat([Int[], Int[]], levelnodes), tree
            )
            @test padded.plan.leveltoidcsptr == vcat([1, 1], plan.leveltoidcsptr)
            @test padded.plan.levelidcs == plan.levelidcs
            @test padded.plan.chdsidcs == plan.chdsidcs
            @test padded.blocks == store.blocks

            notransfers = NestedCrossApproximation._build_transfer_storage(
                transfer, [Int[], Int[]], tree
            )
            @test notransfers.plan.leveltoidcsptr == [1, 1, 1]
            @test notransfers.plan.idxtochdsptr == [1]
            @test isempty(notransfers.blocks)
        end
    end

    # Slots 1 and 2 belong to one node and form one group, slot 3 is a node of its own;
    # the second level is a single node with the slots 4 and 5.
    @testset "_build_dirtransfer_storage ($T)" for T in scalartypes
        transfer = Vector{Vector{Matrix{T}}}(undef, 8)
        transferdirs = Vector{Vector{Int}}(undef, 8)
        for (slot, childslots) in
            (1 => [4, 5], 2 => [4, 5], 3 => [5], 4 => [6, 7], 5 => [8])
            transferdirs[slot] = childslots
            transfer[slot] = [rand(T, 2, 3) for _ in childslots]
        end

        store = @inferred NestedCrossApproximation._build_dirtransfer_storage(
            transfer, transferdirs, [[1, 2, 3], [4, 5]], [[2, 1], [2]]
        )
        plan = store.plan
        @test store isa NestedCrossApproximation.TransferStore{
            T,NestedCrossApproximation.DirTransferTraversalPlan
        }
        @test plan.leveltoidcsptr == [1, 4, 6]
        @test plan.levelidcs == [1, 2, 3, 4, 5]
        @test plan.idxtochdsptr == [1, 3, 5, 6, 8, 9]
        @test plan.chdsidcs == [4, 5, 4, 5, 5, 6, 7, 8]
        @test plan.leveltogroupsptr == [1, 3, 4]
        @test plan.groupsptr == [1, 3, 4, 6]
        @test store.blocks == reduce(vcat, transfer[slot] for slot in 1:5)

        # A level without transfers has no groups either.
        emptylevel = NestedCrossApproximation._build_dirtransfer_storage(
            transfer, transferdirs, [Int[], [1, 2]], [Int[], [2]]
        )
        @test emptylevel.plan.leveltoidcsptr == [1, 1, 3]
        @test emptylevel.plan.leveltogroupsptr == [1, 1, 2]
        @test emptylevel.plan.groupsptr == [1, 3]
        @test emptylevel.blocks == [transfer[1]; transfer[2]]
    end
end
