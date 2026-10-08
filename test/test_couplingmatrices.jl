using Test
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using Random
using StaticArrays

@testset "couplingmatrices.jl" begin
    scalartypes = (Float32, Float64, ComplexF32, ComplexF64)

    # Three test nodes against four trial nodes; trial node 4 has no pivots.
    @testset "FarData ($T, $I)" for T in scalartypes, I in (Int, Int32)
        Random.seed!(1)
        M = rand(T, 9, 7)
        farmatrix = AdaptiveCrossApproximation.PointMatrix{T}((i, j) -> M[i, j], 1:9, 1:7)
        testfardata = NestedCrossApproximation.FarData(I[1, 3, 3, 5], I[2, 4, 1, 4])
        trialfardata = NestedCrossApproximation.FarData(I[1, 2, 3, 3, 5], I[3, 1, 1, 3])
        testpivots = [[1, 4], [2], [5, 9, 3]]
        trialpivots = [[7, 1], [3], [2, 6], Int[]]

        store = @inferred NestedCrossApproximation.assemble_couplingstore(
            farmatrix, testpivots, trialpivots, testfardata, trialfardata
        )
        @test store isa NestedCrossApproximation.CouplingStore{
            T,NestedCrossApproximation.CouplingTraversalPlan
        }
        @test store.plan.nodetofarnodesptr == [1, 3, 3, 5]
        @test store.plan.farnodes == [2, 4, 1, 4]
        @test store.blocks ==
            [M[[1, 4], [3]], zeros(T, 2, 0), M[[5, 9, 3], [7, 1]], zeros(T, 3, 0)]

        nofars = NestedCrossApproximation.FarData(I[1, 1, 1, 1], I[])
        empty = NestedCrossApproximation.assemble_couplingstore(
            farmatrix, testpivots, trialpivots, nofars, nofars
        )
        @test empty.blocks isa Vector{Matrix{T}}
        @test isempty(empty.blocks)
        @test empty.plan.nodetofarnodesptr == [1, 1, 1, 1]
    end

    Random.seed!(3)
    points = [SVector{3,Float64}(rand(3)) for _ in 1:200]
    tree = TwoNTree(points; builder=TwoNTreeBuilder(; minhalfsize=1 / 2^6, minvalues=2))
    nnodes = H2Trees.numberofnodes(tree)
    data, trialdata = NestedCrossApproximation.fardata(
        H2Trees.BlockTree(tree, tree), NestedCrossApproximation.isnearwideband(2π / 1.5)
    )
    farnodes = NestedCrossApproximation.farnodes(data)
    trialfarnodes = NestedCrossApproximation.farnodes(trialdata)
    # The far nodes of a node are stored grouped by direction slot.
    testslots = zeros(Int, length(farnodes))
    trialslots = zeros(Int, length(farnodes))
    for t in 1:nnodes, d in NestedCrossApproximation.dirrange(data, t)
        for i in data.diridxtofarptr[d]:(data.diridxtofarptr[d + 1] - 1)
            s = farnodes[i]
            testslots[i] = d
            trialslots[i] = only(
                sd for sd in NestedCrossApproximation.dirrange(trialdata, s) if t in view(
                    trialfarnodes,
                    trialdata.diridxtofarptr[sd]:(trialdata.diridxtofarptr[sd + 1] - 1),
                )
            )
        end
    end

    # Pivots are per direction slot; every third test slot has none.
    @testset "DirectionalData ($T)" for T in scalartypes
        Random.seed!(2)
        M = rand(T, 200, 200)
        farmatrix = AdaptiveCrossApproximation.PointMatrix{T}(
            (i, j) -> M[i, j], 1:200, 1:200
        )
        testpivots = [rand(1:200, d % 3 == 0 ? 0 : 3) for d in eachindex(data.dirs)]
        trialpivots = [rand(1:200, 2) for _ in eachindex(trialdata.dirs)]

        store = @inferred NestedCrossApproximation.assemble_couplingstore(
            farmatrix, testpivots, trialpivots, data, trialdata
        )
        @test store isa NestedCrossApproximation.CouplingStore{
            T,NestedCrossApproximation.DirCouplingTraversalPlan
        }
        @test length(unique(testslots)) > 100
        @test any(d -> isempty(testpivots[d]), testslots)
        @test store.plan.nodetodiridxptr == NestedCrossApproximation.farptr(data)
        @test store.plan.tidcs == testslots
        @test store.plan.sidcs == trialslots
        @test store.blocks == [
            M[testpivots[testslots[i]], trialpivots[trialslots[i]]] for
            i in eachindex(farnodes)
        ]
    end
end
