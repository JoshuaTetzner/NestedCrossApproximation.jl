using Test
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using LinearAlgebra

@testset "couplingmatrices.jl" begin
    testpts, trialpts = randomwellseparatedclouds(40; seed=4)
    kernel = HelmholtzTestKernel(2.0)
    farmatrix = AdaptiveCrossApproximation.AbstractKernelMatrix(kernel, testpts, trialpts)
    tree = testblocktree(testpts, trialpts; minvalues=10)

    @testset "FarData: matches direct kernel evaluation, skips empty-pivot blocks" begin
        testfardata, trialfardata = NestedCrossApproximation.farinteractions(tree)
        ptr = NestedCrossApproximation.farptr(testfardata)
        nodes = NestedCrossApproximation.farnodes(testfardata)
        @test !isempty(nodes)

        nnodes_t = H2Trees.numberofnodes(H2Trees.testtree(tree))
        nnodes_s = H2Trees.numberofnodes(H2Trees.trialtree(tree))
        # Alternate real/empty pivot sets on the test side so both the populated and the
        # skipped-block paths get exercised in the same run.
        testpivots = [isodd(t) ? [1, 2] : Int[] for t in 1:nnodes_t]
        trialpivots = [[1, 2, 3] for _ in 1:nnodes_s]

        store = NestedCrossApproximation.assemble_couplingstore(
            farmatrix, testpivots, trialpivots, testfardata, trialfardata
        )

        for t in 1:nnodes_t
            for i in ptr[t]:(ptr[t + 1] - 1)
                s = nodes[i]
                expected = zeros(ComplexF64, length(testpivots[t]), length(trialpivots[s]))
                if !isempty(testpivots[t]) && !isempty(trialpivots[s])
                    farmatrix(expected, testpivots[t], trialpivots[s])
                end
                @test store.blocks[i] == expected
            end
        end
    end

    @testset "DirectionalData: matches direct kernel evaluation, skips empty-pivot blocks (regression for the missing isempty guard)" begin
        isnear = NestedCrossApproximation.isnearwideband(2π / 1.5; γ=1.0)
        testdata, trialdata = NestedCrossApproximation.fardata(tree, isnear)
        @test !isempty(NestedCrossApproximation.farnodes(testdata))

        ndirs_t = length(testdata.dirs)
        ndirs_s = length(trialdata.dirs)
        # Same alternating real/empty pattern, but per direction slot (diridx) rather
        # than per node - this is exactly the granularity assemble_couplingstore
        # (DirectionalData) operates at, and exactly where the missing guard could fire.
        testpivots = [isodd(d) ? [1, 2] : Int[] for d in 1:ndirs_t]
        trialpivots = [[1, 2, 3] for _ in 1:ndirs_s]

        store = NestedCrossApproximation.assemble_couplingstore(
            farmatrix, testpivots, trialpivots, testdata, trialdata
        )

        testptr = NestedCrossApproximation.farptr(testdata)
        trialptr = NestedCrossApproximation.farptr(trialdata)
        testnodes = NestedCrossApproximation.farnodes(testdata)
        trialnodes = NestedCrossApproximation.farnodes(trialdata)
        nnodes = length(testptr) - 1

        for t in 1:nnodes
            for (localsidx, sidx) in enumerate(testptr[t]:(testptr[t + 1] - 1))
                s = testnodes[sidx]
                tdiridx = NestedCrossApproximation.diridxfromlocalfaridx(
                    testdata, t, localsidx
                )
                localtidx = findfirst(
                    ==(t), view(trialnodes, trialptr[s]:(trialptr[s + 1] - 1))
                )
                sdiridx = NestedCrossApproximation.diridxfromlocalfaridx(
                    trialdata, s, localtidx
                )

                expected = zeros(
                    ComplexF64, length(testpivots[tdiridx]), length(trialpivots[sdiridx])
                )
                if !isempty(testpivots[tdiridx]) && !isempty(trialpivots[sdiridx])
                    farmatrix(expected, testpivots[tdiridx], trialpivots[sdiridx])
                end
                @test store.blocks[sidx] == expected
            end
        end
    end
end
