using Test
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using LinearAlgebra

# Structural coverage for topdown.jl's testbases/trialbases: the orchestration itself
# must produce well-formed output. Numerical matvec accuracy through the full pipeline
# is covered end-to-end in test_petrovgalerkinnca.jl.
@testset "topdown.jl" begin
    testpts, trialpts = randomwellseparatedclouds(150; seed=5)
    kernel = HelmholtzTestKernel(2.0)
    farmatrix = AdaptiveCrossApproximation.AbstractKernelMatrix(kernel, testpts, trialpts)
    tree = testblocktree(testpts, trialpts; minvalues=15)
    testfardata, trialfardata = NestedCrossApproximation.farinteractions(tree)
    n = length(testpts)
    m = length(trialpts)
    # Below the factorization.jl default of 40, so that a maxrank that is not forwarded
    # to compute_test_pivots! shows up as a rank above this bound.
    maxrank = 25

    # IACA+MimicryPivoting is not exercised here: MimicryPivotingFunctor's deflation
    # step isn't callable with an Int index, a gap in AdaptiveCrossApproximation that
    # affects TopDown and BottomUp alike.
    compressors = [
        "ACA (default pivoting/convergence)" =>
            NestedCrossApproximation.TopDown(; factorization=ACA()),
        "ACA (custom convergence)" => NestedCrossApproximation.TopDown(;
            factorization=ACA(; convergence=FNormEstimator(1e-6))
        ),
    ]

    for (name, compressor) in compressors
        @testset "testbases: $name" begin
            basis_store, transfer_store, tpivots = NestedCrossApproximation.testbases(
                compressor, farmatrix, tree, testfardata; maxrank=maxrank
            )
            @test !isempty(basis_store.blocks)
            for (i, node) in enumerate(basis_store.plan.nodes)
                @test H2Trees.isleaf(H2Trees.testtree(tree), node)
                @test isassigned(tpivots, node)
                @test size(basis_store.blocks[i], 2) == length(tpivots[node])
                @test size(basis_store.blocks[i], 1) ==
                    length(H2Trees.values(H2Trees.testtree(tree), node))
                @test all(p -> 1 <= p <= n, tpivots[node])
                @test length(tpivots[node]) <= maxrank
            end
        end

        @testset "trialbases: $name" begin
            basis_store, transfer_store, spivots = NestedCrossApproximation.trialbases(
                compressor, farmatrix, tree, trialfardata; maxrank=maxrank
            )
            @test !isempty(basis_store.blocks)
            for (i, node) in enumerate(basis_store.plan.nodes)
                @test H2Trees.isleaf(H2Trees.trialtree(tree), node)
                @test isassigned(spivots, node)
                @test size(basis_store.blocks[i], 1) == length(spivots[node])
                @test size(basis_store.blocks[i], 2) ==
                    length(H2Trees.values(H2Trees.trialtree(tree), node))
                @test all(p -> 1 <= p <= m, spivots[node])
                @test length(spivots[node]) <= maxrank
            end
        end
    end
end
