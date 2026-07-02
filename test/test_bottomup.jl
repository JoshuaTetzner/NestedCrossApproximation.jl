using Test
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using LinearAlgebra

# Structural/regression coverage for bottomup.jl's testbases/trialbases (FarData path).
# Numerical matvec accuracy through the full pipeline is covered end-to-end in
# test_petrovgalerkinnca.jl. ACA() is the default factorization for BottomUp, and the
# combination used to be explicitly disabled (a commented-out guard in the BottomUp
# constructor) because of the _stateful_factorization bug fixed alongside this test -
# the "ACA (default)" case below is the regression guard for both of those.
@testset "bottomup.jl" begin
    testpts, trialpts = randomwellseparatedclouds(150; seed=6)
    kernel = HelmholtzTestKernel(2.0)
    farmatrix = AdaptiveCrossApproximation.AbstractKernelMatrix(kernel, testpts, trialpts)
    tree = testblocktree(testpts, trialpts; minvalues=15)
    testfardata, trialfardata = NestedCrossApproximation.farinteractions(tree)
    n = length(testpts)
    m = length(trialpts)
    maxrank = 25

    compressors = [
        "ACA (default pivoting/convergence)" =>
            NestedCrossApproximation.BottomUp(; factorization=ACA()),
        "ACA (custom convergence)" => NestedCrossApproximation.BottomUp(;
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
                @test all(p -> 1 <= p <= m, spivots[node])
                @test length(spivots[node]) <= maxrank
            end
        end
    end
end
