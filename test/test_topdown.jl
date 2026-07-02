using Test
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using LinearAlgebra

# Structural/regression coverage for topdown.jl's testbases/trialbases. Numerical
# matvec accuracy through the full pipeline (this + couplingmatrices.jl +
# transfermatrices.jl + abstractnca.jl) is covered end-to-end in
# test_petrovgalerkinnca.jl; this file checks that the orchestration itself produces
# well-formed output and, in particular, regression-tests the three bugs found while
# verifying this file: unqualified isleaf calls (crashed for every factorization
# choice), the double conversion of far-field indices for inherited parent pivots, and
# a missing maxrank=maxrank forwarded to compute_test_pivots! (silently used the
# hardcoded default of 40 instead of the caller's maxrank).
@testset "topdown.jl" begin
    testpts, trialpts = randomwellseparatedclouds(150; seed=5)
    kernel = HelmholtzTestKernel(2.0)
    farmatrix = AdaptiveCrossApproximation.AbstractKernelMatrix(kernel, testpts, trialpts)
    tree = testblocktree(testpts, trialpts; minvalues=15)
    testfardata, trialfardata = NestedCrossApproximation.farinteractions(tree)
    n = length(testpts)
    m = length(trialpts)
    # Below the factorization.jl default of 40: a regression guard for the missing
    # maxrank=maxrank bug, which only manifests when the caller's maxrank is smaller
    # than that hardcoded default.
    maxrank = 25

    # iACA+MimicryPivoting is deliberately not exercised here: it hits an unrelated,
    # pre-existing gap in AdaptiveCrossApproximation itself (MimicryPivotingFunctor's
    # deflation step isn't callable with an Int index), reproduced identically for both
    # TopDown and BottomUp - outside this package's scope to fix.
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
