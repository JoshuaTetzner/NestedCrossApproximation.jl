using Test
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using LinearAlgebra

# Validates the actual consumer-facing contract of compute_test_pivots!/
# compute_trial_pivots! + nestedtestbasis/nestedtrialbasis together: the nested basis
# built from a small set of pivot rows (resp. columns) must reconstruct the *entire*
# far block to the compressor's tolerance, not just the sampled pivot columns/rows
# (that narrower invariant is what test_factorization.jl checks in isolation).
@testset "nestedbasis.jl + transfermatrices.jl" begin
    testpts, trialpts = randomwellseparatedclouds(60; seed=3)
    kernel = HelmholtzTestKernel(2.0)
    farmatrix = AdaptiveCrossApproximation.AbstractKernelMatrix(kernel, testpts, trialpts)
    A = densereference(kernel, testpts, trialpts)
    n, m = size(farmatrix)
    maxrank = 25

    tree = testblocktree(testpts, trialpts; minvalues=max(n, m))
    sroot = H2Trees.root(H2Trees.trialtree(tree))
    troot = H2Trees.root(H2Trees.testtree(tree))

    @testset "nestedtestbasis reconstructs the full far block from pivot rows" begin
        colbuffer = zeros(ComplexF64, n, maxrank)
        rowbuffer = zeros(ComplexF64, maxrank, m)
        factorization = NestedCrossApproximation._build_stateful_factorization(
            ACA(), farmatrix, maxrank
        )
        tvalues = collect(1:n)
        pivots, _ = NestedCrossApproximation.compute_test_pivots!(
            factorization,
            nothing,
            farmatrix,
            tree,
            tvalues,
            [sroot],
            colbuffer,
            rowbuffer;
            maxrank=maxrank,
        )
        basis = NestedCrossApproximation.nestedtestbasis(tvalues, pivots, colbuffer)
        @test size(basis) == (n, length(pivots))
        @test norm(basis * A[pivots, :] - A) / norm(A) < 1e-3
    end

    @testset "nestedtrialbasis reconstructs the full far block from pivot columns" begin
        colbuffer = zeros(ComplexF64, n, maxrank)
        rowbuffer = zeros(ComplexF64, maxrank, m)
        factorization = NestedCrossApproximation._build_stateful_factorization(
            ACA(), farmatrix, maxrank
        )
        svalues = collect(1:m)
        _, pivots = NestedCrossApproximation.compute_trial_pivots!(
            factorization,
            nothing,
            farmatrix,
            tree,
            [troot],
            svalues,
            colbuffer,
            rowbuffer;
            maxrank=maxrank,
        )
        basis = NestedCrossApproximation.nestedtrialbasis(svalues, pivots, rowbuffer)
        @test size(basis) == (length(pivots), m)
        @test norm(A[:, pivots] * basis - A) / norm(A) < 1e-3
    end

    @testset "testtransfer/trialtransfer compose parent/child bases correctly" begin
        # A child's nested basis at its own (finer) pivot rows must be recoverable, up to
        # compression tolerance, by applying the transfer matrix to the parent's basis
        # restricted to the child's pivot rows - this is exactly the algebraic identity
        # bottomup.jl/topdown.jl rely on to avoid ever forming the full block.
        half = n ÷ 2
        childrows = collect(1:half)
        parentrows = collect(1:n)

        colbuffer = zeros(ComplexF64, n, maxrank)
        rowbuffer = zeros(ComplexF64, maxrank, m)
        fchild = NestedCrossApproximation._build_stateful_factorization(
            ACA(), farmatrix, maxrank
        )
        childpivots, _ = NestedCrossApproximation.compute_test_pivots!(
            fchild,
            nothing,
            farmatrix,
            tree,
            childrows,
            [sroot],
            colbuffer,
            rowbuffer;
            maxrank=maxrank,
        )
        childbasis = NestedCrossApproximation.nestedtestbasis(
            childrows, childpivots, colbuffer
        )

        colbuffer2 = zeros(ComplexF64, n, maxrank)
        rowbuffer2 = zeros(ComplexF64, maxrank, m)
        fparent = NestedCrossApproximation._build_stateful_factorization(
            ACA(), farmatrix, maxrank
        )
        parentpivots, _ = NestedCrossApproximation.compute_test_pivots!(
            fparent,
            nothing,
            farmatrix,
            tree,
            parentrows,
            [sroot],
            colbuffer2,
            rowbuffer2;
            maxrank=maxrank,
        )

        transfer = NestedCrossApproximation.testtransfer(
            parentpivots, childpivots, colbuffer2
        )
        @test size(transfer) == (length(childpivots), length(parentpivots))

        # childbasis * A[childpivots,:] ≈ A[childrows,:], and transfer * A[parentpivots,:]
        # ≈ A[childpivots,:] (the nested-basis identity), so chaining the two should
        # reconstruct the same child block directly from the parent's pivot rows.
        approx = transfer * A[parentpivots, :]
        @test norm(approx - A[childpivots, :]) / norm(A[childpivots, :]) < 1e-2
    end
end
