using Test
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using LinearAlgebra

@testset "factorization.jl" begin
    testpts, trialpts = randomwellseparatedclouds(50; seed=2)
    kernel = HelmholtzTestKernel(2.0)
    farmatrix = AdaptiveCrossApproximation.AbstractKernelMatrix(kernel, testpts, trialpts)
    A = densereference(kernel, testpts, trialpts)
    n, m = size(farmatrix)
    maxrank = 25

    # H2Trees.values(tree, [root]) always recursively collects every point under that
    # node regardless of how many levels the tree actually has, so using the root node
    # as a stand-in for "every point on that side" is robust to the tree's actual shape.
    tree = testblocktree(testpts, trialpts; minvalues=max(n, m))
    sroot = H2Trees.root(H2Trees.trialtree(tree))
    troot = H2Trees.root(H2Trees.testtree(tree))

    @testset "_build_stateful_factorization picks a working ACA/iACA init (regression for the broken default-ACA case)" begin
        # ACA()'s 2-arg callable only resolves for this exact default pivoting/convergence
        # combo and skips matrix-aware sizing; a custom ACA doesn't even have a 2-arg
        # method. _build_stateful_factorization must route both through the 4-arg form.
        for aca in (ACA(), ACA(; convergence=FNormEstimator(1e-6)))
            built = NestedCrossApproximation._build_stateful_factorization(
                aca, farmatrix, maxrank
            )
            @test built isa ACA
        end

        iaca = iACA(
            MaximumValue(),
            MimicryPivoting(testpts, trialpts),
            FNormExtrapolator(iFNormEstimator(1e-3)),
        )
        built = NestedCrossApproximation._build_stateful_factorization(
            iaca, farmatrix, maxrank
        )
        @test built isa iACA
    end

    # compute_test_pivots!/compute_trial_pivots! don't leave colbuffer/rowbuffer holding
    # plain U/V rank factors - after _finalize_test_buffers!/_finalize_trial_buffers!,
    # colbuffer holds A sampled at the pivot *columns* (resp. rowbuffer at the pivot
    # *rows*), which is what nestedbasis.jl's rdiv!/ldiv! step consumes. So the right
    # invariant to check here is that post-finalize buffer slice, not a direct
    # colbuffer*rowbuffer product (full-reconstruction is covered in
    # test_nestedbasis_transfermatrices.jl, the actual consumer of this contract).
    @testset "compute_test_pivots! selects good pivots (ACA, node-level Ft)" begin
        colbuffer = zeros(ComplexF64, n, maxrank)
        rowbuffer = zeros(ComplexF64, maxrank, m)
        factorization = NestedCrossApproximation._build_stateful_factorization(
            ACA(), farmatrix, maxrank
        )
        rows, cols = NestedCrossApproximation.compute_test_pivots!(
            factorization,
            nothing,
            farmatrix,
            tree,
            collect(1:n),
            [sroot],
            colbuffer,
            rowbuffer;
            maxrank=maxrank,
        )
        @test !isempty(rows)
        @test !isempty(cols)
        r = length(rows)
        @test norm(colbuffer[:, 1:r] - A[:, cols]) / norm(A[:, cols]) < 1e-3
    end

    @testset "compute_trial_pivots! selects good pivots (ACA, node-level Fs)" begin
        colbuffer = zeros(ComplexF64, n, maxrank)
        rowbuffer = zeros(ComplexF64, maxrank, m)
        factorization = NestedCrossApproximation._build_stateful_factorization(
            ACA(), farmatrix, maxrank
        )
        rows, cols = NestedCrossApproximation.compute_trial_pivots!(
            factorization,
            nothing,
            farmatrix,
            tree,
            [troot],
            collect(1:m),
            colbuffer,
            rowbuffer;
            maxrank=maxrank,
        )
        @test !isempty(rows)
        @test !isempty(cols)
        r = length(cols)
        @test norm(rowbuffer[1:r, :] - A[rows, :]) / norm(A[rows, :]) < 1e-3
    end

    @testset "PreResolvedIndices (TopDown's already-point-level Ft) matches the node-level path" begin
        # This is the exact mechanism that fixes TopDown's double-conversion bug: Ft
        # wrapped in PreResolvedIndices must be used as-is (point-level), not
        # re-expanded via H2Trees.values as if it were still node-level.
        colbuffer1 = zeros(ComplexF64, n, maxrank)
        rowbuffer1 = zeros(ComplexF64, maxrank, m)
        f1 = NestedCrossApproximation._build_stateful_factorization(
            ACA(), farmatrix, maxrank
        )
        rows1, cols1 = NestedCrossApproximation.compute_test_pivots!(
            f1,
            nothing,
            farmatrix,
            tree,
            collect(1:n),
            [sroot],
            colbuffer1,
            rowbuffer1;
            maxrank=maxrank,
        )

        colbuffer2 = zeros(ComplexF64, n, maxrank)
        rowbuffer2 = zeros(ComplexF64, maxrank, m)
        f2 = NestedCrossApproximation._build_stateful_factorization(
            ACA(), farmatrix, maxrank
        )
        rows2, cols2 = NestedCrossApproximation.compute_test_pivots!(
            f2,
            nothing,
            farmatrix,
            tree,
            collect(1:n),
            NestedCrossApproximation.PreResolvedIndices(collect(1:m)),
            colbuffer2,
            rowbuffer2;
            maxrank=maxrank,
        )

        @test rows1 == rows2
        @test cols1 == cols2
    end

    @testset "adapt_farfield_indices: PreResolvedIndices short-circuits regardless of factorization" begin
        for factorization in (
            ACA(),
            iACA(
                MaximumValue(),
                MimicryPivoting(testpts, trialpts),
                FNormExtrapolator(iFNormEstimator(1e-3)),
            ),
        )
            idcs = NestedCrossApproximation.PreResolvedIndices([3, 1, 4, 1, 5])
            resolved = NestedCrossApproximation.adapt_farfield_indices(
                factorization, nothing, H2Trees.trialtree(tree), collect(1:n), idcs
            )
            @test resolved == [3, 1, 4, 1, 5]
        end
    end
end
