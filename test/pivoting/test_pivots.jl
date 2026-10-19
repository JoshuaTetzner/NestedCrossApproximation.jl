using Test
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using LinearAlgebra
using StaticArrays
using Random

@testset "pivots.jl" begin
    scalartypes = (Float32, Float64, ComplexF32, ComplexF64)
    maxrank = 20

    Random.seed!(3)
    testpts = [SVector{3,Float64}(rand(3)) for _ in 1:60]
    trialpts = [SVector{3,Float64}(rand(3)) + SVector(3.0, 0.0, 0.0) for _ in 1:80]
    builder = TwoNTreeBuilder(; minhalfsize=1 / 2^4, minvalues=10)
    ttree = TwoNTree(testpts; builder=builder)
    stree = TwoNTree(trialpts; builder=builder)
    tree = H2Trees.BlockTree(ttree, stree)
    tvalues = H2Trees.values(ttree, H2Trees.root(ttree))
    svalues = H2Trees.values(stree, H2Trees.root(stree))
    tnodes = collect(H2Trees.children(ttree, H2Trees.root(ttree)))
    snodes = collect(H2Trees.children(stree, H2Trees.root(stree)))
    tcluster = H2Trees.values(
        ttree, argmax(node -> length(H2Trees.values(ttree, node)), tnodes)
    )
    scluster = H2Trees.values(
        stree, argmax(node -> length(H2Trees.values(stree, node)), snodes)
    )

    laplace = AdaptiveCrossApproximation.PointMatrix{Float64}(
        (x, y) -> 1 / norm(x - y), testpts, trialpts
    )
    mimicryiaca = IACA(
        MaximumValue(),
        MimicryPivoting(testpts, trialpts),
        FNormExtrapolator(FNormEstimator(1e-3)),
    )
    treemimicryiaca = IACA(
        MaximumValue(),
        AdaptiveCrossApproximation.TreeMimicryPivoting(testpts, trialpts, stree),
        FNormExtrapolator(FNormEstimator(1e-3)),
    )
    aca = NestedCrossApproximation._build_stateful_factorization(ACA(), laplace, maxrank)
    iaca = NestedCrossApproximation._build_stateful_factorization(
        mimicryiaca, laplace, maxrank
    )
    treeiaca = NestedCrossApproximation._build_stateful_factorization(
        treemimicryiaca, laplace, maxrank
    )
    random = NestedCrossApproximation.RandomRep()
    chebyshev = NestedCrossApproximation.ChebyshevRep(trialpts)
    mimicry = NestedCrossApproximation._stateful_representor(
        NestedCrossApproximation.MimicryRep(
            AdaptiveCrossApproximation.TreeMimicryPivoting(testpts, trialpts, stree)
        ),
        maxrank,
    )

    @testset "_build_stateful_factorization" begin
        @test aca isa ACA
        @test NestedCrossApproximation._build_stateful_factorization(
            ACA(; convergence=FNormEstimator(1e-6)), laplace, maxrank
        ) isa ACA
        @test iaca isa IACA
        @test treeiaca isa IACA
    end

    @testset "_stateful_representor" begin
        @test NestedCrossApproximation._stateful_representor(nothing, maxrank) === nothing
        @test NestedCrossApproximation._stateful_representor(random, maxrank) === random
        @test NestedCrossApproximation._stateful_representor(chebyshev, maxrank) ===
            chebyshev
        @test mimicry isa NestedCrossApproximation.MimicryRepFunctor
    end

    @testset "_ranklimit" begin
        @test NestedCrossApproximation._ranklimit(20, 30, 40) == 20
        @test NestedCrossApproximation._ranklimit(20, 7, 40) == 7
        @test NestedCrossApproximation._ranklimit(20, 30, 5) == 5
        @test NestedCrossApproximation._ranklimit(20, 0, 5) == 0
    end

    @testset "_use_tree_mimicry" begin
        @test !NestedCrossApproximation._use_tree_mimicry(iaca)
        @test NestedCrossApproximation._use_tree_mimicry(treeiaca)
    end

    # Node-level far fields count their points only for tree mimicry, which is handed the
    # nodes themselves; everything else has been resolved to points already.
    @testset "_effective_far_count" begin
        resolved = NestedCrossApproximation.PreResolvedIndices(svalues)
        @test NestedCrossApproximation._effective_far_count(
            aca, stree, snodes, nothing, svalues
        ) == length(svalues)
        @test NestedCrossApproximation._effective_far_count(
            iaca, stree, snodes, nothing, svalues
        ) == length(svalues)
        @test NestedCrossApproximation._effective_far_count(
            treeiaca, stree, snodes, nothing, snodes
        ) == length(svalues)
        @test NestedCrossApproximation._effective_far_count(
            treeiaca, stree, resolved, nothing, svalues[1:7]
        ) == 7
        @test NestedCrossApproximation._effective_far_count(
            treeiaca, stree, snodes, random, svalues[1:7]
        ) == 7
    end

    @testset "adapt_farfield_indices" begin
        resolved = NestedCrossApproximation.PreResolvedIndices([3, 1, 4, 1, 5])
        cluster = tvalues[1:12]

        for factorization in (aca, iaca, treeiaca)
            @test NestedCrossApproximation.adapt_farfield_indices(
                factorization, nothing, stree, cluster, resolved
            ) == [3, 1, 4, 1, 5]
        end
        @test NestedCrossApproximation.adapt_farfield_indices(
            aca, nothing, stree, cluster, snodes
        ) == svalues
        @test NestedCrossApproximation.adapt_farfield_indices(
            iaca, nothing, stree, cluster, snodes
        ) == svalues
        @test NestedCrossApproximation.adapt_farfield_indices(
            treeiaca, nothing, stree, cluster, snodes
        ) == snodes

        # A representor cuts the far field down to the size of the cluster.
        for representor in (random, chebyshev, mimicry)
            reduced = NestedCrossApproximation.adapt_farfield_indices(
                aca, representor, stree, cluster, snodes
            )
            @test length(reduced) == length(cluster)
            @test allunique(reduced)
            @test issubset(reduced, svalues)
        end
        for representor in (random, chebyshev)
            reduced = NestedCrossApproximation.adapt_farfield_indices(
                aca,
                representor,
                stree,
                cluster,
                NestedCrossApproximation.PreResolvedIndices(svalues[1:30]),
            )
            @test length(reduced) == length(cluster)
            @test allunique(reduced)
            @test issubset(reduced, svalues[1:30])
        end
        @test sort(
            NestedCrossApproximation.adapt_farfield_indices(
                aca, chebyshev, stree, tvalues, snodes[1:1]
            ),
        ) == sort(H2Trees.values(stree, snodes[1]))
        @test_throws ArgumentError NestedCrossApproximation.adapt_farfield_indices(
            aca,
            mimicry,
            stree,
            cluster,
            NestedCrossApproximation.PreResolvedIndices(svalues),
        )
    end

    # After the search the column buffer holds the block's pivot columns on the cluster
    # rows, which is what `nestedtestbasis` divides by; the row buffer is handed back
    # clean. The trial side mirrors this.
    @testset "compute_test_pivots! ($T)" for T in scalartypes
        kernel = if T <: Complex
            (x, y) -> T(cis(norm(x - y)) / norm(x - y))
        else
            (x, y) -> T(1 / norm(x - y))
        end
        block = [kernel(x, y) for x in testpts, y in trialpts]
        farmatrix = AdaptiveCrossApproximation.PointMatrix{T}(kernel, testpts, trialpts)
        factorization = NestedCrossApproximation._build_stateful_factorization(
            ACA(), farmatrix, maxrank
        )
        cluster = tcluster
        others = setdiff(tvalues, cluster)

        colbuffer = ones(T, 60, maxrank)
        rowbuffer = zeros(T, maxrank, 80)
        rows, cols = NestedCrossApproximation.compute_test_pivots!(
            factorization,
            nothing,
            farmatrix,
            tree,
            cluster,
            snodes,
            colbuffer,
            rowbuffer;
            maxrank=maxrank,
        )
        rank = length(rows)
        @test 3 < rank == length(cols) < min(maxrank, length(cluster))
        @test allunique(rows) && issubset(rows, cluster)
        @test allunique(cols) && issubset(cols, svalues)
        @test colbuffer[cluster, 1:rank] ≈ block[cluster, cols]
        @test all(isone, colbuffer[others, :])
        @test iszero(rowbuffer)
        basis = NestedCrossApproximation.nestedtestbasis(cluster, rows, colbuffer)
        @test basis isa Matrix{T}
        @test basis * block[rows, :] ≈ block[cluster, :] rtol = 1e-3

        resolved = NestedCrossApproximation.compute_test_pivots!(
            factorization,
            nothing,
            farmatrix,
            tree,
            cluster,
            NestedCrossApproximation.PreResolvedIndices(svalues),
            colbuffer,
            rowbuffer;
            maxrank=maxrank,
        )
        @test resolved == (rows, cols)

        capped, _ = NestedCrossApproximation.compute_test_pivots!(
            factorization,
            nothing,
            farmatrix,
            tree,
            cluster,
            snodes,
            colbuffer,
            rowbuffer;
            maxrank=3,
        )
        @test capped == rows[1:3]

        # Buffers with exactly `maxrank` rows, and the search running into that limit.
        tight = zeros(T, 3, 80)
        capped, _ = NestedCrossApproximation.compute_test_pivots!(
            factorization,
            nothing,
            farmatrix,
            tree,
            cluster,
            snodes,
            colbuffer,
            tight;
            maxrank=3,
        )
        @test capped == rows[1:3]
        @test iszero(tight)
    end

    @testset "compute_trial_pivots! ($T)" for T in scalartypes
        kernel = if T <: Complex
            (x, y) -> T(cis(norm(x - y)) / norm(x - y))
        else
            (x, y) -> T(1 / norm(x - y))
        end
        block = [kernel(x, y) for x in testpts, y in trialpts]
        farmatrix = AdaptiveCrossApproximation.PointMatrix{T}(kernel, testpts, trialpts)
        factorization = NestedCrossApproximation._build_stateful_factorization(
            ACA(), farmatrix, maxrank
        )
        cluster = scluster
        others = setdiff(svalues, cluster)

        colbuffer = zeros(T, 60, maxrank)
        rowbuffer = ones(T, maxrank, 80)
        rows, cols = NestedCrossApproximation.compute_trial_pivots!(
            factorization,
            nothing,
            farmatrix,
            tree,
            tnodes,
            cluster,
            colbuffer,
            rowbuffer;
            maxrank=maxrank,
        )
        rank = length(cols)
        @test 3 < rank == length(rows) < min(maxrank, length(cluster))
        @test allunique(rows) && issubset(rows, tvalues)
        @test allunique(cols) && issubset(cols, cluster)
        @test rowbuffer[1:rank, cluster] ≈ block[rows, cluster]
        @test all(isone, rowbuffer[:, others])
        @test iszero(colbuffer)
        basis = NestedCrossApproximation.nestedtrialbasis(cluster, cols, rowbuffer)
        @test basis isa Matrix{T}
        @test block[:, cols] * basis ≈ block[:, cluster] rtol = 1e-3

        resolved = NestedCrossApproximation.compute_trial_pivots!(
            factorization,
            nothing,
            farmatrix,
            tree,
            NestedCrossApproximation.PreResolvedIndices(tvalues),
            cluster,
            colbuffer,
            rowbuffer;
            maxrank=maxrank,
        )
        @test resolved == (rows, cols)

        _, capped = NestedCrossApproximation.compute_trial_pivots!(
            factorization,
            nothing,
            farmatrix,
            tree,
            tnodes,
            cluster,
            colbuffer,
            rowbuffer;
            maxrank=3,
        )
        @test capped == cols[1:3]

        tight = zeros(T, 60, 3)
        _, capped = NestedCrossApproximation.compute_trial_pivots!(
            factorization,
            nothing,
            farmatrix,
            tree,
            tnodes,
            cluster,
            tight,
            rowbuffer;
            maxrank=3,
        )
        @test capped == cols[1:3]
        @test iszero(tight)
    end

    block = [1 / norm(x - y) for x in testpts, y in trialpts]
    cluster = tcluster

    # The far-side pivots can only come from the set the representor picked.
    @testset "compute_test_pivots! ($name)" for (name, representor) in (
        ("RandomRep", random), ("ChebyshevRep", chebyshev), ("MimicryRep", mimicry)
    )
        Random.seed!(7)
        picked = NestedCrossApproximation.adapt_farfield_indices(
            aca, representor, stree, cluster, snodes
        )
        @test length(picked) < length(svalues) / 2

        colbuffer = zeros(60, maxrank)
        rowbuffer = zeros(maxrank, 80)
        Random.seed!(7)
        rows, cols = NestedCrossApproximation.compute_test_pivots!(
            aca,
            representor,
            laplace,
            tree,
            cluster,
            snodes,
            colbuffer,
            rowbuffer;
            maxrank=maxrank,
        )
        @test length(rows) >= 3
        @test issubset(cols, picked)
        @test colbuffer[cluster, 1:length(rows)] ≈ block[cluster, cols]
        @test iszero(rowbuffer)
        basis = NestedCrossApproximation.nestedtestbasis(cluster, rows, colbuffer)
        @test basis * block[rows, :] ≈ block[cluster, :] rtol = 1e-2
    end

    # IACA searches with a `maxrank x maxrank` scratch buffer and leaves the same
    # contract behind.
    @testset "compute_test_pivots! ($name)" for (name, factorization) in (
        ("IACA, MimicryPivoting", iaca), ("IACA, TreeMimicryPivoting", treeiaca)
    )
        colbuffer = zeros(60, maxrank)
        rowbuffer = zeros(maxrank, maxrank)
        rows, cols = NestedCrossApproximation.compute_test_pivots!(
            factorization,
            nothing,
            laplace,
            tree,
            cluster,
            snodes,
            colbuffer,
            rowbuffer;
            maxrank=maxrank,
        )
        @test length(rows) >= 3
        @test allunique(rows) && issubset(rows, cluster)
        @test allunique(cols) && issubset(cols, svalues)
        @test colbuffer[cluster, 1:length(rows)] ≈ block[cluster, cols]
        @test iszero(rowbuffer)
        basis = NestedCrossApproximation.nestedtestbasis(cluster, rows, colbuffer)
        @test basis * block[rows, :] ≈ block[cluster, :] rtol = 1e-2

        tight = zeros(2, 2)
        capped, _ = NestedCrossApproximation.compute_test_pivots!(
            NestedCrossApproximation._build_stateful_factorization(
                name == "IACA, MimicryPivoting" ? mimicryiaca : treemimicryiaca, laplace, 2
            ),
            nothing,
            laplace,
            tree,
            cluster,
            snodes,
            colbuffer,
            tight;
            maxrank=2,
        )
        @test length(capped) == 2
        @test iszero(tight)
    end

    @testset "_right_multiply_rows! ($T)" for T in scalartypes
        Random.seed!(1)
        A = rand(T, 8, 5)
        right = rand(T, 4, 9)
        rows = [2, 5, 7]
        cols = [9, 1, 4]
        untouched = setdiff(1:8, rows)

        B = copy(A)
        @test NestedCrossApproximation._right_multiply_rows!(B, rows, right, cols, 3) === B
        @test B[rows, 1:3] ≈ A[rows, 1:3] * right[1:3, cols]
        @test B[rows, 4:5] == A[rows, 4:5]
        @test B[untouched, :] == A[untouched, :]

        B = copy(A)
        NestedCrossApproximation._right_multiply_rows!(B, rows, right, 3)
        @test B[rows, 1:3] ≈ A[rows, 1:3] * right[1:3, 1:3]
        @test B[rows, 4:5] == A[rows, 4:5]
        @test B[untouched, :] == A[untouched, :]
    end

    @testset "_left_multiply_cols! ($T)" for T in scalartypes
        Random.seed!(2)
        A = rand(T, 5, 8)
        left = rand(T, 9, 4)
        cols = [2, 5, 7]
        rows = [9, 1, 4]
        untouched = setdiff(1:8, cols)

        B = copy(A)
        @test NestedCrossApproximation._left_multiply_cols!(left, rows, B, cols, 3) === B
        @test B[1:3, cols] ≈ left[rows, 1:3] * A[1:3, cols]
        @test B[4:5, cols] == A[4:5, cols]
        @test B[:, untouched] == A[:, untouched]

        B = copy(A)
        NestedCrossApproximation._left_multiply_cols!(left, B, cols, 3)
        @test B[1:3, cols] ≈ left[1:3, 1:3] * A[1:3, cols]
        @test B[4:5, cols] == A[4:5, cols]
        @test B[:, untouched] == A[:, untouched]
    end
end
