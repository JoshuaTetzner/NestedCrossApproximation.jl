using Test
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using LinearAlgebra
using StaticArrays
using Random

# representor.jl reduces a far-field candidate index set down to at most `npivots`
# representative indices before ACA/iACA ever sees it. All three strategies share the
# same call protocol: representor(refidcs, idcs, npivots) -> Vector{Int} (a subset of
# idcs, of length min(npivots, length(idcs))).
@testset "representor.jl" begin
    testpts, trialpts = randomwellseparatedclouds(80; seed=8)
    n = length(testpts)

    @testset "RandomRep" begin
        rep = NestedCrossApproximation.RandomRep()
        idcs = collect(10:40)
        refidcs = collect(1:5)

        chosen = rep(refidcs, idcs, 12)
        @test length(chosen) == 12
        @test allunique(chosen)
        @test issubset(chosen, idcs)

        # npivots above the candidate count: capped at the candidate count, not padded.
        chosen_all = rep(refidcs, idcs, 1000)
        @test sort(chosen_all) == idcs
    end

    @testset "ChebyshevRep" begin
        rep = NestedCrossApproximation.ChebyshevRep(testpts)
        idcs = collect(1:n)
        refidcs = Int[]

        chosen = rep(refidcs, idcs, 10)
        @test length(chosen) == 10
        @test allunique(chosen)
        @test issubset(chosen, idcs)

        chosen_all = rep(refidcs, idcs, 10_000)
        @test sort(chosen_all) == idcs
    end

    @testset "MimicryRep / MimicryRepFunctor" begin
        tree = testblocktree(testpts, trialpts; minvalues=10)
        pivoting = AdaptiveCrossApproximation.TreeMimicryPivoting(
            testpts, trialpts, H2Trees.trialtree(tree)
        )
        rep = NestedCrossApproximation.MimicryRep(pivoting)
        functor = NestedCrossApproximation._stateful_representor(rep, 25)
        @test functor isa NestedCrossApproximation.MimicryRepFunctor

        refidcs = collect(1:10)
        # Far-field candidate set: a couple of trial-tree leaf nodes (node-level, as
        # MimicryRep requires).
        Ft = collect(
            H2Trees.LevelIterator(
                H2Trees.trialtree(tree), maximum(H2Trees.levels(H2Trees.trialtree(tree)))
            ),
        )[1:2]
        navailable = sum(
            length(H2Trees.values(H2Trees.trialtree(tree), node)) for node in Ft
        )

        chosen = functor(refidcs, Ft, min(5, navailable))
        @test length(chosen) == min(5, navailable)
        @test allunique(chosen)
        @test issubset(chosen, H2Trees.values(H2Trees.trialtree(tree), Ft))
    end

    @testset "_stateful_representor" begin
        @test NestedCrossApproximation._stateful_representor(nothing, 25) === nothing
        randomrep = NestedCrossApproximation.RandomRep()
        @test NestedCrossApproximation._stateful_representor(randomrep, 25) === randomrep
        chebyrep = NestedCrossApproximation.ChebyshevRep(testpts)
        @test NestedCrossApproximation._stateful_representor(chebyrep, 25) === chebyrep
    end

    @testset "adapt_farfield_indices reduction" begin
        tree = testblocktree(testpts, trialpts; minvalues=10)
        ttree = H2Trees.trialtree(tree)
        leafnodes = collect(H2Trees.LevelIterator(ttree, maximum(H2Trees.levels(ttree))))[1:2]
        allvalues = H2Trees.values(ttree, leafnodes)
        t = collect(1:8)
        aca = ACA()

        @testset "RandomRep: raw node-level Ft" begin
            rep = NestedCrossApproximation.RandomRep()
            result = NestedCrossApproximation.adapt_farfield_indices(
                aca, rep, ttree, t, leafnodes
            )
            @test length(result) == min(length(t), length(allvalues))
            @test issubset(result, allvalues)
        end

        @testset "RandomRep: PreResolvedIndices" begin
            rep = NestedCrossApproximation.RandomRep()
            pre = NestedCrossApproximation.PreResolvedIndices(allvalues)
            result = NestedCrossApproximation.adapt_farfield_indices(
                aca, rep, ttree, t, pre
            )
            @test length(result) == min(length(t), length(allvalues))
            @test issubset(result, allvalues)
        end

        @testset "no representor: PreResolvedIndices passes through untouched" begin
            pre = NestedCrossApproximation.PreResolvedIndices(allvalues)
            result = NestedCrossApproximation.adapt_farfield_indices(
                aca, nothing, ttree, t, pre
            )
            @test result == allvalues
        end

        @testset "MimicryRepFunctor rejects PreResolvedIndices" begin
            pivoting = AdaptiveCrossApproximation.TreeMimicryPivoting(
                testpts, trialpts, ttree
            )
            functor = NestedCrossApproximation._stateful_representor(
                NestedCrossApproximation.MimicryRep(pivoting), 25
            )
            pre = NestedCrossApproximation.PreResolvedIndices(allvalues)
            @test_throws ArgumentError NestedCrossApproximation.adapt_farfield_indices(
                aca, functor, ttree, t, pre
            )
        end
    end
end

# End-to-end matvec accuracy through the full pipeline, mirroring
# test_petrovgalerkinnca.jl, but exercising each representor strategy. MimicryRep is
# only tested with BottomUp (TopDown isn't supported, see adapt_farfield_indices above).
@testset "representor.jl (end-to-end matvec)" begin
    testpts, trialpts = randomwellseparatedclouds(200; seed=9)
    kernel = HelmholtzTestKernel(2.0)
    tree = testblocktree(testpts, trialpts; minvalues=15)
    A = densereference(kernel, testpts, trialpts)
    n, m = size(A)

    Random.seed!(43)
    x = rand(ComplexF64, m)

    mimicry = NestedCrossApproximation.MimicryRep(
        AdaptiveCrossApproximation.TreeMimicryPivoting(
            testpts, trialpts, H2Trees.trialtree(tree)
        ),
    )

    compressors = [
        "BottomUp + RandomRep" => NestedCrossApproximation.BottomUp(;
            factorization=ACA(), representor=NestedCrossApproximation.RandomRep()
        ),
        "BottomUp + ChebyshevRep" => NestedCrossApproximation.BottomUp(;
            factorization=ACA(),
            representor=NestedCrossApproximation.ChebyshevRep(testpts),
        ),
        "BottomUp + MimicryRep" =>
            NestedCrossApproximation.BottomUp(; factorization=ACA(), representor=mimicry),
        "TopDown + RandomRep" => NestedCrossApproximation.TopDown(;
            factorization=ACA(), representor=NestedCrossApproximation.RandomRep()
        ),
        "TopDown + ChebyshevRep" => NestedCrossApproximation.TopDown(;
            factorization=ACA(),
            representor=NestedCrossApproximation.ChebyshevRep(testpts),
        ),
    ]

    for (name, compressor) in compressors
        @testset "$name" begin
            hmat = NestedCrossApproximation.PetrovGalerkinNCA(
                kernel,
                testpts,
                trialpts,
                tree;
                testcompressor=compressor,
                trialcompressor=compressor,
                maxrank=25,
            )
            y = hmat * x
            @test norm(A * x - y) / norm(A * x) < 1e-2
        end
    end

    @testset "TopDown + MimicryRep throws" begin
        compressor = NestedCrossApproximation.TopDown(;
            factorization=ACA(), representor=mimicry
        )
        @test_throws TaskFailedException NestedCrossApproximation.PetrovGalerkinNCA(
            kernel,
            testpts,
            trialpts,
            tree;
            testcompressor=compressor,
            trialcompressor=compressor,
            maxrank=25,
        )
    end
end
