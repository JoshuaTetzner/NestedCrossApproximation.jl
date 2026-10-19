using Test
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using LinearAlgebra
using StaticArrays
using Random

@testset "representor.jl" begin
    Random.seed!(8)
    points = [SVector{3,Float64}(rand(3)) for _ in 1:1000]
    ptree = TwoNTree(points; builder=TwoNTreeBuilder(; minhalfsize=1 / 2^6, minvalues=20))
    nnodes = H2Trees.numberofnodes(ptree)

    @testset "RandomRep" begin
        rep = NestedCrossApproximation.RandomRep()
        idcs = collect(10:40)

        chosen = rep(Int[], idcs, 12)
        @test length(chosen) == 12
        @test allunique(chosen)
        @test issubset(chosen, idcs)
        @test idcs == 10:40
        @test rep(Int[], idcs, 12) != chosen

        @test sort(rep(Int[], idcs, 1000)) == idcs
        @test isempty(rep(Int[], Int[], 5))
        @test_throws ArgumentError rep(Int[], idcs, 0)
    end

    # A cloud in a rotated box with half extents 4, 2 and 1 whose centroid is not the centre.
    @testset "covmat" begin
        Random.seed!(1)
        Q = Matrix(qr(randn(3, 3)).Q)
        halfextents = SVector(4.0, 2.0, 1.0)
        shift = SVector(1.0, -2.0, 3.0)
        cloud = [
            SVector{3}(Q * (halfextents .* (2 .* rand(3) .^ 2 .- 1))) + shift for
            _ in 1:2000
        ]

        axes, hs, centroid = NestedCrossApproximation.covmat(cloud)
        @test centroid ≈ sum(cloud) / length(cloud)
        @test axes' * axes ≈ I
        @test abs.(axes' * Q) ≈ I atol = 0.1
        @test all(
            maximum(abs(dot(p - centroid, axes[:, a])) for p in cloud) ≈ hs[a] for a in 1:3
        )
        # The box is covered on both sides of the centroid, not only the long one.
        @test all(all(abs.(axes' * (p - centroid)) .<= hs .* (1 + 1e-12)) for p in cloud)
        @test all(hs .> halfextents)
    end

    @testset "_chebyshevnodes1d ($F)" for F in (Float32, Float64)
        for N in (1, 2, 5, 8)
            nodes = NestedCrossApproximation._chebyshevnodes1d(N, F)
            @test nodes isa Vector{F}
            @test length(nodes) == N
            @test issorted(nodes; rev=true)
            @test all(x -> abs(cos(N * acos(x))) < 100 * eps(F), nodes)
        end
    end

    @testset "_chebyshevgrid3d" begin
        for (npivots, gridsize) in ((0, 1), (1, 1), (2, 8), (8, 8), (9, 27), (28, 64))
            grid = NestedCrossApproximation._chebyshevgrid3d(npivots, Float64)
            @test length(grid) == gridsize
            @test allunique(grid)
            @test all(node -> all(abs.(node) .< 1), grid)
        end
        @test NestedCrossApproximation._chebyshevgrid3d(8, Float32) isa
            Vector{SVector{3,Float32}}
    end

    @testset "ChebyshevRep" begin
        rep = NestedCrossApproximation.ChebyshevRep(points)
        idcs = collect(101:400)

        chosen = rep(Int[], idcs, 27)
        @test length(chosen) == 27
        @test allunique(chosen)
        @test issubset(chosen, idcs)
        @test rep([1, 2, 3], idcs, 27) == chosen
        @test length(rep(Int[], idcs, 20)) == 20

        # Spread over the whole candidate cloud: no candidate is far from a chosen point.
        coverage = maximum(
            minimum(norm(points[i] - points[c]) for c in chosen) for i in idcs
        )
        firsts = maximum(
            minimum(norm(points[i] - points[c]) for c in idcs[1:27]) for i in idcs
        )
        @test coverage < 0.45
        @test coverage < firsts

        @test rep(Int[], idcs, 1000) == idcs
        @test_throws ArgumentError rep(Int[], idcs, 0)
        planar = NestedCrossApproximation.ChebyshevRep([
            SVector(rand(), rand()) for _ in 1:10
        ])
        @test_throws AssertionError planar(Int[], collect(1:10), 3)
    end

    @testset "MimicryRep" begin
        rep = NestedCrossApproximation.MimicryRep(
            AdaptiveCrossApproximation.TreeMimicryPivoting(points, points, ptree)
        )
        functor = NestedCrossApproximation._stateful_representor(rep, 25)
        @test functor isa NestedCrossApproximation.MimicryRepFunctor

        level = collect(H2Trees.LevelIterator(ptree, 3))
        reference = H2Trees.values(ptree, level[1])
        for farnodes in (level[(end - 2):end], level[(end - 5):(end - 3)])
            candidates = H2Trees.values(ptree, farnodes)
            chosen = functor(reference, farnodes, 10)
            @test length(chosen) == 10
            @test allunique(chosen)
            @test issubset(chosen, candidates)
            @test functor(reference, farnodes, 10) == chosen
        end
        # Never more than the maximum rank the functor was built for.
        @test length(functor(reference, level[(end - 2):end], 1000)) == 25
        @test_throws ArgumentError functor(reference, level[(end - 2):end], 0)
    end

    @testset "_clustersize" begin
        @test all(
            NestedCrossApproximation._clustersize(ptree, node) ==
            length(H2Trees.values(ptree, node)) for node in 1:nnodes
        )
        @test NestedCrossApproximation._clustersize(ptree, H2Trees.root(ptree)) == 1000
    end

    # Through the assembly: fewer kernel evaluations, and the matvec stays accurate.
    tree = H2Trees.BlockTree(ptree, ptree)
    kernel = HelmholtzTestKernel(2.0)
    dense = densereference(kernel, points, points)
    Random.seed!(43)
    x = rand(ComplexF64, length(points))
    nevals = Threads.Atomic{Int}(0)
    countingmatrix = AdaptiveCrossApproximation.PointMatrix{ComplexF64}(
        (p, q) -> (Threads.atomic_add!(nevals, 1); kernel(p, q)), points, points
    )
    testfardata, trialfardata = NestedCrossApproximation.farinteractions(tree)
    mimicry = NestedCrossApproximation.MimicryRep(
        AdaptiveCrossApproximation.TreeMimicryPivoting(points, points, ptree)
    )

    @testset "end to end ($compressor, $name)" for compressor in (
            NestedCrossApproximation.BottomUp, NestedCrossApproximation.TopDown
        ),
        (name, representor) in (
            ("RandomRep", NestedCrossApproximation.RandomRep()),
            ("ChebyshevRep", NestedCrossApproximation.ChebyshevRep(points)),
            ("MimicryRep", mimicry),
        )

        reduced = compressor(; representor=representor)
        if compressor === NestedCrossApproximation.TopDown && name == "MimicryRep"
            @test_throws ArgumentError NestedCrossApproximation.testbases(
                reduced, countingmatrix, tree, testfardata; ntasks=1
            )
            continue
        end

        counts = Int[]
        for comp in (compressor(), reduced),
            (bases, fardata) in (
                (NestedCrossApproximation.testbases, testfardata),
                (NestedCrossApproximation.trialbases, trialfardata),
            )

            nevals[] = 0
            bases(comp, countingmatrix, tree, fardata; ntasks=1)
            push!(counts, nevals[])
        end
        @test counts[3] < counts[1] / 5
        @test counts[4] < counts[2] / 5

        nca = NestedCrossApproximation.PetrovGalerkinNCA(
            kernel, points, points, tree; testcompressor=reduced, trialcompressor=reduced
        )
        @test norm(nca * x - dense * x) / norm(dense * x) < 5e-4
    end
end
