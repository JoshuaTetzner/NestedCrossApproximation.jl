using Test
using H2Trees
using NestedCrossApproximation
using StaticArrays
using Random
using LinearAlgebra
using ParallelKMeans
using CompScienceMeshes

@testset "boundingballdirections.jl" begin
    Random.seed!(5)
    unitvectors = Dict(
        dim => [normalize(SVector{dim,Float64}(randn(dim))) for _ in 1:10000] for
        dim in (2, 3)
    )

    @testset "sphericalfibonaccipoints" begin
        for n in (6, 100, 1600)
            pts = NestedCrossApproximation.sphericalfibonaccipoints(n)

            @test length(pts) == n
            @test all(p -> isapprox(norm(p), 1.0), pts)
            # Evenly spread, not merely distinct: both the covering radius and the
            # closest pair have to scale like 1/sqrt(n).
            @test maximum(
                acos(clamp(maximum(dot(p, v) for p in pts), -1, 1)) for v in unitvectors[3]
            ) < 3 / sqrt(n)
            @test minimum(norm(pts[i] - pts[j]) for i in 1:n for j in (i + 1):n) >
                1.3 / sqrt(n)
        end
    end

    @testset "directions ($(dim)D)" for dim in (2, 3)
        @test length(NestedCrossApproximation.directions(Val(dim), 1.0, 0.1, 1.0)) ==
            2 * dim
        @test eltype(
            eltype(NestedCrossApproximation.directions(Val(dim), 1.0f0, 10.0f0, 1.0f0))
        ) === Float32

        counts = Int[]
        for kd in (5.0, 10.0, 20.0)
            evecs = NestedCrossApproximation.directions(Val(dim), kd, 1.0, 1.0)
            push!(counts, length(evecs))

            @test all(e -> isapprox(norm(e), 1.0), evecs)
            # The count comes from the admissibility aperture asin(γ / (k * diam)), so
            # the set has to cover the sphere that finely. 2D hits it exactly; the
            # Fibonacci lattice used in 3D is within ~15%.
            @test maximum(
                acos(clamp(maximum(dot(e, v) for e in evecs), -1, 1)) for
                v in unitvectors[dim]
            ) < 1.2 * asin(1.0 / kd)
        end
        # And at the expected cost: doubling k * diam doubles the count in 2D and
        # quadruples it in 3D.
        @test all(isapprox(counts[i + 1] / counts[i], 2^(dim - 1); rtol=0.05) for i in 1:2)
    end

    @testset "_nearest_direction" begin
        evecs = [SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0)]
        @test NestedCrossApproximation._nearest_direction(evecs, SVector(0.9, 0.1, 0.1)) ==
            1
        @test NestedCrossApproximation._nearest_direction(evecs, SVector(0.1, 0.9, 0.1)) ==
            2
        @test NestedCrossApproximation._nearest_direction(evecs, SVector(0.1, 0.1, 0.9)) ==
            3

        # Exactly the argmax, on a set where the answer is not obvious by inspection.
        pool = NestedCrossApproximation.sphericalfibonaccipoints(50)
        @test all(
            NestedCrossApproximation._nearest_direction(pool, v) ==
            argmax(i -> dot(pool[i], v), eachindex(pool)) for v in unitvectors[3]
        )
    end

    Random.seed!(3)
    points = [SVector{3,Float64}(rand(3)) for _ in 1:200]
    kmeansbuilder = KMeansTreeBuilder(; numberofclusters=2, minvalues=4)
    Random.seed!(11)
    ttree = KMeansTree(points; builder=kmeansbuilder)
    tree = H2Trees.BlockTree(ttree, KMeansTree(points; builder=kmeansbuilder))
    @testset "directionalfardata ($name)" for (name, λ, haslf, hashf, mixed) in (
        ("all LF", 50.0, true, false, false),
        ("all HF", 0.02, false, true, false),
        ("mixed", 1.5, true, true, true),
    )
        isnear = NestedCrossApproximation.isnearwideband(2π / λ; γ=1.0)
        data, _ = NestedCrossApproximation.fardata(tree, isnear)
        nnodes = H2Trees.numberofnodes(ttree)
        alldirs = NestedCrossApproximation.dirs.(Ref(data), 1:nnodes)
        lf = [isnear.islf(ttree, n) for n in 1:nnodes]
        parents = [H2Trees.parent(ttree, n) for n in 1:nnodes]

        @test any(==([0]), alldirs) == haslf
        @test any(d -> !isempty(d) && d != [0], alldirs) == hashf
        # Per node - from the sibling-group mean radius - not per level.
        @test all((alldirs[n] == [0]) == lf[n] for n in 1:nnodes if !isempty(alldirs[n]))

        # The transition only ever runs HF -> LF, since `_bbgenradius` keeps the
        # generalized radius non-increasing downwards.
        @test all(!lf[parents[n]] || lf[n] for n in 1:nnodes if parents[n] != 0)

        # Crossing it, the sentinel is never inherited: an LF node under an HF parent
        # carries [0] only if it has far nodes of its own.
        atboundary = [n for n in 1:nnodes if parents[n] != 0 && lf[n] && !lf[parents[n]]]
        @test !isempty(atboundary) == mixed
        @test all(
            (alldirs[n] == [0]) == (NestedCrossApproximation.nfarnodes(data, n) > 0) for
            n in atboundary
        )
        # Below it, an LF node under an LF parent does inherit one with no far nodes.
        mixed && @test any(
            alldirs[n] == [0] && NestedCrossApproximation.nfarnodes(data, n) == 0 for
            n in 1:nnodes if parents[n] != 0 && lf[n] && lf[parents[n]]
        )
    end

    # sin of the angle between a far node's interaction vector and the direction it is
    # filed under, relative to the aperture 1 / (k * cluster diameter). γ only steers the
    # construction; the bounds are what each γ currently achieves.
    spherepoints = meshicosphere(100, 1.0).vertices
    Random.seed!(11)
    spheretree = KMeansTree(
        spherepoints; builder=KMeansTreeBuilder(; numberofclusters=2, minvalues=50)
    )
    @testset "directional admissibility (γ=$γ)" for (γ, ownbound, inheritedbound) in
                                                    ((1.0, 1.18, 2.18), (0.5, 0.6, 1.14))
        k = 2π / 0.25
        isnear = NestedCrossApproximation.isnearwideband(k; γ=γ)
        data, _ = NestedCrossApproximation.fardata(
            H2Trees.BlockTree(spheretree, spheretree), isnear
        )
        @test !any(
            node -> isnear.islf(spheretree, node), 1:H2Trees.numberofnodes(spheretree)
        )

        sinangle(v, c) = dot(v, c) <= 0 ? 1.0 : norm(cross(v, c)) / (norm(v) * norm(c))

        worstown, worstinherited = 0.0, 0.0
        for node in 1:H2Trees.numberofnodes(spheretree)
            diam = 2 * H2Trees.radius(spheretree, node)
            evecs = NestedCrossApproximation.directions(Val(3), diam, k, γ)
            for dir in NestedCrossApproximation.dirs(data, node)
                # `dirfarfield` lists the node's own far nodes first, then the inherited.
                own = length(NestedCrossApproximation.dirfarnodes(data, node, dir))
                farfield = NestedCrossApproximation.dirfarfield(spheretree, data, node, dir)
                for (i, farnode) in enumerate(farfield)
                    vec = NestedCrossApproximation.interaction(
                        node, farnode, spheretree, spheretree
                    )
                    outside = sinangle(vec, evecs[dir]) * k * diam
                    if i <= own
                        worstown = max(worstown, outside)
                    else
                        worstinherited = max(worstinherited, outside)
                    end
                end
            end
        end

        @test worstown <= ownbound
        @test worstinherited <= inheritedbound
    end
end
