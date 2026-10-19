using Test
using H2Trees
using NestedCrossApproximation
using StaticArrays
using Random
using LinearAlgebra
using CompScienceMeshes

@testset "twondirections.jl" begin
    Random.seed!(5)
    unitvectors = Dict(
        dim => [normalize(SVector{dim,Float64}(randn(dim))) for _ in 1:10000] for
        dim in (2, 3)
    )

    @testset "DirectionTree ($(dim)D)" for dim in (2, 3)
        maxlevel = 3
        branching = 2^(dim - 1)
        dirtree = NestedCrossApproximation.DirectionTree(Val(dim), maxlevel, 5)
        nnodes = NestedCrossApproximation.numberofnodes(dirtree)
        roots = NestedCrossApproximation.rootnodes(dirtree)

        @test nnodes == 2 * dim * (branching^maxlevel - 1) ÷ (branching - 1)
        @test roots == 1:(2 * dim)
        @test NestedCrossApproximation.children(dirtree, 0) == roots
        # +x, -x, +y, -y, ...
        @test [NestedCrossApproximation.directionvector(dirtree, r) for r in roots] == [
            SVector{dim,Float64}(ntuple(i -> i == axis ? sign : 0.0, dim)) for axis in
                                                                               1:dim for
            sign in (1.0, -1.0)
        ]
        @test all(
            norm(NestedCrossApproximation.directionvector(dirtree, n)) ≈ 1 for n in 1:nnodes
        )

        @test all(NestedCrossApproximation.parent(dirtree, r) == 0 for r in roots)
        @test all(
            NestedCrossApproximation.parent(dirtree, c) == n for n in 1:nnodes for
            c in NestedCrossApproximation.children(dirtree, n)
        )
        @test all(
            length(NestedCrossApproximation.children(dirtree, n)) == branching for
            n in 1:nnodes if !NestedCrossApproximation.isleaf(dirtree, n)
        )
        @test count(n -> NestedCrossApproximation.isleaf(dirtree, n), 1:nnodes) ==
            2 * dim * branching^(maxlevel - 1)
    end

    @testset "DirectionTree (arguments)" begin
        dirtree = NestedCrossApproximation.DirectionTree(Val(3), 2, 1, Float32)
        @test eltype(NestedCrossApproximation.directionvector(dirtree, 7)) === Float32
        @test_throws ErrorException NestedCrossApproximation.DirectionTree(Val(1), 2)
        @test_throws ErrorException NestedCrossApproximation.DirectionTree(Val(3), 0)
    end

    @testset "direction ($(dim)D)" for (dim, cones) in (
        (2, (0.79, 0.47, 0.25, 0.13)), (3, (0.96, 0.62, 0.34, 0.18))
    )
        dirroot, maxlevel = 6, 4
        branching = 2^(dim - 1)
        dirtree = NestedCrossApproximation.DirectionTree(Val(dim), maxlevel, dirroot)
        vectors = unitvectors[dim]
        pick(v, level) = NestedCrossApproximation.direction(v, dirtree, level)

        @test all(
            pick(NestedCrossApproximation.directionvector(dirtree, r), dirroot) == r for
            r in NestedCrossApproximation.rootnodes(dirtree)
        )
        # Below the direction root a cluster is low frequency and gets no direction.
        @test all(pick(v, dirroot + 1) == 0 for v in vectors)
        # Above the depth of the direction tree the descent is clamped.
        @test all(
            pick(v, dirroot - maxlevel - 3) ==
            NestedCrossApproximation.direction(v, dirtree) for v in vectors
        )

        for depth in 1:maxlevel
            level = dirroot - depth + 1
            picks = [pick(v, level) for v in vectors]

            @test all(dirtree.nodes[p].level == depth for p in picks)
            @test length(unique(picks)) == 2 * dim * branching^(depth - 1)
            # One level coarser gives the parent direction.
            depth > 1 && @test all(
                NestedCrossApproximation.parent(dirtree, p) == pick(v, level + 1) for
                (v, p) in zip(vectors, picks)
            )
            # Worst angle to the direction a vector is filed under, in radians.
            @test maximum(
                acos(
                    clamp(
                        dot(v, NestedCrossApproximation.directionvector(dirtree, p)), -1, 1
                    ),
                ) for (v, p) in zip(vectors, picks)
            ) <= cones[depth]
        end
    end

    Random.seed!(3)
    points = [SVector{3,Float64}(rand(3)) for _ in 1:200]
    builder = TwoNTreeBuilder(; minhalfsize=1 / 2^6, minvalues=2)
    ttree = TwoNTree(points; builder=builder)
    tree = H2Trees.BlockTree(ttree, TwoNTree(points; builder=builder))
    nnodes = H2Trees.numberofnodes(ttree)
    levelof = [H2Trees.level(ttree, n) for n in 1:nnodes]
    parents = [H2Trees.parent(ttree, n) for n in 1:nnodes]

    @testset "_dirroot_and_maxlevel" begin
        @test extrema(levelof) == (1, 5)
        # A far pointer with one far node on each of the given levels.
        function ptron(levels)
            nodes = [findfirst(==(level), levelof) for level in levels]
            return cumsum([1; [n in nodes ? 1 : 0 for n in 1:nnodes]])
        end
        dirrootandmaxlevel(levels, λ) = NestedCrossApproximation._dirroot_and_maxlevel(
            ttree, ptron(levels), NestedCrossApproximation.islf(1.0, 2π / λ)
        )

        # Levels 1 to 4 are high frequency at λ=1, level 5 is not.
        islf = NestedCrossApproximation.islf(1.0, 2π / 1.0)
        @test !islf(ttree, 4) && islf(ttree, 5)
        @test dirrootandmaxlevel(Int[], 1.0) == (4, 1)
        @test dirrootandmaxlevel([4], 1.0) == (4, 1)
        @test dirrootandmaxlevel([3], 1.0) == (4, 2)
        @test dirrootandmaxlevel([2, 4], 1.0) == (4, 3)
        # A far node on the low-frequency level does not deepen the direction tree.
        @test dirrootandmaxlevel([5], 1.0) == (4, 1)
        @test dirrootandmaxlevel([3, 5], 1.0) == (4, 2)

        # All levels low frequency: the root lies above the tree.
        @test dirrootandmaxlevel([3], 50.0) == (0, 1)
        # All levels high frequency: the root lies below the leaves.
        @test dirrootandmaxlevel([3], 0.2) == (6, 4)

        @test_throws ErrorException NestedCrossApproximation._dirroot_and_maxlevel(
            ttree, ones(Int, nnodes), islf
        )
    end

    @testset "directionalfardata ($name)" for (name, λ, haslf, hashf) in (
        ("all LF", 50.0, true, false),
        ("all HF", 0.2, false, true),
        ("mixed", 1.5, true, true),
    )
        isnear = NestedCrossApproximation.isnearwideband(2π / λ; γ=1.0)
        data, _ = NestedCrossApproximation.fardata(tree, isnear)
        dirtree = NestedCrossApproximation.DirectionTree(
            ttree, NestedCrossApproximation.farptr(data), isnear.islf
        )
        alldirs = NestedCrossApproximation.dirs.(Ref(data), 1:nnodes)
        lf = [isnear.islf(ttree, levelof[n]) for n in 1:nnodes]
        ishf(n) = !isempty(alldirs[n]) && alldirs[n] != [0]

        @test any(==([0]), alldirs) == haslf
        @test any(ishf, 1:nnodes) == hashf
        # Per level.
        @test all((alldirs[n] == [0]) == lf[n] for n in 1:nnodes if !isempty(alldirs[n]))

        # The depth of a direction follows the level of its cluster.
        @test all(
            dirtree.nodes[dir].level == dirtree.dirroot - levelof[n] + 1 for
            n in 1:nnodes if ishf(n) for dir in alldirs[n]
        )
        # A parent direction is handed down as its parent in the direction tree.
        @test all(
            alldirs[n][NestedCrossApproximation.dirmap(data, n)[slot]] ==
            NestedCrossApproximation.parent(dirtree, pdir) for
            n in 1:nnodes if ishf(n) && parents[n] != 0 && ishf(parents[n]) for
            (slot, pdir) in enumerate(alldirs[parents[n]])
        )

        # An LF node under an HF parent carries [0] only if it has far nodes of its own.
        atboundary = [n for n in 1:nnodes if parents[n] != 0 && lf[n] && !lf[parents[n]]]
        @test !isempty(atboundary) == (haslf && hashf)
        @test all(
            (alldirs[n] == [0]) == (NestedCrossApproximation.nfarnodes(data, n) > 0) for
            n in atboundary
        )
        # An LF node under an LF parent inherits it also without far nodes.
        haslf && @test any(
            alldirs[n] == [0] && NestedCrossApproximation.nfarnodes(data, n) == 0 for
            n in 1:nnodes if parents[n] != 0 && lf[n] && lf[parents[n]]
        )
    end

    # sin of the angle between a far node's interaction vector and the direction it is
    # filed under, relative to the aperture 1 / (k * cluster diameter). γ only steers the
    # construction; the bounds are what each γ currently achieves.
    spherepoints = meshicosphere(100, 1.0).vertices
    spheretree = TwoNTree(spherepoints; builder=TwoNTreeBuilder(; minvalues=100))
    @testset "directional admissibility (γ=$γ)" for (γ, ownbound, inheritedbound) in (
        (1.0, 1.9, 2.15), (0.5, 0.97, 1.28), (0.25, 0.56, 0.82)
    )
        k = 2π / 0.25
        isnear = NestedCrossApproximation.isnearwideband(k; γ=γ)
        data, _ = NestedCrossApproximation.fardata(
            H2Trees.BlockTree(spheretree, spheretree), isnear
        )
        @test !any(level -> isnear.islf(spheretree, level), H2Trees.levels(spheretree))
        dirtree = NestedCrossApproximation.DirectionTree(
            spheretree, NestedCrossApproximation.farptr(data), isnear.islf
        )

        sinangle(v, c) = dot(v, c) <= 0 ? 1.0 : norm(cross(v, c)) / (norm(v) * norm(c))

        worstown, worstinherited = 0.0, 0.0
        for node in 1:H2Trees.numberofnodes(spheretree)
            diam = 2 * sqrt(3) * H2Trees.halfsize(spheretree, node)
            for dir in NestedCrossApproximation.dirs(data, node)
                evec = NestedCrossApproximation.directionvector(dirtree, dir)
                # `dirfarfield` lists the node's own far nodes first, then the inherited.
                own = length(NestedCrossApproximation.dirfarnodes(data, node, dir))
                farfield = NestedCrossApproximation.dirfarfield(spheretree, data, node, dir)
                for (i, farnode) in enumerate(farfield)
                    vec = NestedCrossApproximation.interaction(
                        node, farnode, spheretree, spheretree
                    )
                    outside = sinangle(vec, evec) * k * diam
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
