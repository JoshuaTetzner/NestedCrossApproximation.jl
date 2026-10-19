using Test
using NestedCrossApproximation
using H2Trees
using ParallelKMeans
using StaticArrays
using Random

# Near/far classification itself lives in AdaptiveCrossApproximation; what this package
# owns is the wideband predicate and its low-frequency classifier.

# Single-node trees carrying a prescribed effective radius - the box circumradius for
# TwoNTree, the ball radius for BoundingBallTree - at distance `c` along x.
twontreeat(F, r, c) = H2Trees.TwoNTree(SVector{3,F}(c, 0, 0), F(r / sqrt(3)))
balltreeat(F, r, c) = H2Trees.BoundingBallTree(SVector{3,F}(c, 0, 0), F(r))

@testset "nearinteractions.jl" begin
    @testset "islf ($F)" for F in (Float32, Float64)
        f = NestedCrossApproximation.islf(F(1), F(2))

        @test f isa NestedCrossApproximation.IsLowFrequencyFunctor{F}
        @test NestedCrossApproximation.wavenumber(f) === F(2)
        @test f(F(0.5)) isa Bool
        @test f(F(0.5))  # k * diam == γ is still low frequency
        @test !f(F(0.6))

        # diam = 2√3 * halfsize / 2^(level - minimumlevel), so k = 2, γ = 1 puts the
        # crossover between levels 3 and 4.
        tree = H2Trees.TwoNTree(zero(SVector{3,F}), one(F))
        @test !f(tree, 3)
        @test f(tree, 4)
    end

    # The node method derives its diameter from `_bbgenradius`, which averages over
    # siblings so that a whole sibling group is classified alike.
    @testset "islf (BoundingBallTree nodes)" begin
        Random.seed!(3)
        points = [SVector{3,Float64}(rand(3)) for _ in 1:600]
        Random.seed!(11)
        ttree = KMeansTree(
            points; builder=KMeansTreeBuilder(; numberofclusters=2, minvalues=4)
        )
        # k is picked so that a sibling-group mean radius straddles the regime threshold,
        # which is where the minimum against the parent is load bearing.
        f = NestedCrossApproximation.islf(1.0, 4.4)
        regimes = [f(ttree, node) for node in 1:H2Trees.numberofnodes(ttree)]

        @test !allequal(regimes)
        @test all(
            allequal([regimes[c] for c in H2Trees.children(ttree, node)]) for
            node in 1:H2Trees.numberofnodes(ttree) if !H2Trees.isleaf(ttree, node)
        )

        # Taking the minimum against the parent keeps the generalized radius from growing
        # downwards, so a low-frequency node can have no high-frequency descendant.
        @test all(
            !regimes[node] || all([regimes[c] for c in H2Trees.children(ttree, node)]) for
            node in 1:H2Trees.numberofnodes(ttree) if !H2Trees.isleaf(ttree, node)
        )
    end

    @testset "isnearwideband ($F)" for F in (Float32, Float64)
        f = NestedCrossApproximation.isnearwideband(F(2); ηlf=F(1), ηhf=F(5), γ=F(1))

        @test f isa NestedCrossApproximation.IsNearWidebandFunctor{F}
        @test (f.k, f.ηlf, f.ηhf) === (F(2), F(1), F(5))
        @test f.islf isa NestedCrossApproximation.IsLowFrequencyFunctor{F}

        # The wavenumber alone fixes the element type; the defaults must not widen it.
        g = NestedCrossApproximation.isnearwideband(F(2))
        @test g isa NestedCrossApproximation.IsNearWidebandFunctor{F}
        @test (g.ηlf, g.ηhf) === (F(1), F(5))
        @test g.islf isa NestedCrossApproximation.IsLowFrequencyFunctor{F}
    end

    # An integer wavenumber still has to produce a float predicate.
    let f = NestedCrossApproximation.isnearwideband(2)
        @test f isa NestedCrossApproximation.IsNearWidebandFunctor{Float64}
        @test f.islf isa NestedCrossApproximation.IsLowFrequencyFunctor{Float64}
    end

    @testset "IsNearWidebandFunctor ($name, $F)" for (name, treeat) in (
            "TwoNTree" => twontreeat, "BoundingBallTree" => balltreeat
        ),
        F in (Float32, Float64)

        wb(k, ηlf) =
            NestedCrossApproximation.isnearwideband(F(k); ηlf=F(ηlf), ηhf=F(5), γ=F(1))
        isnear(f, ra, rb, d) = f(treeat(F, ra, 0), treeat(F, rb, d), 1, 1)

        lf, hf = wb(0.1, 1), wb(40, 1)

        # Overlapping nodes clamp the separation to zero, which no criterion can pass.
        @test isnear(lf, 1, 1, 0)
        @test isnear(hf, 1, 1, 0)

        # Low frequency: 2 max(r) > ηlf * sep, i.e. near up to d = 4r.
        @test isnear(lf, 1, 1, 3.9)
        @test !isnear(lf, 1, 1, 4.1)
        @test !isnear(wb(0.1, 4), 1, 1, 3.9)

        # High frequency: 4 k max(r)^2 > ηhf * sep.
        @test !isnear(hf, 1, 1, 40)
        @test isnear(wb(60, 1), 1, 1, 40)

        # The smaller node picks the branch. At k = 40 a radius-0.01 node is low frequency
        # and a radius-1 node is not, and here the two criteria disagree: the low-frequency
        # one calls the pair far, the high-frequency one would call it near.
        @test !isnear(hf, 1, 0.01, 4)
        @test isnear(hf, 1, 1, 4)

        @test isnear(hf, 1, 0.01, 4) == isnear(hf, 0.01, 1, 4)
    end

    # Both tree types feed the same criteria, so a box and the ball around it agree.
    let f = NestedCrossApproximation.isnearwideband(40.0; ηlf=1.0, ηhf=5.0, γ=1.0)
        for r in (0.01, 1.0), d in (0.0, 3.9, 4.1, 40.0)
            @test f(twontreeat(Float64, 1, 0), twontreeat(Float64, r, d), 1, 1) ==
                f(balltreeat(Float64, 1, 0), balltreeat(Float64, r, d), 1, 1)
        end
    end

    # Float32 in, Float32 out: nothing inside the predicates may reach for a Float64
    # literal. `optimize=false` keeps the IR to the method itself, before Base internals
    # (Float32^Int among them) are inlined.
    let f = NestedCrossApproximation.isnearwideband(2.0f0),
        boxes = typeof.((twontreeat(Float32, 1, 0), twontreeat(Float32, 1, 4))),
        balls = typeof.((balltreeat(Float32, 1, 0), balltreeat(Float32, 1, 4)))

        nowiden(g, argtypes) =
            !any(
                T -> T === Float64,
                only(code_typed(g, argtypes; optimize=false))[1].ssavaluetypes,
            )

        @test nowiden(f, (boxes..., Int, Int))
        @test nowiden(f, (balls..., Int, Int))
        @test nowiden(f.islf, (Float32,))
        @test nowiden(f.islf, (boxes[1], Int))
        @test nowiden(f.islf, (balls[1], Int))
    end
end
