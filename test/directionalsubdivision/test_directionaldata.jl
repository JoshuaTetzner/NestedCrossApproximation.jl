using Test
using H2Trees
using NestedCrossApproximation
using StaticArrays
using Random
using ParallelKMeans

# -----------------------------------------------------------------------------
# Shared invariant checks, applied to both TwoNTree and BoundingBallTree
# results so the two `directionalfardata` implementations (twondirections.jl,
# boundingballdirections.jl) are held to the same contract.
# -----------------------------------------------------------------------------

# For every node, splitting its far nodes by direction (via `dirfarnodes`) and
# concatenating them back must reproduce `farnodes(data, node)` exactly: no
# far node is dropped, duplicated, or misfiled under the wrong direction.
function check_dirfars_partition(tree, data)
    for node in 1:H2Trees.numberofnodes(tree)
        allfars = collect(NestedCrossApproximation.farnodes(data, node))
        collected = Int[]
        for dir in NestedCrossApproximation.dirs(data, node)
            append!(collected, NestedCrossApproximation.dirfarnodes(data, node, dir))
        end
        @test sort(collected) == sort(allfars)
    end
end

# `diridxtofarptr` gives, per direction slot, the range into farnodes. Summed
# (as per-slot range lengths) over all of a node's directions it must equal
# that node's total far node count - i.e. the per-direction ranges are a true
# partition of the count, not just of the node indices checked above.
function check_dircount_sums(tree, data)
    for node in 1:H2Trees.numberofnodes(tree)
        nf = NestedCrossApproximation.nfarnodes(data, node)
        total = sum(
            data.diridxtofarptr[i + 1] - data.diridxtofarptr[i] for
            i in NestedCrossApproximation.dirrange(data, node);
            init=0,
        )
        @test total == nf
    end
end

# `dirmap(data, node)` is only meaningful as "one entry per parent direction"
# for nodes using real (HF) directions: those are built directly off
# `dirs(parent)` in the builders. LF nodes (dirs ∈ {[], [0]}) follow a
# different, sentinel-only contract (see the dedicated edge-case testset
# below), so they are excluded here on purpose.
function check_dirmap_shape(tree, data)
    for node in 1:H2Trees.numberofnodes(tree)
        d = NestedCrossApproximation.dirs(data, node)
        (isempty(d) || d == [0]) && continue
        pnode = H2Trees.parent(tree, node)
        pnode == 0 && continue
        dm = NestedCrossApproximation.dirmap(data, node)
        @test length(dm) == NestedCrossApproximation.ndirections(data, pnode)
        nd = NestedCrossApproximation.ndirections(data, node)
        @test all(v -> 0 <= v <= nd, dm)
    end
end

# Replays exactly the traversal pattern bottomup.jl uses to build nested
# bases/transfer operators: for every node and every one of its direction
# slots, compute `dirfarfield`, and - unless `isbasisnode` says to stop here -
# look up the matching direction slot of every child via
# `dirmap(child)[localdiridx]` followed by `dirrange(child)[...]`. This is the
# actual contract `DirectionalData` must satisfy for bottomup.jl to not throw
# a BoundsError; replaying it here is the most direct way to answer "can the
# rest of the code handle this `DirectionalData`" without standing up the
# full kernel/compression pipeline.
function check_bottomup_traversal_safe(tree, data)
    for t in 1:H2Trees.numberofnodes(tree)
        for (localdiridx, diridx) in enumerate(NestedCrossApproximation.dirrange(data, t))
            dir = data.dirs[diridx]
            Ft = NestedCrossApproximation.dirfarfield(tree, data, t, dir)
            isempty(Ft) && continue
            NestedCrossApproximation.isbasisnode(tree, data, t, dir) && continue
            for child in H2Trees.children(tree, t)
                cdiridx = NestedCrossApproximation.dirmap(data, child)[localdiridx]
                NestedCrossApproximation.dirrange(data, child)[cdiridx]
            end
        end
    end
    return true
end

# `dirfarfield` must start with the node's own direct dirfarnodes, before any
# far nodes inherited from ancestors.
function check_dirfarfield_prefix(tree, data)
    for node in 1:H2Trees.numberofnodes(tree)
        for dir in NestedCrossApproximation.dirs(data, node)
            own = NestedCrossApproximation.dirfarnodes(data, node, dir)
            field = NestedCrossApproximation.dirfarfield(tree, data, node, dir)
            @test field[1:length(own)] == own
        end
    end
end

# For every raw far pair (t, s), the per-direction slot lookup
# (`diridxfromlocalfaridx`) must resolve on *both* sides - this is exactly
# what `couplingmatrices.jl` relies on to assemble the coupling matrix for a
# pair from the test side's and the trial side's own pivot storage.
function check_diridx_roundtrip(testdata, trialdata, nnodes)
    for t in 1:nnodes
        for (localsidx, sidx) in enumerate(NestedCrossApproximation.farrange(testdata, t))
            s = NestedCrossApproximation.farnodes(testdata)[sidx]
            tdiridx = NestedCrossApproximation.diridxfromlocalfaridx(testdata, t, localsidx)
            @test !isnothing(tdiridx)
            localtidx = findfirst(==(t), NestedCrossApproximation.farnodes(trialdata, s))
            @test !isnothing(localtidx)
            sdiridx = NestedCrossApproximation.diridxfromlocalfaridx(
                trialdata, s, localtidx
            )
            @test !isnothing(sdiridx)
        end
    end
end

# -----------------------------------------------------------------------------
# TwoNTree
# -----------------------------------------------------------------------------

@testset "DirectionalData (TwoNTree)" begin
    Random.seed!(3)
    pts = [SVector{3,Float64}(rand(3)) for _ in 1:600]
    pts2 = [SVector{3,Float64}(rand(3) .+ [3.0, 0, 0]) for _ in 1:600]
    ttree = H2Trees.TwoNTree(pts, 1 / 2^6; minvalues=4)
    stree = H2Trees.TwoNTree(pts2, 1 / 2^6; minvalues=4)
    tree = H2Trees.BlockTree(ttree, stree)

    @testset "all-LF regime (long wavelength)" begin
        # k tiny enough that islf holds at every level: no node may end up
        # with a real (non-sentinel) direction.
        isnear = NestedCrossApproximation.isnearwideband(2π / 50.0; γ=1.0)
        testdata, trialdata = NestedCrossApproximation.fardata(tree, isnear)
        for node in 1:H2Trees.numberofnodes(ttree)
            d = NestedCrossApproximation.dirs(testdata, node)
            @test isempty(d) || d == [0]
        end
    end

    @testset "all-HF regime (short wavelength)" begin
        # k huge enough that islf never holds, even at the finest level: the
        # LF sentinel direction 0 must never appear.
        isnear = NestedCrossApproximation.isnearwideband(2π / 0.02; γ=1.0)
        testdata, trialdata = NestedCrossApproximation.fardata(tree, isnear)
        for node in 1:H2Trees.numberofnodes(ttree)
            @test 0 ∉ NestedCrossApproximation.dirs(testdata, node)
        end
    end

    @testset "mixed LF/HF regime" begin
        # k chosen so the tree genuinely splits into an HF top and an LF
        # bottom (verified to give a real mix of empty/HF nodes for this
        # seed) - this is the regime where the LF/HF boundary discussed with
        # the user actually occurs.
        isnear = NestedCrossApproximation.isnearwideband(2π / 1.5; γ=1.0)
        testdata, trialdata = NestedCrossApproximation.fardata(tree, isnear)

        check_dirfars_partition(ttree, testdata)
        check_dircount_sums(ttree, testdata)
        check_dirmap_shape(ttree, testdata)
        check_dirfarfield_prefix(ttree, testdata)
        check_dirfarfield_prefix(stree, trialdata)
        @test check_bottomup_traversal_safe(ttree, testdata)
        @test check_bottomup_traversal_safe(stree, trialdata)
        check_diridx_roundtrip(testdata, trialdata, H2Trees.numberofnodes(ttree))
    end
end

# -----------------------------------------------------------------------------
# BoundingBallTree (KMeansTree)
# -----------------------------------------------------------------------------

@testset "DirectionalData (BoundingBallTree)" begin
    Random.seed!(3)
    pts = [SVector{3,Float64}(rand(3)) for _ in 1:600]
    pts2 = [SVector{3,Float64}(rand(3) .+ [3.0, 0, 0]) for _ in 1:600]
    Random.seed!(11)
    ttree = H2Trees.KMeansTree(pts, 2; minvalues=4)
    stree = H2Trees.KMeansTree(pts2, 2; minvalues=4)
    tree = H2Trees.BlockTree(ttree, stree)

    @testset "all-LF regime (long wavelength)" begin
        isnear = NestedCrossApproximation.isnearwideband(2π / 50.0; γ=1.0)
        testdata, trialdata = NestedCrossApproximation.fardata(tree, isnear)
        for node in 1:H2Trees.numberofnodes(ttree)
            d = NestedCrossApproximation.dirs(testdata, node)
            @test isempty(d) || d == [0]
        end
    end

    @testset "all-HF regime (short wavelength)" begin
        isnear = NestedCrossApproximation.isnearwideband(2π / 0.02; γ=1.0)
        testdata, trialdata = NestedCrossApproximation.fardata(tree, isnear)
        for node in 1:H2Trees.numberofnodes(ttree)
            @test 0 ∉ NestedCrossApproximation.dirs(testdata, node)
        end
    end

    @testset "mixed LF/HF regime" begin
        # For BoundingBallTree the LF/HF split is decided per sibling group
        # (mean radius of all of a node's siblings, see `_bbgenradius`), not
        # per global level - this k/seed combination is verified to produce
        # a real empty/HF mix under that rule.
        isnear = NestedCrossApproximation.isnearwideband(2π / 2.5; γ=1.0)
        testdata, trialdata = NestedCrossApproximation.fardata(tree, isnear)

        check_dirfars_partition(ttree, testdata)
        check_dircount_sums(ttree, testdata)
        check_dirmap_shape(ttree, testdata)
        check_dirfarfield_prefix(ttree, testdata)
        check_dirfarfield_prefix(stree, trialdata)
        @test check_bottomup_traversal_safe(ttree, testdata)
        @test check_bottomup_traversal_safe(stree, trialdata)
        check_diridx_roundtrip(testdata, trialdata, H2Trees.numberofnodes(ttree))
    end
end

# -----------------------------------------------------------------------------
# Dedicated edge case: HF parent whose children fall into the LF regime, with
# no connection between a node's own regime and its parent's. Raised while
# reviewing the directionalsubdivision code: a child with zero far
# interactions of its own, sitting right below an HF parent, ends up with
# `dirs(data, child) == []` (not even the `[0]` LF sentinel), since there is
# no LF ancestor to inherit a sentinel from. The question was whether the
# rest of the code (specifically bottomup.jl's dirmap-driven traversal) can
# handle that `[]` correctly, or whether `[0]` needs to be assigned instead.
#
# Traced through bottomup.jl: `isbasisnode` decides whether to recurse into
# children's `dirmap` based only on the *first* child's `dirs`. Because a
# node's LF/HF classification is always uniform across siblings (same level
# for TwoNTree, same sibling-group mean radius for BoundingBallTree), the
# first child reflects every sibling's regime, so `isbasisnode` always
# returns `true` for an HF node whose children are LF - the dangerous
# children-dirmap branch is never taken in this case, regardless of which
# particular sibling has its own far interactions and which doesn't. This
# synthetic test pins that down for both possible orderings (the far-less
# child listed first, and listed second) directly, rather than relying on
# random point clouds to happen to hit the case.
# -----------------------------------------------------------------------------

@testset "DirectionalData edge case: HF parent, LF children, no shared fars" begin
    Random.seed!(7)
    pts = [SVector{3,Float64}(rand(3)) for _ in 1:600]
    pts2 = [SVector{3,Float64}(rand(3) .+ [3.0, 0, 0]) for _ in 1:600]
    ttree = H2Trees.TwoNTree(pts, 1 / 2^6; minvalues=4)
    stree = H2Trees.TwoNTree(pts2, 1 / 2^6; minvalues=4)
    nnodes = H2Trees.numberofnodes(ttree)

    A = 1
    kids = collect(H2Trees.children(ttree, A))
    @test length(kids) >= 2
    firstkid = H2Trees.firstchild(ttree, A)
    secondkid = first(c for c in kids if c != firstkid)

    # Builds a synthetic FarData with exactly one far interaction at `A`
    # (making it HF with one real direction) and exactly one at `farskid`
    # (an LF child with its own far interaction, so it gets the `[0]`
    # sentinel); the other LF child gets none, so it ends up with `[]`.
    # `farskid`/`emptykid` are not the same variable across the two calls
    # below: which physical child plays which role is swapped to cover both
    # firstchild orderings.
    function build_case(farskid, emptykid)
        counts = zeros(Int, nnodes)
        counts[A] = 1
        counts[farskid] = 1
        ptr = NestedCrossApproximation.ptrfromcounts(counts)
        farn = Int[]
        for node in 1:nnodes
            node == A && push!(farn, 2)
            node == farskid && push!(farn, 3)
        end
        synthfardata = NestedCrossApproximation.FarData(ptr, farn)
        boundary = H2Trees.level(ttree, A) + 1
        syntheticisnear = (; islf=(_, level) -> level >= boundary)
        data = NestedCrossApproximation.directionalfardata(
            synthfardata, ttree, stree, syntheticisnear
        )
        @test emptykid != farskid
        @test NestedCrossApproximation.dirs(data, emptykid) == Int[]
        @test NestedCrossApproximation.dirs(data, farskid) == [0]
        @test NestedCrossApproximation.ndirections(data, A) == 1
        # The protective check: A must be treated as a basis node for its
        # one real direction, so bottomup.jl never tries to index into
        # `dirmap(emptykid)` (which is empty and would throw).
        realdir = NestedCrossApproximation.dirs(data, A)[1]
        @test NestedCrossApproximation.isbasisnode(ttree, data, A, realdir)
        # The generic, no-shortcuts replay of bottomup.jl's traversal over
        # the *whole* tree must still complete without error.
        @test check_bottomup_traversal_safe(ttree, data)
    end

    build_case(firstkid, secondkid)  # far interaction on the first child
    build_case(secondkid, firstkid)  # far interaction on the second child
end
