using Test
using NestedCrossApproximation
using AdaptiveCrossApproximation
using H2Trees
using ParallelKMeans
using LinearAlgebra
using StaticArrays
using CompScienceMeshes

struct HelmholtzKernel
    k::Float64
end
Base.eltype(::HelmholtzKernel) = ComplexF64
function (kernel::HelmholtzKernel)(x, y)
    r = norm(x - y)
    return r == 0 ? zero(ComplexF64) : exp(-im * kernel.k * r) / (4π * r)
end
@testset "petrovgalerkinnca.jl" begin
    # Vertices of a mesh with edge length h on squares of side a in the plane z = 0.
    a = 1.0
    small = meshrectangle(a, a, 0.16).vertices
    large = meshrectangle(a, a, 0.01).vertices

    geometries = (
        ("single square", 0.16, small, small),
        ("separated squares", 0.16, small, small .+ Ref(SVector(3a, 0.0, 0.0))),
        ("touching squares", 0.01, large, large .+ Ref(SVector(a, 0.0, 0.0))),
    )

    @testset "low frequency ($name)" for (name, h, testpts, trialpts) in geometries
        kernel = HelmholtzKernel(2π / (500h))

        reference = Matrix{ComplexF64}(undef, length(testpts), length(trialpts))
        Threads.@threads for j in eachindex(trialpts)
            for i in eachindex(testpts)
                reference[i, j] = kernel(testpts[i], trialpts[j])
            end
        end
        x = randn(ComplexF64, length(trialpts))
        y = reference * x

        kmeansbuilder = KMeansTreeBuilder(; minvalues=50)
        kmeanstree = BlockTree(
            KMeansTree(testpts; builder=kmeansbuilder),
            KMeansTree(trialpts; builder=kmeansbuilder),
        )
        twonbuilder = TwoNTreeBuilder(; minvalues=100)
        twontree = BlockTree(
            testpts,
            trialpts;
            builder=BlockTreeBuilder(; test=twonbuilder, trial=twonbuilder),
        )

        for tree in (kmeanstree, twontree), tol in (1e-2, 1e-4, 1e-6)
            testpivoting = TreeMimicryPivoting(testpts, trialpts, H2Trees.trialtree(tree))
            trialpivoting = TreeMimicryPivoting(trialpts, testpts, H2Trees.testtree(tree))
            convergence = FNormExtrapolator(tol)
            fullaca = ACA(;
                tol=tol, rowpivoting=FullPivoting(), columnpivoting=FullPivoting()
            )
            compressors = (
                (
                    NestedCrossApproximation.TopDown(;
                        factorization=IACA(
                            MaximumValue(), MimicryPivoting(testpts, trialpts), convergence
                        ),
                    ),
                    NestedCrossApproximation.TopDown(;
                        factorization=IACA(
                            MimicryPivoting(trialpts, testpts), MaximumValue(), convergence
                        ),
                    ),
                ),
                (
                    NestedCrossApproximation.BottomUp(;
                        factorization=IACA(MaximumValue(), testpivoting, convergence)
                    ),
                    NestedCrossApproximation.BottomUp(;
                        factorization=IACA(trialpivoting, MaximumValue(), convergence)
                    ),
                ),
                (
                    NestedCrossApproximation.BatchedBottomUp(;
                        pivotselector=testpivoting, compressor=fullaca
                    ),
                    NestedCrossApproximation.BatchedBottomUp(;
                        pivotselector=trialpivoting, compressor=fullaca
                    ),
                ),
            )

            for (testcompressor, trialcompressor) in compressors
                nca = NestedCrossApproximation.PetrovGalerkinNCA(
                    kernel,
                    testpts,
                    trialpts,
                    tree;
                    testcompressor=testcompressor,
                    trialcompressor=trialcompressor,
                    maxrank=50,
                )
                @test norm(nca * x - y) / norm(y) < 5tol
            end
        end
    end
    @testset "high frequency ($name)" for (name, h, testpts, trialpts) in geometries
        kernel = HelmholtzKernel(2π / (40h))

        reference = Matrix{ComplexF64}(undef, length(testpts), length(trialpts))
        Threads.@threads for j in eachindex(trialpts)
            for i in eachindex(testpts)
                reference[i, j] = kernel(testpts[i], trialpts[j])
            end
        end
        x = randn(ComplexF64, length(trialpts))
        y = reference * x

        kmeansbuilder = KMeansTreeBuilder(; minvalues=50)
        kmeanstree = BlockTree(
            KMeansTree(testpts; builder=kmeansbuilder),
            KMeansTree(trialpts; builder=kmeansbuilder),
        )
        twonbuilder = TwoNTreeBuilder(; minvalues=100)
        twontree = BlockTree(
            testpts,
            trialpts;
            builder=BlockTreeBuilder(; test=twonbuilder, trial=twonbuilder),
        )

        for tree in (kmeanstree, twontree), tol in (1e-2, 1e-4, 1e-6)
            testpivoting = TreeMimicryPivoting(testpts, trialpts, H2Trees.trialtree(tree))
            trialpivoting = TreeMimicryPivoting(trialpts, testpts, H2Trees.testtree(tree))
            convergence = FNormExtrapolator(tol)
            testcompressor = NestedCrossApproximation.BottomUp(;
                factorization=IACA(MaximumValue(), testpivoting, convergence)
            )
            trialcompressor = NestedCrossApproximation.BottomUp(;
                factorization=IACA(trialpivoting, MaximumValue(), convergence)
            )

            nca = NestedCrossApproximation.PetrovGalerkinNCA(
                kernel,
                testpts,
                trialpts,
                tree;
                testcompressor=testcompressor,
                trialcompressor=trialcompressor,
                isnear=NestedCrossApproximation.isnearwideband(
                    kernel.k; ηlf=1.0, ηhf=1.0, γ=1.0
                ),
            )
            @test norm(nca * x - y) / norm(y) < 10tol
        end
    end
    @testset "single far interaction" begin
        _, h, testpts, trialpts = geometries[2]
        kernel = HelmholtzKernel(2π / (100h))

        reference = Matrix{ComplexF64}(undef, length(testpts), length(trialpts))
        Threads.@threads for j in eachindex(trialpts)
            for i in eachindex(testpts)
                reference[i, j] = kernel(testpts[i], trialpts[j])
            end
        end
        x = randn(ComplexF64, length(trialpts))
        y = reference * x

        kmeansbuilder = KMeansTreeBuilder(; minvalues=60)
        tree = BlockTree(
            KMeansTree(testpts; builder=kmeansbuilder),
            KMeansTree(trialpts; builder=kmeansbuilder),
        )

        for tol in (1e-2, 1e-4, 1e-6)
            factorizations = (
                (ACA(; tol=tol), ACA(; tol=tol)),
                (
                    IACA(
                        MaximumValue(),
                        MimicryPivoting(testpts, trialpts),
                        FNormExtrapolator(tol),
                    ),
                    IACA(
                        MimicryPivoting(trialpts, testpts),
                        MaximumValue(),
                        FNormExtrapolator(tol),
                    ),
                ),
            )
            representors = (
                (nothing, nothing),
                (
                    NestedCrossApproximation.RandomRep(),
                    NestedCrossApproximation.RandomRep(),
                ),
                (
                    NestedCrossApproximation.ChebyshevRep(trialpts),
                    NestedCrossApproximation.ChebyshevRep(testpts),
                ),
                (
                    NestedCrossApproximation.MimicryRep(
                        TreeMimicryPivoting(testpts, trialpts, H2Trees.trialtree(tree))
                    ),
                    NestedCrossApproximation.MimicryRep(
                        TreeMimicryPivoting(trialpts, testpts, H2Trees.testtree(tree))
                    ),
                ),
            )

            for compressor in
                (NestedCrossApproximation.TopDown, NestedCrossApproximation.BottomUp),
                (testfactorization, trialfactorization) in factorizations,
                (testrepresentor, trialrepresentor) in representors

                # MimicryRep is only supported with BottomUp.
                compressor === NestedCrossApproximation.TopDown &&
                    testrepresentor isa NestedCrossApproximation.MimicryRep &&
                    continue

                nca = NestedCrossApproximation.PetrovGalerkinNCA(
                    kernel,
                    testpts,
                    trialpts,
                    tree;
                    testcompressor=compressor(;
                        factorization=testfactorization, representor=testrepresentor
                    ),
                    trialcompressor=compressor(;
                        factorization=trialfactorization, representor=trialrepresentor
                    ),
                )
                @test norm(nca * x - y) / norm(y) < 5tol
            end
        end
    end
end
