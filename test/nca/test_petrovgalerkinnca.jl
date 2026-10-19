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
    @testset "workspace" begin
        _, h, testpts, trialpts = geometries[3]
        kernel = HelmholtzKernel(2π / (500h))
        tol = 1e-4

        reference = Matrix{ComplexF64}(undef, length(testpts), length(trialpts))
        Threads.@threads for j in eachindex(trialpts)
            for i in eachindex(testpts)
                reference[i, j] = kernel(testpts[i], trialpts[j])
            end
        end

        kmeansbuilder = KMeansTreeBuilder(; minvalues=50)
        tree = BlockTree(
            KMeansTree(testpts; builder=kmeansbuilder),
            KMeansTree(trialpts; builder=kmeansbuilder),
        )
        convergence = FNormExtrapolator(tol)
        nca = NestedCrossApproximation.PetrovGalerkinNCA(
            kernel,
            testpts,
            trialpts,
            tree;
            testcompressor=NestedCrossApproximation.TopDown(;
                factorization=IACA(
                    MaximumValue(), MimicryPivoting(testpts, trialpts), convergence
                ),
            ),
            trialcompressor=NestedCrossApproximation.TopDown(;
                factorization=IACA(
                    MimicryPivoting(trialpts, testpts), MaximumValue(), convergence
                ),
            ),
        )
        core = NestedCrossApproximation.core(nca)
        H2Workspace = NestedCrossApproximation.H2Workspace
        modes(A, x, u) = [A * x, transpose(A) * u, adjoint(A) * u]
        x = randn(ComplexF64, size(nca, 2))
        u = randn(ComplexF64, size(nca, 1))
        expected = modes(nca, x, u)

        @testset "products agree with the dense matrix" begin
            dense = [reference * x, transpose(reference) * u, adjoint(reference) * u]
            for (y, ref) in zip(expected, dense)
                @test norm(y - ref) / norm(ref) < 10tol
            end
        end

        @testset "a second workspace over the same core agrees exactly" begin
            other = H2Workspace(core)
            @test NestedCrossApproximation.core(other) === core
            @test NestedCrossApproximation.core(transpose(other)) === core
            @test other.nearinteractions === nca.nearinteractions
            @test other.trialcoeffs !== nca.trialcoeffs
            @test other.testcoeffs !== nca.testcoeffs
            @test modes(other, x, u) == expected
            serial = H2Workspace(core; scheduler=NestedCrossApproximation.SerialScheduler())
            @test modes(serial, x, u) ≈ expected
        end

        @testset "repeated products on one workspace are stable" begin
            @test modes(nca, x, u) == expected
            @test modes(nca, x, u) == expected
        end

        @testset "far and near field" begin
            far = NestedCrossApproximation.farmatrix(nca)
            near = NestedCrossApproximation.nearmatrix(nca)
            @test far.trialcoeffs !== nca.trialcoeffs
            @test far * x + near * x ≈ expected[1]
        end

        @testset "linear maps" begin
            y = randn(ComplexF64, size(nca, 1))
            y0 = copy(y)
            α, β = 0.3 + 0.2im, -1.1
            @test mul!(y, nca, x, α, β) ≈ α * expected[1] + β * y0
            @test (nca + 2nca) * x ≈ 3expected[1]
            @test (nca' * nca) * x ≈ adjoint(nca) * expected[1]
            @test (transpose(nca) - nca') * u ≈ expected[2] - expected[3]
            X = randn(ComplexF64, size(nca, 2), 3)
            Y = zeros(ComplexF64, size(nca, 1), 3)
            @test mul!(Y, nca, X) ≈ reduce(hcat, [nca * X[:, j] for j in 1:3])
            xreal = randn(size(nca, 2))
            @test nca * xreal ≈ nca * complex(xreal)
            # Element type of the buffers differs from the vectors: allocating fallback.
            single = H2Workspace(core, ComplexF32)
            @test single * x ≈ expected[1]
            @test !nca.busy[]
        end

        @testset "concurrent workspaces, independent inputs" begin
            workspaces = [H2Workspace(core) for _ in 1:4]
            inputs = [randn(ComplexF64, size(nca, 2)) for _ in 1:4]
            serials = [nca * xi for xi in inputs]
            for _ in 1:3
                tasks = map(zip(workspaces, inputs)) do (ws, xi)
                    return Threads.@spawn ws * xi
                end
                @test fetch.(tasks) == serials
            end
        end

        @testset "one product at a time" begin
            y = similar(expected[1])
            busyerror = NestedCrossApproximation.ConcurrentWorkspaceError
            @test_throws ErrorException NestedCrossApproximation.withworkspace(nca) do
                return error("boom")
            end
            @test !nca.busy[]
            @test_throws DimensionMismatch mul!(y, nca, x[1:(end - 1)])
            @test !nca.busy[]

            NestedCrossApproximation.withworkspace(nca) do
                @test nca.busy[]
                @test_throws busyerror mul!(y, nca, x)
                @test_throws busyerror mul!(similar(x), transpose(nca), u)
                @test_throws busyerror mul!(similar(x), adjoint(nca), u)
            end
            @test !nca.busy[]
            @test nca * x == expected[1]
        end
    end
end
