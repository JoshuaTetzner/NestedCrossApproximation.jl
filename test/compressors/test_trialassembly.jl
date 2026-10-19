using Test
using NestedCrossApproximation
using AdaptiveCrossApproximation
using H2Trees
using ParallelKMeans
using StaticArrays
using Random
using LinearAlgebra
using OhMyThreads

struct TrialAssemblyKernel end
Base.eltype(::TrialAssemblyKernel) = ComplexF64
(::TrialAssemblyKernel)(x, y) = (1 + im*x[1] + x[2]^2) * (exp(2y[2]) + im*y[3])

@testset "Trial assembly for nonsymmetric and rectangular operators" begin
    for (nt, ns) in ((160, 160), (96, 160), (160, 96))
        rng = MersenneTwister(17)
        testpts = [SVector{3,Float64}(rand(rng, 3)) for _ in 1:nt]
        trialpts = [SVector{3,Float64}(rand(rng, 3) .+ [10, 0, 0]) for _ in 1:ns]
        kernel = TrialAssemblyKernel()
        dense = [kernel(x, y) for x in testpts, y in trialpts]
        matrix = AbstractKernelMatrix(kernel, testpts, trialpts)
        owntrial = [[ns-4, ns-3, ns-2, ns-1, ns], [2, 4, 6, 8]]
        fartest = [[1, 3, 5], [nt-1, nt]]
        blocks = NestedCrossApproximation.assemble_level(
            matrix, owntrial, fartest; maxrank=10, transposed=true
        )
        for i in eachindex(blocks)
            @test blocks[i] == transpose(dense[fartest[i], owntrial[i]])
        end

        function cluster(points)
            if isdefined(H2Trees, :KMeansTreeBuilder)
                builder = H2Trees.KMeansTreeBuilder(;
                    numberofclusters=2,
                    minvalues=80,
                    updateradii=H2Trees.boundingsphere,
                    splitterkwargs=(; rng=MersenneTwister(1)),
                )
                return KMeansTree(points; builder)
            end
            return KMeansTree(
                points,
                2;
                minvalues=80,
                updateradii=H2Trees.boundingsphere,
                rng=MersenneTwister(1),
            )
        end
        tree = BlockTree(cluster(testpts), cluster(trialpts))
        testcompressor = NestedCrossApproximation.BatchedBottomUp(;
            pivotselector=TreeMimicryPivoting(testpts, trialpts, H2Trees.trialtree(tree)),
            compressor=ACA(FullPivoting(), FullPivoting(), FNormEstimator(1e-8)),
        )
        trialcompressor = NestedCrossApproximation.BatchedBottomUp(;
            pivotselector=TreeMimicryPivoting(trialpts, testpts, H2Trees.testtree(tree)),
            compressor=ACA(FullPivoting(), FullPivoting(), FNormEstimator(1e-8)),
        )
        A = NestedCrossApproximation.PetrovGalerkinNCA(
            kernel,
            testpts,
            trialpts,
            tree;
            testcompressor,
            trialcompressor,
            maxrank=10,
            ntasks=1,
            verbose=false,
        )
        x = randn(rng, ComplexF64, ns)
        y = randn(rng, ComplexF64, nt)
        @test norm(A*x - dense*x) / norm(dense*x) < 1e-10
        @test norm(adjoint(A)*y - adjoint(dense)*y) / norm(adjoint(dense)*y) < 1e-10
    end
end
