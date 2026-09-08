using Test
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using LinearAlgebra
using Random

# End-to-end matvec accuracy for the full PetrovGalerkinNCA pipeline (nearinteractions +
# fardata + testbases/trialbases + transfermatrices + couplingmatrices + abstractnca's
# aggregate/couple/disaggregate matvec), compared against a dense reference. This is the
# contract every other test file's narrower checks exist to support.
@testset "petrovgalerkinnca.jl (end-to-end matvec)" begin
    testpts, trialpts = randomwellseparatedclouds(200; seed=7)
    kernel = HelmholtzTestKernel(2.0)
    tree = testblocktree(testpts, trialpts; minvalues=15)
    A = densereference(kernel, testpts, trialpts)
    n, m = size(A)

    Random.seed!(42)
    x = rand(ComplexF64, m)
    xt = rand(ComplexF64, n)

    compressors = [
        "TopDown+ACA (default)" => NestedCrossApproximation.TopDown(; factorization=ACA()),
        "TopDown+ACA (custom convergence)" => NestedCrossApproximation.TopDown(;
            factorization=ACA(; convergence=FNormEstimator(1e-6))
        ),
        "BottomUp+ACA (default)" =>
            NestedCrossApproximation.BottomUp(; factorization=ACA()),
        "BottomUp+ACA (custom convergence)" => NestedCrossApproximation.BottomUp(;
            factorization=ACA(; convergence=FNormEstimator(1e-6))
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
            @test size(hmat) == (n, m)
            @test eltype(hmat) == ComplexF64

            y = hmat * x
            @test norm(A * x - y) / norm(A * x) < 1e-2

            yt = transpose(hmat) * xt
            @test norm(transpose(A) * xt - yt) / norm(transpose(A) * xt) < 1e-2

            ya = adjoint(hmat) * xt
            @test norm(adjoint(A) * xt - ya) / norm(adjoint(A) * xt) < 1e-2
        end
    end
end
