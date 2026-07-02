using Test
using NestedCrossApproximation
using AdaptiveCrossApproximation
using H2Trees
using BEAST
using CompScienceMeshes
using ParallelKMeans
using LinearAlgebra
using StaticArrays
using Random

struct ToyKernel{T}
    α::T
end
(kern::ToyKernel)(x, y) = exp(-kern.α * norm(x - y))
Base.eltype(::ToyKernel{T}) where {T} = T

function test_assemblenears_against_reference(nears, Aref, N)
    @test size(nears) == (N, N)
    # Materialize once via Matrix(nears) (mul!-based) rather than scalar-indexing
    # `nears` directly: BlockSparseMatrix is a LinearMaps.LinearMap, and scalar
    # getindex on AbstractBlockMatrix has been broken upstream since
    # BlockSparseMatrices v0.3.0 (regression, still present in v0.3.1).
    densenears = Matrix(nears)
    nnzcount = count(!iszero, densenears)
    @test 0 < nnzcount < N^2
    @test all(
        iszero(densenears[i, j]) || isapprox(densenears[i, j], Aref[i, j]; rtol=1e-8) for
        i in 1:N, j in 1:N
    )
    @test all(!iszero(densenears[i, i]) for i in 1:N)
end

@testset "IsLowFrequencyFunctor" begin
    γ = 1.0
    k = 2.0
    f = NestedCrossApproximation.islf(γ, k)

    @test NestedCrossApproximation.wavenumber(f) == k
    @test f(γ / k) == true        # k * diam == γ -> still admissible under <=
    @test f(0.5 * γ / k) == true  # well inside the low-frequency regime
    @test f(2 * γ / k) == false   # clearly high-frequency
end

@testset "IsNearWidebandFunctor" begin
    # γ = 1.0 throughout, so islf(diam) reduces to k * diam <= 1.
    isnearlf = NestedCrossApproximation.isnearwideband(0.1; ηlf=1.0, ηhf=5.0, γ=1.0)
    isnearhf = NestedCrossApproximation.isnearwideband(2.0; ηlf=1.0, ηhf=5.0, γ=1.0)
    @test isnearlf.islf(2.0) == true
    @test isnearhf.islf(2.0) == false

    @testset "TwoNTree" begin
        treea = H2Trees.TwoNTree(SVector(0.0, 0.0, 0.0), 1.0)
        treebfar = H2Trees.TwoNTree(SVector(20.0, 0.0, 0.0), 1.0)
        treebnear = H2Trees.TwoNTree(SVector(1.0, 0.0, 0.0), 1.0)

        @test isnearlf(treea, treebfar, 1, 1) == false
        @test isnearlf(treea, treebnear, 1, 1) == true
        @test isnearhf(treea, treebfar, 1, 1) == false
        @test isnearhf(treea, treebnear, 1, 1) == true
    end

    @testset "BoundingBallTree" begin
        treea = H2Trees.BoundingBallTree(SVector(0.0, 0.0, 0.0), 1.0)
        treebfar = H2Trees.BoundingBallTree(SVector(20.0, 0.0, 0.0), 1.0)
        treebnear = H2Trees.BoundingBallTree(SVector(1.0, 0.0, 0.0), 1.0)

        @test isnearlf(treea, treebfar, 1, 1) == false
        @test isnearlf(treea, treebnear, 1, 1) == true
        @test isnearhf(treea, treebfar, 1, 1) == false
        @test isnearhf(treea, treebnear, 1, 1) == true
    end
end

@testset "assemblenears (PointMatrix toy example, TwoNTree)" begin
    Random.seed!(1)
    pts = [@SVector rand(3) for _ in 1:80]
    tree = H2Trees.TwoNTree(pts, pts, 1 / 2^6; testminvalues=8, trialminvalues=8)

    kernel = ToyKernel(1.0)
    nears = NestedCrossApproximation.assemblenears(
        kernel, pts, pts, tree, NestedCrossApproximation.PreserveSpaceOrder()
    )

    N = length(pts)
    Aref = [kernel(pts[i], pts[j]) for i in 1:N, j in 1:N]
    test_assemblenears_against_reference(nears, Aref, N)
end

@testset "assemblenears (PointMatrix toy example, KMeansTree)" begin
    Random.seed!(1)
    pts = [@SVector rand(3) for _ in 1:80]
    clustertree = H2Trees.KMeansTree(pts, 2; minvalues=10)
    tree = H2Trees.BlockTree(clustertree, clustertree)

    kernel = ToyKernel(1.0)
    nears = NestedCrossApproximation.assemblenears(
        kernel, pts, pts, tree, NestedCrossApproximation.PreserveSpaceOrder()
    )

    N = length(pts)
    Aref = [kernel(pts[i], pts[j]) for i in 1:N, j in 1:N]
    test_assemblenears_against_reference(nears, Aref, N)
end

@testset "assemblenears (BEAST toy example, TwoNTree)" begin
    Γ = meshrectangle(1.0, 1.0, 0.2)
    X = raviartthomas(Γ)
    op = Maxwell3D.singlelayer(; wavenumber=1.0)
    Aref = assemble(op, X, X)

    tree = H2Trees.TwoNTree(X.pos, X.pos, 1 / 2^4; testminvalues=5, trialminvalues=5)
    nears = NestedCrossApproximation.assemblenears(
        op, X, X, tree, NestedCrossApproximation.PreserveSpaceOrder()
    )

    test_assemblenears_against_reference(nears, Aref, length(X))
end

@testset "assemblenears (BEAST toy example, KMeansTree)" begin
    Γ = meshrectangle(1.0, 1.0, 0.2)
    X = raviartthomas(Γ)
    op = Maxwell3D.singlelayer(; wavenumber=1.0)
    Aref = assemble(op, X, X)

    Random.seed!(1)
    clustertree = H2Trees.KMeansTree(X.pos, 2; minvalues=5)
    tree = H2Trees.BlockTree(clustertree, clustertree)
    nears = NestedCrossApproximation.assemblenears(
        op, X, X, tree, NestedCrossApproximation.PreserveSpaceOrder()
    )

    test_assemblenears_against_reference(nears, Aref, length(X))
end
