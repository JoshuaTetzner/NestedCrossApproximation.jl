using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using ParallelKMeans
using StaticArrays
using Random
using LinearAlgebra

# Self-contained point-based Helmholtz kernel, used throughout the nca/* tests so they
# don't depend on BEAST/CompScienceMeshes (those are only needed for the manual,
# non-CI scripts test_nca.jl / test_wnca*.jl). Implemented as a callable struct (not a
# plain function) to use AbstractKernelMatrix's recommended interface and avoid the
# "plain function kernel" warning.
struct HelmholtzTestKernel
    k::Float64
end
Base.eltype(::HelmholtzTestKernel) = ComplexF64
function (g::HelmholtzTestKernel)(x, y)
    r = norm(x - y)
    return r < 1e-12 ? zero(ComplexF64) : exp(im * g.k * r) / (4π * r)
end

# Two well-separated random point clouds, so far-field blocks between them are
# genuinely (numerically) low rank - the property every compressor under test relies on.
function randomwellseparatedclouds(n::Int; seed=1, offset=SVector(3.0, 0.0, 0.0))
    Random.seed!(seed)
    testpts = [SVector{3,Float64}(rand(3)) for _ in 1:n]
    trialpts = [SVector{3,Float64}(rand(3) .+ offset) for _ in 1:n]
    return testpts, trialpts
end

function testblocktree(testpts, trialpts; minvalues=20)
    return H2Trees.BlockTree(
        KMeansTree(testpts, 2; minvalues=minvalues),
        KMeansTree(trialpts, 2; minvalues=minvalues),
    )
end

function densereference(kernel, testpts, trialpts)
    n, m = length(testpts), length(trialpts)
    A = zeros(eltype(kernel), n, m)
    for j in 1:m, i in 1:n
        A[i, j] = kernel(testpts[i], trialpts[j])
    end
    return A
end
