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

function testkmeanstree(pts; numberofclusters=2, minvalues=20)
    return KMeansTree(
        pts;
        builder=KMeansTreeBuilder(; numberofclusters=numberofclusters, minvalues=minvalues),
    )
end

function testblocktree(testpts, trialpts; minvalues=20)
    return H2Trees.BlockTree(
        testkmeanstree(testpts; minvalues=minvalues),
        testkmeanstree(trialpts; minvalues=minvalues),
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

# Plane-wave kernels with an analytically known spectrum. With `r` directions,
# K(x,y) = Σᵢ aᵢ exp(i cᵢ·x) exp(-i cᵢ·y) has rank exactly `r` by construction, and the
# amplitudes aᵢ prescribe the singular-value decay. Equal amplitudes (decay=1) make every
# far block exactly rank r, so at maxrank = r the whole H2 operator has to reproduce the
# dense matrix to round-off - an exactness statement, not a tolerance. Geometrically
# decaying amplitudes instead give a convergence curve with known targets: truncating at
# rank k leaves roughly decay^k. Both are far sharper instruments than the Helmholtz
# fixture, whose spectrum is whatever the geometry happens to produce.
struct PlaneWaveKernel
    dirs::Vector{SVector{3,Float64}}
    amps::Vector{Float64}
end
Base.eltype(::PlaneWaveKernel) = ComplexF64
function (g::PlaneWaveKernel)(x, y)
    return sum(
        g.amps[i] * exp(im * dot(g.dirs[i], x)) * exp(-im * dot(g.dirs[i], y)) for
        i in eachindex(g.dirs)
    )
end

function planewavekernel(r::Int; decay=1.0, seed=5, spread=3.0)
    Random.seed!(seed)
    return PlaneWaveKernel(
        [SVector{3,Float64}(randn(3) * spread) for _ in 1:r], [decay^(i - 1) for i in 1:r]
    )
end
