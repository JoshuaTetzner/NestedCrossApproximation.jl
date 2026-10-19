#!/usr/bin/env julia
# Creates and saves a 50k test H2-matrix to JLD2 for threading diagnostics.
# Run with: julia -t N scripts/create_test_matrix.jl
# Outputs: test/data/testmatrix50k.jld2
using Pkg
Pkg.activate(dirname(@__DIR__))

using JLD2
using NestedCrossApproximation
using H2Trees
using AdaptiveCrossApproximation
using ParallelKMeans
using StaticArrays
using Random
using LinearAlgebra

# Reuse HelmholtzTestKernel from testutils.jl
include(joinpath(@__DIR__, "..", "test", "testutils.jl"))

N = 25_000
Random.seed!(42)
testpts  = [SVector{3,Float64}(rand(3)) for _ in 1:N]
trialpts = [SVector{3,Float64}(rand(3) .+ 3.0) for _ in 1:N]
kernel   = HelmholtzTestKernel(2.0)

# Build tree: small minvalues → many levels for better parallel stress testing
minvalues = 15
@info "Building KMeansTree (N=$N, minvalues=$minvalues)..."
tree          = H2Trees.BlockTree(testkmeanstree(testpts; minvalues=minvalues), testkmeanstree(trialpts; minvalues=minvalues))
nlevels_test  = length(H2Trees.levels(H2Trees.testtree(tree)))
nlevels_trial = length(H2Trees.levels(H2Trees.trialtree(tree)))
@info "  test levels: $nlevels_test   trial levels: $nlevels_trial"

@info "Assembling H2-matrix (serial, BottomUp+ACA)..."
hmat = NestedCrossApproximation.PetrovGalerkinNCA(
    kernel,
    testpts,
    trialpts,
    tree;
    testcompressor  = NestedCrossApproximation.BottomUp(; factorization=ACA()),
    trialcompressor = NestedCrossApproximation.BottomUp(; factorization=ACA()),
    ntasks          = 1,
    maxrank         = 30,
)

Random.seed!(99)
x        = rand(ComplexF64, N)
y_serial = hmat * x

@info "Saving to test/data/testmatrix50k.jld2..."
mkpath(joinpath(@__DIR__, "..", "test", "data"))
jldsave(
    joinpath(@__DIR__, "..", "test", "data", "testmatrix_50k.jld2");
    testpts,
    trialpts,
    x,
    y_serial,
    nestedtestbases       = hmat.nestedtestbases,
    nestedtrialbases      = hmat.nestedtrialbases,
    testtransfermatrices  = hmat.testtransfermatrices,
    trialtransfermatrices = hmat.trialtransfermatrices,
    couplingmatrices      = hmat.couplingmatrices,
    aggregationplan       = hmat.aggregationplan,
    disaggregationplan    = hmat.disaggregationplan,
    dim                   = hmat.dim,
    nlevels_test,
    nlevels_trial,
)
@info "Saved. Reference MVP result stored as y_serial."
@info "Reload and wrap in PetrovGalerkinNCA using load_test_matrix() below."
