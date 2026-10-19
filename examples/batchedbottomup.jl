using BEAST
using CompScienceMeshes
using ParallelKMeans
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using CUDA
using Random
using LinearAlgebra
using OhMyThreads
using StaticArrays

Random.seed!(1)
##
λ = 4.0
k = 2 * pi / λ

Γ1 = meshicosphere(400, 1.0)
tRT = raviartthomas(Γ1)
sRT = raviartthomas(Γ1)
println("Size RT ", length(tRT))

op = Maxwell3D.singlelayer(; wavenumber=k)

tree = H2Trees.BlockTree(
    KMeansTree(tRT.pos; builder=KMeansTreeBuilder(; minvalues=100)),
    KMeansTree(sRT.pos; builder=KMeansTreeBuilder(; minvalues=100)),
)
##
# GPU-backed far kernel matrix (ACABEASTCUDA extension, needs BEAST + CUDA loaded).
farquadstrat = AdaptiveCrossApproximation.tofarquadstrat(
    AdaptiveCrossApproximation.defaultmatrixdata(op, tRT, sRT)
)
farmatrix = AdaptiveCrossApproximation.AbstractKernelMatrix(
    op,
    tRT,
    sRT;
    matrixdata=AdaptiveCrossApproximation.GPUMatrixData(
        farquadstrat; ndevices=1, devices=[0]
    ),
);

testfardata, trialfardata = NestedCrossApproximation.fardata(
    tree, AdaptiveCrossApproximation.isnear()
);

testcompressor = NestedCrossApproximation.BatchedBottomUp(;
    pivotselector=TreeMimicryPivoting(tRT.pos, sRT.pos, H2Trees.trialtree(tree)),
    compressor=ACA(FullPivoting(), FullPivoting(), FNormEstimator(1e-3)),
)
trialcompressor = NestedCrossApproximation.BatchedBottomUp(;
    pivotselector=TreeMimicryPivoting(sRT.pos, tRT.pos, H2Trees.testtree(tree)),
    compressor=ACA(FullPivoting(), FullPivoting(), FNormEstimator(1e-3)),
);
##
# testbases/trialbases now return the full (basis, transfer, pivots) triple,
# same contract as BottomUp/TopDown.
CUDA.@time nestedtestbases, testtransfermats, testpivots = NestedCrossApproximation.testbases(
    testcompressor, farmatrix, tree, testfardata; maxrank=20
);
CUDA.@time nestedtrialbases, trialtransfermats, trialpivots = NestedCrossApproximation.trialbases(
    trialcompressor, farmatrix, tree, trialfardata; maxrank=30
);
##
# Far coupling matrices between the reduced test/trial pivot sets, same call
# `PetrovGalerkinNCA` makes internally after both traversals.
CUDA.@time couplingmatrices = NestedCrossApproximation.assemble_couplingstore(
    farmatrix, testpivots, trialpivots, testfardata, trialfardata
);
##
nactive_test = count(isassigned(testpivots, i) for i in eachindex(testpivots))
nactive_trial = count(isassigned(trialpivots, i) for i in eachindex(trialpivots))
println("Active test nodes: ", nactive_test, " / ", length(testpivots))
println("Active trial nodes: ", nactive_trial, " / ", length(trialpivots))
println("Coupling blocks: ", length(couplingmatrices.blocks))

samplet = findfirst(i -> isassigned(testpivots, i), eachindex(testpivots))
samples = findfirst(i -> isassigned(trialpivots, i), eachindex(trialpivots))
println(
    "Sample test node $samplet pivots (",
    length(testpivots[samplet]),
    "): ",
    testpivots[samplet],
)
println(
    "Sample trial node $samples pivots (",
    length(trialpivots[samples]),
    "): ",
    trialpivots[samples],
)
