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

# Minimal PetrovGalerkinNCA comparison: BatchedBottomUp far-field compression
# run against a CPU (BEASTKernelMatrix) vs a GPU (GPUBEASTKernelMatrix) far
# kernel matrix, on a ~20k RWG problem. Near interactions stay CPU-assembled
# in both cases - GPU near assembly is a separate, already-exercised path
# (see nearinteractions_gpu.jl) and isn't what BatchedBottomUp exists for.
#
# Run the CPU section with multiple threads, e.g.:
#   JULIA_NUM_THREADS=16 julia --project=<workspace> examples/batchedbottomup_gpu_vs_cpu.jl

Random.seed!(1)
##
λ = 4.0
k = 2 * pi / λ

Γ = meshicosphere(60, 1.0)
tRT = raviartthomas(Γ)
sRT = raviartthomas(Γ)
println("Size RT ", length(tRT))

op = Maxwell3D.singlelayer(; wavenumber=k)

tree = H2Trees.BlockTree(
    KMeansTree(
        tRT.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
    ),
    KMeansTree(
        sRT.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
    ),
)

maxrank = 30
tol = 1e-3
x = randn(ComplexF64, length(sRT))

buildcompressors() = (
    NestedCrossApproximation.BatchedBottomUp(;
        pivotselector=TreeMimicryPivoting(tRT.pos, sRT.pos, H2Trees.trialtree(tree)),
        compressor=ACA(FullPivoting(), FullPivoting(), FNormEstimator(tol)),
    ),
    NestedCrossApproximation.BatchedBottomUp(;
        pivotselector=TreeMimicryPivoting(sRT.pos, tRT.pos, H2Trees.testtree(tree)),
        compressor=ACA(FullPivoting(), FullPivoting(), FNormEstimator(tol)),
    ),
)

nearquadstrat = BEAST.DoubleNumSauterQstrat(4, 5, 3, 3, 4, 3)
farquadstrat = AdaptiveCrossApproximation.tofarquadstrat(
    AdaptiveCrossApproximation.defaultmatrixdata(op, tRT, sRT)
)
## CPU: BatchedBottomUp + plain BEASTKernelMatrix far matrix
println("=== CPU (BatchedBottomUp, ", Threads.nthreads(), " threads) ===")
testcompressor_cpu, trialcompressor_cpu = buildcompressors()
@time Acpu = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    tRT,
    sRT,
    tree;
    testcompressor=testcompressor_cpu,
    trialcompressor=trialcompressor_cpu,
    maxrank=maxrank,
    matrixdata=nearquadstrat,
    farmatrixdata=farquadstrat,
    scheduler=DynamicScheduler(),
    verbose=false,
);
#@time ycpu = Acpu * x;
##
## GPU: BatchedBottomUp + GPUBEASTKernelMatrix far matrix (near interactions
## still CPU-assembled - only farmatrixdata switches to GPUMatrixData).
println("=== GPU (BatchedBottomUp) ===")
testcompressor_gpu, trialcompressor_gpu = buildcompressors()
CUDA.@time Agpu = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    tRT,
    sRT,
    tree;
    testcompressor=testcompressor_gpu,
    trialcompressor=trialcompressor_gpu,
    matrixdata=AdaptiveCrossApproximation.GPUMatrixData(nearquadstrat),
    farmatrixdata=AdaptiveCrossApproximation.GPUMatrixData(farquadstrat),
    maxrank=maxrank,
    scheduler=DynamicScheduler(),
    verbose=false,
)
#@time ygpu = Agpu * x;
##
println("reldiff CPU vs GPU far mat-vec: ", norm(ycpu - ygpu) / norm(ycpu))

##
