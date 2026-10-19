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

Γ1 = meshicosphere(60, 1.0)
Γ2 = meshicosphere(60, 1.0)
tRT = raviartthomas(Γ1)
sRT = raviartthomas(Γ2)
println("Size RT ", length(tRT))

op = Maxwell3D.singlelayer(; wavenumber=k)

tree = H2Trees.BlockTree(
    KMeansTree(
        tRT.pos;
        builder=KMeansTreeBuilder(;
            minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
        ),
    ),
    KMeansTree(
        sRT.pos;
        builder=KMeansTreeBuilder(;
            minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
        ),
    ),
)
##
# The ACABEASTCUDA extension has no GPU Wilton correction, so GPU near-field
# assembly needs an explicit Sauter (not the BEAST default Wilton-Sauter)
# quadrature strategy. (2,3,6,7,5,5,4,3) is BEAST's own WiltonSauter default for
# Maxwell/RT (mwops.jl); dropping the Wilton-specific leading pair (2,3) and
# keeping the rest gives the matching plain-Sauter fallback used elsewhere in
# BEAST for non-Wilton operators.
nearquadstrat = BEAST.DoubleNumSauterQstrat(4, 5, 3, 3, 4, 3)
gpumatrixdata = AdaptiveCrossApproximation.GPUMatrixData(
    nearquadstrat; ndevices=1, devices=[7]
)
##
# `assemblenears` returns a single `BlockSparseMatrix`, not a
# (blocks, values, nearvalues) tuple - each entry of `nears.blocks` carries its
# own `rowindices`/`colindices` (see BlockSparseMatrices.DenseMatrixBlock).
@time nears_cpu = NestedCrossApproximation.assemblenears(
    op,
    tRT,
    sRT,
    tree;
    isnear=AdaptiveCrossApproximation.isnear(),
    ntasks=Threads.nthreads(),
    matrixdata=nearquadstrat,
);
##
CUDA.@time nears_gpu = NestedCrossApproximation.assemblenears(
    op,
    tRT,
    sRT,
    tree;
    isnear=AdaptiveCrossApproximation.isnear(),
    ntasks=Threads.nthreads(),
    matrixdata=gpumatrixdata,
);
#println("Near blocks: ", length(nears_gpu.blocks))
##
# Sanity-check a handful of blocks against the CPU path, same quadrature
# strategy (so this isolates CPU/GPU numerical agreement, not a difference in
# quadrature choice). Block ordering is deterministic (BlockSparseMatrix's
# constructor sorts by max row/col index) and both calls share the same
# tree/isnear, so blocks at the same position correspond to the same
# (rowindices, colindices) pair.
for i in 1:5
    cpublk = nears_cpu.blocks[i].matrix
    gpublk = Array(nears_gpu.blocks[i].matrix)
    println("block $i reldiff = ", norm(gpublk - cpublk) / norm(cpublk))
end
