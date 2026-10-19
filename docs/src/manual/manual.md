# [General Usage](@id Manual)

This page walks through the steps required to assemble an $\mathcal{H}^2$-matrix with NestedCrossApproximation.jl.
The central entry point is [`PetrovGalerkinNCA`](@ref), which takes an operator, test and trial spaces, a block tree, and a choice of compression strategy.

## Prerequisites

You need a kernel matrix (an `AbstractKernelMatrix`) and a compatible block tree.
The block tree partitions the index sets of the test and trial spaces into a hierarchy of clusters; it is typically built with [H2Trees.jl](https://github.com/FastBEAST/H2Trees.jl).

```julia
using NestedCrossApproximation
using AdaptiveCrossApproximation
using H2Trees
using ParallelKMeans
```

## Building the cluster tree

```julia
Random.seed!(1)
testtree  = KMeansTree(testspace.pos,  2; minvalues=100)
Random.seed!(1)
trialtree = KMeansTree(trialspace.pos, 2; minvalues=100)
tree = H2Trees.BlockTree(testtree, trialtree)
```

`minvalues=100` controls the leaf size: clusters with fewer than 100 degrees of freedom are leaves and are handled as near (dense) interactions.

## Choosing a compressor

### TopDown (ACA)

Suitable for low-frequency problems. Constructs nested bases top-down using ACA.

```julia
testcompressor  = TopDown(; factorization=ACA())
trialcompressor = TopDown(; factorization=ACA())
```

### BottomUp (iACA)

Suitable for low-frequency problems where pivot reuse across levels reduces cost.
Requires a `TreeMimicryPivoting` strategy that mirrors the opposite tree's geometry.

```julia
using AdaptiveCrossApproximation: iACA, MaximumValue, TreeMimicryPivoting, FNormExtrapolator, iFNormEstimator

tol = 1e-4
testcompressor = BottomUp(;
    factorization=iACA(
        MaximumValue(),
        TreeMimicryPivoting(testspace.pos, trialspace.pos, H2Trees.trialtree(tree)),
        FNormExtrapolator(iFNormEstimator(tol)),
    ),
)
trialcompressor = BottomUp(;
    factorization=iACA(
        TreeMimicryPivoting(testspace.pos, trialspace.pos, H2Trees.testtree(tree)),
        MaximumValue(),
        FNormExtrapolator(iFNormEstimator(tol)),
    ),
)
```

## Assembling the H2-matrix

```julia
h2mat = PetrovGalerkinNCA(
    op, testspace, trialspace, tree;
    testcompressor  = testcompressor,
    trialcompressor = trialcompressor,
    tol             = 1e-4,
    verbose         = true,
)
```

The returned object supports `*` (matrix-vector product), `transpose`, and `adjoint`.

## Parallelism

Pass a scheduler from [OhMyThreads.jl](https://juliafolds2.github.io/OhMyThreads.jl/) to use multiple threads:

```julia
using OhMyThreads

h2mat = PetrovGalerkinNCA(
    op, testspace, trialspace, tree;
    testcompressor  = testcompressor,
    trialcompressor = trialcompressor,
    scheduler       = DynamicScheduler(),
)
```

## Admissibility

The default admissibility condition is the standard low-frequency condition.
For problems in the wideband regime, use [`isnearwideband`](@ref):

```julia
isnear = isnearwideband(k; ηhf=5.0, γ=1.0)

h2mat = PetrovGalerkinNCA(
    op, testspace, trialspace, tree;
    isnear          = isnear,
    testcompressor  = testcompressor,
    trialcompressor = trialcompressor,
)
```
