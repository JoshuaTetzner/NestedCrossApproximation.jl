# [Using $\mathcal{H}^2$-Matrices](@id manual)

[`PetrovGalerkinNCA`](@ref NestedCrossApproximation.PetrovGalerkinNCA) constructs an
$\mathcal{H}^2$-matrix from selected entries of a kernel matrix. Dense blocks store
near interactions, while nested bases and small coupling matrices represent the far
field. The result implements Julia's matrix-vector product interface.

Point kernels provide a self-contained way to use and test the nested construction
without a boundary-element space or quadrature rule. Define a callable kernel with
an `eltype`, cluster its test and trial positions, and pass the kernel, positions,
and block tree to `PetrovGalerkinNCA`.

The currently tested scope comprises Helmholtz point kernels and the EFIE
single-layer operator through [BEAST.jl](https://github.com/krcools/BEAST.jl). The
EFIE workflow uses the same constructor and is shown under
[Application Examples](@ref examples). Double-layer operators are not currently
part of the tested NestedCrossApproximation workflow.

## Point-Kernel Matrices

Define a callable kernel and its matrix entry type. The two point sets below are
separated so that the kernel has no singular entries.

```@example manual-h2
using NestedCrossApproximation: PetrovGalerkinNCA, TopDown, storage
using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using ParallelKMeans
using Random
using StaticArrays

struct HelmholtzKernel
    k::Float64
end

Base.eltype(::HelmholtzKernel) = ComplexF64
function (kernel::HelmholtzKernel)(x, y)
    r = norm(x - y)
    return cis(kernel.k * r) / (4pi * r)
end

testpoints = [
    SVector(x, y, 0.0)
    for x in range(0, 1; length=32), y in range(0, 1; length=32)
][:]
trialpoints = testpoints .+ Ref(SVector(1.1, 0.0, 0.2))
kernel = HelmholtzKernel(2pi / 10)
nothing # hide
```

The block tree combines one cluster tree for the test points with one for the trial
points. For general use, a k-means tree is the recommended backend because its
balanced clusters adapt to the geometry. Loading ParallelKMeans.jl enables this
backend in H2Trees.jl.

```@example manual-h2
testbuilder = KMeansTreeBuilder(;
    numberofclusters=2,
    minvalues=32,
    splitterkwargs=(; rng=MersenneTwister(1)),
)
trialbuilder = KMeansTreeBuilder(;
    numberofclusters=2,
    minvalues=32,
    splitterkwargs=(; rng=MersenneTwister(1)),
)
tree = BlockTree(
    KMeansTree(testpoints; builder=testbuilder),
    KMeansTree(trialpoints; builder=trialbuilder),
)
nothing # hide
```

`numberofclusters` is the number of children created when a cluster is split, and
`minvalues` limits the number of points in a leaf. Smaller leaves expose more
compression opportunities; larger leaves reduce tree depth and create larger dense
terminal blocks.

Each side of the matrix has its own nested basis. The `testcompressor` constructs
the test basis and the `trialcompressor` constructs the trial basis. A direct first
choice is [`TopDown`](@ref NestedCrossApproximation.TopDown) with ACA:

```@example manual-h2
tol = 1e-4
h2mat = PetrovGalerkinNCA(
    kernel,
    testpoints,
    trialpoints,
    tree;
    testcompressor=TopDown(; factorization=ACA(; tol=tol)),
    trialcompressor=TopDown(; factorization=ACA(; tol=tol)),
)
size(h2mat)
```

This example has 1,024 points on each side and a nonempty compressed far field.
[`storage`](@ref NestedCrossApproximation.storage) reports the stored memory and
its ratio to a dense matrix:

```@example manual-h2
storage(h2mat)
nothing # hide
```

The factorization tolerance controls every local basis construction. The error of
the complete matrix is usually of the same order, but is not bounded by `tol` because
local errors can accumulate through the cluster tree.

## Working with the Result

`PetrovGalerkinNCA` is a `LinearMap` and supports ordinary, transposed, and adjoint
products. A dense matrix is useful for checking a small problem:

```@example manual-h2
dense = [kernel(t, s) for t in testpoints, s in trialpoints]
Random.seed!(2)
x = randn(ComplexF64, length(trialpoints))
z = randn(ComplexF64, length(testpoints))

norm(h2mat * x - dense * x) / norm(dense * x)
```

```@example manual-h2
norm(transpose(h2mat) * z - transpose(dense) * z) / norm(transpose(dense) * z)
```

```@example manual-h2
norm(h2mat' * z - dense' * z) / norm(dense' * z)
```

The compressed far field and the dense near field can be applied separately:

```@example manual-h2
using NestedCrossApproximation: farmatrix, nearmatrix

far = farmatrix(h2mat)
near = nearmatrix(h2mat)
norm(far * x + near * x - h2mat * x) / norm(h2mat * x)
```

## Choosing a Compressor

A compressor combines the tree traversal with a low-rank factorization from
AdaptiveCrossApproximation.jl.

| Compressor | Construction | Typical factorization | Use case |
|:-----------|:-------------|:----------------------|:---------|
| [`TopDown`](@ref NestedCrossApproximation.TopDown) | root to leaves | `ACA` or `IACA` | direct construction with strong accuracy at tight tolerances |
| [`BottomUp`](@ref NestedCrossApproximation.BottomUp) | leaves to root | `IACA` with `TreeMimicryPivoting` | fewer matrix-entry evaluations |
| [`BatchedBottomUp`](@ref NestedCrossApproximation.BatchedBottomUp) | leaves to root, level by level | fully pivoted `ACA` | kernels that evaluate many blocks in one batch |

### BottomUp with IACA

`BottomUp` compresses a cluster against its complete far field and restricts the
candidates of every parent to pivots inherited from its children. Consequently,
tree-aware geometric pivoting is the natural choice. The pivoting on the test side
uses the trial tree, and vice versa:

```@example manual-h2
using NestedCrossApproximation: BottomUp

convergence = FNormExtrapolator(tol)
testpivoting = TreeMimicryPivoting(
    testpoints, trialpoints, H2Trees.trialtree(tree)
)
trialpivoting = TreeMimicryPivoting(
    trialpoints, testpoints, H2Trees.testtree(tree)
)

testcompressor = BottomUp(;
    factorization=IACA(MaximumValue(), testpivoting, convergence),
)
trialcompressor = BottomUp(;
    factorization=IACA(trialpivoting, MaximumValue(), convergence),
)
nothing # hide
```

`MaximumValue()` selects a pivot from an evaluated residual row or column.
`TreeMimicryPivoting` selects the complementary pivot geometrically, without first
evaluating the corresponding complete residual vector. This is the incomplete
adaptive cross approximation described in [[4]](@ref refs).

### Other Configurations

`TopDown` can also use IACA. Its far field is a set of points rather than tree
nodes, so it requires `MimicryPivoting`:

```@example manual-h2
testcompressor = TopDown(;
    factorization=IACA(
        MaximumValue(),
        MimicryPivoting(testpoints, trialpoints),
        convergence,
    ),
)
trialcompressor = TopDown(;
    factorization=IACA(
        MimicryPivoting(trialpoints, testpoints),
        MaximumValue(),
        convergence,
    ),
)
nothing # hide
```

Both traversal directions accept ordinary ACA as well:

```@example manual-h2
bottomup_aca = BottomUp(; factorization=ACA(; tol=tol))
nothing # hide
```

`BatchedBottomUp` first selects bounded far-field subsets, assembles the resulting
small blocks together, and then applies a dense fully pivoted ACA to every block.
It is intended for kernel backends whose main advantage is batched evaluation.

## [Far-Field Representors](@id representors)

A representor reduces the complete far field of a cluster before ACA or IACA is
applied. This can substantially reduce the number of matrix entries evaluated by
`TopDown` or `BottomUp`. The resulting factorization tests convergence only on the
representative block, so the reduction can also increase the approximation error.

Three strategies are available:

| Representor | Selection | Supported compressor | Restriction |
|:------------|:----------|:---------------------|:------------|
| [`RandomRep`](@ref NestedCrossApproximation.RandomRep) | uniform random sample | `TopDown`, `BottomUp` | none |
| [`ChebyshevRep`](@ref NestedCrossApproximation.ChebyshevRep) | points closest to a tensor Chebyshev grid fitted to the far field | `TopDown`, `BottomUp` | three-dimensional positions |
| [`MimicryRep`](@ref NestedCrossApproximation.MimicryRep) | tree-mimicry selection from the geometry | `BottomUp` | requires the far field as cluster-tree nodes |

The representor is passed to each compressor independently. `RandomRep` needs no
geometrical data:

```julia
using NestedCrossApproximation: RandomRep

testcompressor = TopDown(;
    factorization=ACA(; tol=tol),
    representor=RandomRep(),
)
trialcompressor = TopDown(;
    factorization=ACA(; tol=tol),
    representor=RandomRep(),
)
```

`ChebyshevRep` receives the positions on the far side of the corresponding
compressor. The test basis is compressed against the trial far field, while the
trial basis is compressed against the test far field:

```julia
using NestedCrossApproximation: ChebyshevRep

testcompressor = BottomUp(;
    factorization=ACA(; tol=tol),
    representor=ChebyshevRep(trialpoints),
)
trialcompressor = BottomUp(;
    factorization=ACA(; tol=tol),
    representor=ChebyshevRep(testpoints),
)
```

`MimicryRep` wraps a `TreeMimicryPivoting` constructed with the tree on the far
side. It works only with `BottomUp`, because that compressor retains the
cluster-node representation of the inherited far field:

```julia
using NestedCrossApproximation: MimicryRep

testrepresentor = MimicryRep(
    TreeMimicryPivoting(testpoints, trialpoints, H2Trees.trialtree(tree)),
)
trialrepresentor = MimicryRep(
    TreeMimicryPivoting(trialpoints, testpoints, H2Trees.testtree(tree)),
)

testcompressor = BottomUp(;
    factorization=ACA(; tol=tol),
    representor=testrepresentor,
)
trialcompressor = BottomUp(;
    factorization=ACA(; tol=tol),
    representor=trialrepresentor,
)
```

`BatchedBottomUp` has no separate `representor` keyword. Its first phase already
performs a geometry-based reduction of every far field through its
`pivotselector`.

## Admissibility and Alternative Tree Backend

By default, a pair of clusters $t$ and $s$ enters the far field when

```math
\max\{\operatorname{diam}(t),\operatorname{diam}(s)\}
\leq \eta\,\operatorname{dist}(t,s),
```

with $\eta=1$. Pass `isnear(η)` to change the separation parameter:

```julia
h2mat = PetrovGalerkinNCA(
    kernel,
    testpoints,
    trialpoints,
    tree;
    isnear=isnear(0.5),
    testcompressor,
    trialcompressor,
)
```

A smaller `η` demands a larger separation and therefore stores more interactions in
the dense near field. For electrically large clusters, use the mixed-frequency
condition described in [Using Directional $\mathcal{H}^2$-Matrices](@ref wideband).

The basic example uses the recommended geometry-adaptive k-means tree. A `TwoNTree`
can also be used. It provides a lightweight quadtree or octree for regular point
sets and small examples:

```julia
twonbuilder = TwoNTreeBuilder(; minvalues=32)
tree = BlockTree(
    testpoints,
    trialpoints;
    builder=BlockTreeBuilder(; test=twonbuilder, trial=twonbuilder),
)
```

## Assembly Options

The principal keywords of `PetrovGalerkinNCA` are:

| Keyword | Default | Meaning |
|:--------|:--------|:--------|
| `testcompressor` | `TopDown()` | construction of the test basis |
| `trialcompressor` | `TopDown()` | construction of the trial basis |
| `isnear` | `isnear()` | separation of near and far interactions |
| `maxrank` | `50` | upper bound for every local basis rank |
| `ntasks` | `Threads.nthreads()` | number of parallel tasks for assembly and products |
| `verbose` | `false` | display assembly progress |

Reaching `maxrank` can indicate that the requested local tolerance was not attained.
Assembly and matrix-vector products run on `ntasks` parallel tasks;
`ntasks=1` runs on the calling task. The scheduling itself is chosen internally.

For the nested representation and the construction of its bases, transfer matrices,
and coupling matrices, see the [$\mathcal{H}^2$-matrix theory](@ref nca_theory).
