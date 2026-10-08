# [Using Directional $\mathcal{H}^2$-Matrices](@id wideband)

For an oscillatory kernel, the rank of a far-field block grows when its clusters
become large compared with the wavelength. A directional $\mathcal{H}^2$-matrix
divides such a far field according to the propagation direction and constructs one
nested basis for every active cluster-direction pair [[2, 3, 5]](@ref refs).

NestedCrossApproximation.jl combines this directional high-frequency treatment with
the ordinary low-frequency construction. The same matrix can therefore contain
directional bases on coarse levels and a single basis per cluster on fine levels.

## Basic Usage

Directional assembly uses [`isnearwideband`](@ref
NestedCrossApproximation.isnearwideband) together with [`BottomUp`](@ref
NestedCrossApproximation.BottomUp). The condition needs the wavenumber of the
kernel. The point-kernel interface provides a self-contained example: define a
callable kernel with an `eltype`, cluster the test and trial positions, and pass all
four objects to `PetrovGalerkinNCA`.

The directional construction is currently tested for Helmholtz point kernels and
for the EFIE single-layer operator through BEAST.jl. Double-layer operators are not
currently part of the tested NestedCrossApproximation workflow.

```@example manual-dh2
using NestedCrossApproximation: PetrovGalerkinNCA, BottomUp, isnearwideband, storage
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

λ = 0.5
kernel = HelmholtzKernel(2pi / λ)

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

The geometry-adaptive k-means tree is the recommended backend, particularly for
irregular boundary-element meshes. Its cluster-specific bounding balls are also
used to determine how many directions an electrically large cluster requires.

`BottomUp` is paired with IACA and tree-mimicry pivoting. As in the low-frequency
case, the test compressor receives the trial tree and the trial compressor receives
the test tree:

```@example manual-dh2
tol = 1e-4
convergence = FNormExtrapolator(tol)
testpivoting = TreeMimicryPivoting(
    testpoints, trialpoints, H2Trees.trialtree(tree)
)
trialpivoting = TreeMimicryPivoting(
    trialpoints, testpoints, H2Trees.testtree(tree)
)

dh2mat = PetrovGalerkinNCA(
    kernel,
    testpoints,
    trialpoints,
    tree;
    isnear=isnearwideband(kernel.k),
    testcompressor=BottomUp(;
        factorization=IACA(MaximumValue(), testpivoting, convergence),
    ),
    trialcompressor=BottomUp(;
        factorization=IACA(trialpivoting, MaximumValue(), convergence),
    ),
)
size(dh2mat)
```

The 1,024-by-1,024 example contains both a dense near field and a compressed
directional far field. Inspect the resulting storage relative to the dense matrix
with

```@example manual-dh2
storage(dh2mat)
nothing # hide
```

For this moderate example, compare a product with the dense matrix:

```@example manual-dh2
dense = [kernel(t, s) for t in testpoints, s in trialpoints]
Random.seed!(2)
x = randn(ComplexF64, length(trialpoints))
norm(dh2mat * x - dense * x) / norm(dense * x)
```

The IACA tolerance controls the local directional bases. Errors from different
directions and tree levels contribute to the final product error, so `tol` is not a
strict bound for the complete matrix.

## Mixed-Frequency Admissibility

Construct the condition as

```julia
condition = isnearwideband(k; ηlf=1.0, ηhf=5.0, γ=1.0)
```

For clusters $t$ and $s$, let $d_t$ and $d_s$ denote their diameters and let
$\operatorname{dist}(t,s)$ denote their distance. The pair is treated as low
frequency if

```math
k\min\{d_t,d_s\}\leq\gamma.
```

In this regime, the pair enters the far field if

```math
\max\{d_t,d_s\}\leq
\eta_{\mathrm{lf}}\operatorname{dist}(t,s).
```

All other pairs are classified as high frequency and use the parabolic separation
condition

```math
k\max\{d_t^2,d_s^2\}\leq
\eta_{\mathrm{hf}}\operatorname{dist}(t,s).
```

The three parameters therefore have distinct roles:

| Parameter | Role |
|:----------|:-----|
| `γ` | transition between low- and high-frequency interactions |
| `ηlf` | geometric separation in the low-frequency regime |
| `ηhf` | parabolic separation in the high-frequency regime |

Decreasing `ηlf` or `ηhf` demands a larger separation and enlarges the dense near
field. Increasing `γ` moves the transition toward larger clusters, so more
interactions use the low-frequency condition.

The interaction-level classification above uses the smaller diameter of a pair.
The package separately assigns each cluster a consistent basis regime so that
siblings and inherited bases remain compatible. This distinction matters primarily
for geometry-adaptive k-means trees and is described on the
[directional theory page](@ref wideband_theory).

## Direction Sets and Alternative Tree Backend

The directional subdivision is generated automatically from the cluster tree and
the wavenumber:

- With the k-means `BoundingBallTree`, every electrically large cluster receives a
  geometry-adaptive number of nearly uniform directions. Parent directions are
  mapped to the nearest directions of the child so that the directional bases stay
  nested.
- With a `TwoNTree`, a hierarchical direction tree begins with the coordinate-axis
  directions. Larger clusters on coarser levels receive successively finer angular
  subdivisions.

The basic example uses the k-means backend. A `TwoNTree` remains available as a
lightweight quadtree or octree for regular point sets:

```julia
twonbuilder = TwoNTreeBuilder(; minvalues=32)
tree = BlockTree(
    testpoints,
    trialpoints;
    builder=BlockTreeBuilder(; test=twonbuilder, trial=twonbuilder),
)
```

## Supported Compressor

Directional bases are currently constructed only by `BottomUp`. It processes the
tree from the leaves to the root and runs one IACA for every active
cluster-direction pair. The pivots of the children become the candidates of the
parent, which preserves nestedness without evaluating the complete far field.

`TopDown` and `BatchedBottomUp` currently construct ordinary $\mathcal{H}^2$ bases
and cannot be combined with `isnearwideband`. An EFIE workflow is shown under
[Application Examples](@ref examples).

For the directional approximation, the hierarchy of directions, and the
mixed-frequency construction, see the
[directional $\mathcal{H}^2$-matrix theory](@ref wideband_theory).
