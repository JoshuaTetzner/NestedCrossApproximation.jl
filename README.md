# NestedCrossApproximation.jl

[![Docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JoshuaTetzner.github.io/NestedCrossApproximation.jl/stable/)
[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JoshuaTetzner.github.io/NestedCrossApproximation.jl/dev/)
[![CI](https://github.com/JoshuaTetzner/NestedCrossApproximation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JoshuaTetzner/NestedCrossApproximation.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/JoshuaTetzner/NestedCrossApproximation.jl/graph/badge.svg)](https://codecov.io/gh/JoshuaTetzner/NestedCrossApproximation.jl)

## Introduction

This package constructs $\mathcal H^2$-matrices with the nested cross
approximation (NCA) [1]. It builds nested cluster bases, transfer matrices, and
coupling matrices directly from selected entries of a dense matrix. In the
low-frequency regime, this reduces storage and the cost of a matrix-vector product
from $\mathcal O(N^2)$ to $\mathcal O(N)$ under the usual bounded-rank and sparsity
assumptions.

The package also provides a wideband directional $\mathcal H^2$ construction for
oscillatory kernels [2, 3, 5]. It combines ordinary low-frequency bases with
directional bases for electrically large clusters and reaches
$\mathcal O(N\log N)$ complexity under the corresponding assumptions.

## Features

The following capabilities are implemented (✔) and planned (⌛):

### $\mathcal H^2$ Construction

- ✔ Kernel-independent construction from matrix entries
- ✔ Top-down, bottom-up, and batched bottom-up compressors
- ✔ ACA and IACA low-rank factorizations through
  [AdaptiveCrossApproximation.jl](https://github.com/JoshuaTetzner/AdaptiveCrossApproximation.jl)
- ✔ Random, Chebyshev-grid, and tree-mimicry far-field representors
- ✔ Geometry-adaptive k-means trees and regular $2^N$-trees through
  [H2Trees.jl](https://github.com/djukic14/H2Trees.jl)
- ✔ Multithreaded assembly and matrix-vector products
- ✔ Matrix-vector, transpose, and adjoint products through the Julia matrix interface
- ⌛ Convenience `assemble` interface for point kernels and BEAST operators
- ⌛ GPU support for assembly and $\mathcal H^2$ matrix operations
- ⌛ $\mathcal H^2$-matrix algebra, including compressed matrix products and direct
  solvers

### Directional $\mathcal H^2$ Construction

- ✔ Mixed low- and high-frequency admissibility
- ✔ Geometry-adaptive direction sets for k-means bounding-ball trees
- ✔ Hierarchical direction sets for $2^N$-trees
- ✔ Bottom-up wideband NCA with IACA and tree-mimicry pivoting
- ✔ Orientation-aware pivoting for EFIE matrices
- ⌛ Convenience `assemble` interface for point kernels and BEAST operators
- ⌛ GPU support for assembly and $\mathcal H^2$ matrix operations
- ⌛ $\mathcal H^2$-matrix algebra, including compressed matrix products and direct
  solvers


### Kernels and Applications

- ✔ Point-kernel matrices, including Helmholtz kernels
- ✔ BEAST integration tested for the EFIE single-layer operator
- ✔ Calderón-preconditioned EFIE example with Raviart--Thomas and
  Buffa--Christiansen spaces
- ⌛ Validation and examples for double-layer and additional boundary integral
  operators

## Installation

Enter the Julia package manager by pressing `]` at the REPL and run:

```
pkg> add https://github.com/JoshuaTetzner/NestedCrossApproximation.jl.git
```

The recommended geometry-adaptive k-means tree backend is optional. Install it
for the example below with

```
pkg> add ParallelKMeans
```

## $\mathcal H^2$-Matrices for Point Kernels

The package does not export any names, so its types are imported explicitly. The
following example constructs a geometry-adaptive k-means tree and compresses a
Helmholtz point-kernel matrix:

```julia
using NestedCrossApproximation: PetrovGalerkinNCA, TopDown, storage
using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using ParallelKMeans
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

builder = KMeansTreeBuilder(; numberofclusters=2, minvalues=32)
tree = BlockTree(
    KMeansTree(testpoints; builder=builder),
    KMeansTree(trialpoints; builder=builder),
)

kernel = HelmholtzKernel(2pi / 10)
h2mat = PetrovGalerkinNCA(
    kernel,
    testpoints,
    trialpoints,
    tree;
    testcompressor=TopDown(; factorization=ACA(; tol=1e-4)),
    trialcompressor=TopDown(; factorization=ACA(; tol=1e-4)),
)

x = ones(ComplexF64, size(h2mat, 2))
y = h2mat * x
storage(h2mat)
```

The result is a `LinearMap` supporting `*`, `mul!`, `transpose`, and `adjoint`.
See the [$\mathcal H^2$ manual](https://JoshuaTetzner.github.io/NestedCrossApproximation.jl/dev/manual/h2matrix/)
for bottom-up and batched compressors, representors, admissibility parameters, and
alternative tree backends.

## Directional $\mathcal H^2$-Matrices and BEAST

Wideband compression uses `isnearwideband` and the `BottomUp` compressor. For
geometry-adaptive k-means trees, the number of directions is determined separately
for every cluster from its electrical size. The implementation is currently tested
with Helmholtz point kernels and BEAST EFIE single-layer operators.

The
[directional manual](https://JoshuaTetzner.github.io/NestedCrossApproximation.jl/dev/manual/dh2matrix/)
contains a complete point-kernel example. The
[application examples](https://JoshuaTetzner.github.io/NestedCrossApproximation.jl/dev/manual/examples/)
show the explicit BEAST construction for a Calderón-preconditioned EFIE system.

## Far-Field Representors

The `TopDown` and `BottomUp` compressors can reduce a cluster's complete far field
before applying ACA or IACA:

- `RandomRep` samples the far field uniformly.
- `ChebyshevRep` selects three-dimensional points near a tensor Chebyshev grid fitted
  to the principal axes of the far field.
- `MimicryRep` selects representatives geometrically with tree-mimicry pivoting and
  is available with `BottomUp`.

Representors reduce matrix-entry evaluations at the cost of testing convergence on
a smaller interaction block. Configuration examples and the compatibility rules are
given in the
[$\mathcal H^2$ manual](https://JoshuaTetzner.github.io/NestedCrossApproximation.jl/dev/manual/h2matrix/).

## Related Packages

- [AdaptiveCrossApproximation.jl](https://github.com/JoshuaTetzner/AdaptiveCrossApproximation.jl)
  provides ACA, IACA, pivoting strategies, convergence criteria, and kernel-matrix
  adapters.
- [H2Trees.jl](https://github.com/djukic14/H2Trees.jl) provides the cluster trees.
- [BEAST.jl](https://github.com/krcools/BEAST.jl) provides boundary integral
  operators, spaces, and quadrature.
- [BlockSparseMatrices.jl](https://github.com/djukic14/BlockSparseMatrices.jl)
  stores and applies the dense near field.

## References

- [1] Bebendorf, M., and R. Venn. *Constructing Nested Bases Approximations from
  the Entries of Non-Local Operators.* Numerische Mathematik 121, no. 4 (2012):
  609–35. [https://doi.org/10.1007/s00211-012-0449-9](https://doi.org/10.1007/s00211-012-0449-9).
- [2] Bebendorf, M., C. Kuske, and R. Venn. *Wideband Nested Cross Approximation
  for Helmholtz Problems.* Numerische Mathematik 130, no. 1 (2015): 1–34.
  [https://doi.org/10.1007/s00211-014-0656-7](https://doi.org/10.1007/s00211-014-0656-7).
- [3] Börm, Steffen. *Directional $\mathcal H^2$-Matrix Compression for
  High-Frequency Problems.* Numerical Linear Algebra with Applications 24, no. 6
  (2017): e2112. [https://doi.org/10.1002/nla.2112](https://doi.org/10.1002/nla.2112).
- [4] Tetzner, Joshua M., and Simon B. Adrian. *The Incomplete Adaptive Cross
  Approximation for the Fast Construction of $\mathcal H^2$-Matrices and Its
  Application to the Electric Field Integral Equation for Electrically Small
  Problems.* IEEE Transactions on Antennas and Propagation 74, no. 8 (2026):
  7835–47. [https://doi.org/10.1109/TAP.2026.3687274](https://doi.org/10.1109/TAP.2026.3687274).
- [5] Tetzner, Joshua M., and Simon B. Adrian. *Wideband Directional
  $\mathcal H^2$-Matrix Compression for the Electric Field Integral Equation with
  Geometry-Adaptive Cluster Trees.* Preprint, arXiv:2609.19431, 2026.
  [https://doi.org/10.48550/arXiv.2609.19431](https://doi.org/10.48550/arXiv.2609.19431).
