"""
    NestedCrossApproximation

Nested cross approximation (NCA) for the kernel-independent construction of H²-matrices.

The admissible blocks of a matrix are compressed into nested cluster bases, transfer
matrices and coupling matrices, all of which are built from matrix entries only.
Key features:

  - **H²-matrices**: Dense near field and nested far field with a fast matrix-vector product
  - **Compressors**: Top-down, bottom-up and batched bottom-up construction of the nested bases
  - **Factorizations**: The ACA and the IACA of AdaptiveCrossApproximation.jl select the pivots
  - **Representors**: Optional reduction of the far field before the pivot search
  - **Directional H²-matrices**: Wideband admissibility with a directional subdivision of the
    far field for electrically large clusters

The package does not export any names.

# Main API

**H²-matrix:**

  - [`PetrovGalerkinNCA`](@ref): H²-matrix type and its assembly
  - [`farmatrix`](@ref), [`nearmatrix`](@ref): Extract the far and the near field
  - [`storage`](@ref): Report the memory footprint

**Compressors:**

  - [`TopDown`](@ref): Pivots are passed from the root to the leaves
  - [`BottomUp`](@ref): Pivots are passed from the leaves to the root
  - [`BatchedBottomUp`](@ref): Bottom-up variant that evaluates one batch of blocks per level

**Representors:**

  - [`RandomRep`](@ref): Random subset of the far field
  - [`ChebyshevRep`](@ref): Points close to a Chebyshev grid around the far field
  - [`MimicryRep`](@ref): Geometric selection by tree mimicry pivoting

**Admissibility:**

  - `AdaptiveCrossApproximation.isnear`: Standard admissibility condition
  - [`isnearwideband`](@ref): Wideband condition, leads to a directional H²-matrix

# Example

```julia
using NestedCrossApproximation: PetrovGalerkinNCA, TopDown
using AdaptiveCrossApproximation, H2Trees, ParallelKMeans

builder = KMeansTreeBuilder(; minvalues=100)
tree = BlockTree(
    KMeansTree(testpoints; builder=builder), KMeansTree(trialpoints; builder=builder)
)
h2mat = PetrovGalerkinNCA(
    operator,
    testspace,
    trialspace,
    tree;
    testcompressor=TopDown(; factorization=ACA(; tol=1e-4)),
    trialcompressor=TopDown(; factorization=ACA(; tol=1e-4)),
)
y = h2mat * x
```

# See also

  - AdaptiveCrossApproximation.jl for the low-rank factorizations, pivoting strategies and
    kernel matrices
  - H2Trees.jl for the cluster trees
  - BlockSparseMatrices.jl for the storage of the near field
"""
module NestedCrossApproximation

using AdaptiveCrossApproximation
using BlockSparseMatrices
using H2Trees
using LinearMaps
using LinearAlgebra
using OhMyThreads
using ProgressMeter
using StaticArrays

include("admissibility/nearinteractions.jl")
include("admissibility/farinteractions.jl")

include("admissibility/directionalsubdivision/abstractdirections.jl")
include("admissibility/directionalsubdivision/twondirections.jl")
include("admissibility/directionalsubdivision/boundingballdirections.jl")

include("utils/utils.jl")
include("nestedbasis.jl")
include("transfermatrices.jl")

include("compressors/abstractcompressor.jl")
include("compressors/topdown.jl")
include("compressors/bottomup.jl")
include("compressors/batchedbottomup.jl")
include("compressors/buffer.jl")
include("utils/workload.jl")

include("pivoting/representor.jl")
include("pivoting/pivots.jl")

include("couplingmatrices.jl")

include("nca/abstractnca.jl")
include("nca/aggregate.jl")
include("nca/couple.jl")
include("nca/disaggregate.jl")
include("nca/petrovgalerkinnca.jl")
end
