# NestedCrossApproximation.jl

This package constructs $\mathcal{H}^2$-matrices with the nested cross approximation (NCA) [[1]](@ref refs).
An $\mathcal{H}^2$-matrix is a data-sparse representation of the matrices arising in boundary element methods: interactions between well-separated clusters are compressed into nested low-rank factors, which reduces storage and the cost of a matrix-vector product from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$ in the low-frequency regime.

The nested bases are built purely from matrix entries, so the package is kernel independent. It provides

- three compressors for the nested bases: [`TopDown`](@ref NestedCrossApproximation.TopDown), [`BottomUp`](@ref NestedCrossApproximation.BottomUp), and [`BatchedBottomUp`](@ref NestedCrossApproximation.BatchedBottomUp),
- the adaptive cross approximation (ACA) and the incomplete ACA (IACA) [[4]](@ref refs) of [AdaptiveCrossApproximation.jl](https://github.com/JoshuaTetzner/AdaptiveCrossApproximation.jl) as the low-rank factorizations inside the compressors,
- optional [far-field representors](@ref representors) based on random sampling, a
  Chebyshev grid, or tree-mimicry pivoting,
- a wideband extension based on a directional subdivision of the far field [[2, 3, 5]](@ref refs) for problems that contain electrically large clusters.

## Installation

Installing NestedCrossApproximation is done by entering the package manager (enter `]` at the julia REPL) and issuing:

```
pkg> add https://github.com/JoshuaTetzner/NestedCrossApproximation.jl.git
```

## Quick start

The package does not export any names; import what you need explicitly.

```julia
using NestedCrossApproximation: PetrovGalerkinNCA, TopDown
using AdaptiveCrossApproximation
using H2Trees
using ParallelKMeans

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

The [manual](@ref manual) walks through a complete example.

## [References](@id refs)

- [1] Bebendorf, M., and R. Venn. *Constructing Nested Bases Approximations from the Entries of Non-Local Operators.* Numerische Mathematik 121, no. 4 (2012): 609–35. [https://doi.org/10.1007/s00211-012-0449-9](https://doi.org/10.1007/s00211-012-0449-9).
- [2] Bebendorf, M., C. Kuske, and R. Venn. *Wideband Nested Cross Approximation for Helmholtz Problems.* Numerische Mathematik 130, no. 1 (2015): 1–34. [https://doi.org/10.1007/s00211-014-0656-7](https://doi.org/10.1007/s00211-014-0656-7).
- [3] Börm, Steffen. *Directional $\mathcal{H}^2$-Matrix Compression for High-Frequency Problems.* Numerical Linear Algebra with Applications 24, no. 6 (2017): e2112. [https://doi.org/10.1002/nla.2112](https://doi.org/10.1002/nla.2112).
- [4] Tetzner, Joshua M., and Simon B. Adrian. *The Incomplete Adaptive Cross Approximation for the Fast Construction of $\mathcal{H}^2$-Matrices and Its Application to the Electric Field Integral Equation for Electrically Small Problems.* IEEE Transactions on Antennas and Propagation 74, no. 8 (2026): 7835–47. [https://doi.org/10.1109/TAP.2026.3687274](https://doi.org/10.1109/TAP.2026.3687274).
- [5] Tetzner, Joshua M., and Simon B. Adrian. *Wideband Directional $\mathcal{H}^2$-Matrix Compression for the Electric Field Integral Equation with Geometry-Adaptive Cluster Trees.* Preprint, arXiv:2609.19431, 2026. [https://doi.org/10.48550/arXiv.2609.19431](https://doi.org/10.48550/arXiv.2609.19431).
