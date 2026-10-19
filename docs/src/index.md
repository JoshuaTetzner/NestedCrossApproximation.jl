# NestedCrossApproximation.jl

NestedCrossApproximation.jl provides construction of $\mathcal{H}^2$-matrices via the Nested Cross Approximation (NCA).
$\mathcal{H}^2$-matrices are a data-sparse representation of kernel matrices arising in boundary element methods (BEM).
By exploiting the smooth, far-field decay of the underlying kernel, interactions between well-separated clusters of degrees of freedom are compressed into nested low-rank factors, reducing both storage and matrix-vector product complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$ or $\mathcal{O}(N \log N)$.

The package supports two complementary compression strategies and extends to the wideband regime via directional admissibility:

- **TopDown** — constructs the nested bases from root to leaves using the [AdaptiveCrossApproximation.jl](https://github.com/FastBEAST/AdaptiveCrossApproximation.jl) (ACA) algorithm.
- **BottomUp** — constructs the nested bases from leaves to root using the incomplete ACA (iACA), enabling efficient reuse of pivot information across levels.
- **Wideband / directional** — extends BottomUp to mixed low-frequency / high-frequency problems via a directional subdivision of far interactions, keeping the $\mathcal{O}(N \log N)$ complexity in the high-frequency regime.

## Installation

```
pkg> add https://github.com/JoshuaTetzner/NestedCrossApproximation.jl.git
```

## Quick start

```julia
using NestedCrossApproximation
using AdaptiveCrossApproximation
using H2Trees

# Build a cluster tree and block tree
testtree  = KMeansTree(testpoints,  2; minvalues=100)
trialtree = KMeansTree(trialpoints, 2; minvalues=100)
tree = H2Trees.BlockTree(testtree, trialtree)

# Assemble an H2-matrix with TopDown + ACA
h2mat = PetrovGalerkinNCA(
    op, testspace, trialspace, tree;
    testcompressor  = TopDown(; factorization=ACA()),
    trialcompressor = TopDown(; factorization=ACA()),
)
```

See the [Manual](@ref) for a complete worked example.

## [References](@id refs)

- [1] Hackbusch, W., and S. Börm. *Data-Sparse Approximation by Adaptive $\mathcal{H}^2$-Matrices.* Computing 69, no. 1 (2002): 1–35. <https://doi.org/10.1007/s00607-002-1450-4>.
- [2] Bebendorf, Mario. *Adaptive Cross Approximation of Multidimensional Arrays.* Computing 70, no. 1 (2003): 1–24. <https://doi.org/10.1007/s00607-002-0019-1>.
- [3] Bauer, M., M. Bebendorf, and B. Feist. *Kernel-Independent Adaptive Construction of $\mathcal{H}^2$-Matrix Approximations.* Numerische Mathematik 150, no. 1 (2022): 1–32. <https://doi.org/10.1007/s00211-021-01255-y>.
- [4] Tetzner, Joshua M., and Simon B. Adrian. *The Incomplete Adaptive Cross Approximation for the Fast Construction of $\mathcal{H}^2$-Matrices.* Preprint, 2025. <https://doi.org/10.36227/techrxiv.176281137.74736897/v1>.
