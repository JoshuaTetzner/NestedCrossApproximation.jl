# TopDown Compression

## Algorithm

The TopDown compressor builds nested bases by traversing the cluster tree from root to leaves.
At each level it collects the far-field index sets of the current cluster *and* inherits the pivot indices selected by its parent.
The enlarged index set is then compressed with the [Adaptive Cross Approximation](https://github.com/FastBEAST/AdaptiveCrossApproximation.jl) (ACA) to produce the local pivot set and the local basis.

Traversing top-down ensures that the nesting condition is satisfied automatically: the column space of a child's basis is a subspace of the column space computed at the parent.

## Usage

```julia
using NestedCrossApproximation
using AdaptiveCrossApproximation

compressor = TopDown(; factorization=ACA())
```

The `factorization` keyword accepts any ACA variant from AdaptiveCrossApproximation.jl.
The `representor` keyword (optional) controls the representor used to build the leaf bases from the pivot set.

## Cost

In the low-frequency regime, each active level of the cluster tree costs roughly $|\text{leaves}(T)|$ kernel evaluations, so the total cost of the bases construction is $\mathcal{O}(N k \log N)$ where $k$ is the local rank.

## References

See the [References](@ref refs) section on the Introduction page.
