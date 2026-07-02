# BottomUp Compression

## Algorithm

The BottomUp compressor builds nested bases by traversing the cluster tree from leaves to root.
At each leaf it runs the incomplete ACA (iACA) against the leaf's far-field index set to select a local pivot set.
When moving up to a parent cluster, the pivots of all children are collected and passed to the next iACA run, propagating pivot information upward and enforcing the nesting condition.

This bottom-up propagation allows the algorithm to reuse pivot information across levels efficiently: each non-leaf cluster's iACA run starts from a good initial pivot set inherited from its children, typically converging in fewer iterations.

## Usage

```julia
using NestedCrossApproximation
using AdaptiveCrossApproximation

tol = 1e-4
testcompressor = BottomUp(;
    factorization=iACA(
        MaximumValue(),
        TreeMimicryPivoting(testspace.pos, trialspace.pos, H2Trees.trialtree(tree)),
        FNormExtrapolator(iFNormEstimator(tol)),
    ),
)
```

The `TreeMimicryPivoting` strategy selects trial-side pivot candidates by mimicking the geometric structure of the trial tree, which is well-suited to the nested structure of $\mathcal{H}^2$-matrices.

## Cost

In the low-frequency regime, each active cluster costs approximately one coupling-matrix equivalent, so the total cost scales like the number of admissible leaf pairs — $\mathcal{O}(N k)$.

## References

See the [References](@ref refs) section on the Introduction page.
