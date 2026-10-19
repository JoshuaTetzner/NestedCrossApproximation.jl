# Application Examples

## EFIE — Maxwell single-layer operator

This example assembles an $\mathcal{H}^2$-matrix approximation of the electric field integral equation (EFIE) singular-layer operator on a sphere mesh and verifies the result against the dense reference.

```julia
using BEAST
using CompScienceMeshes
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using ParallelKMeans
using OhMyThreads
using LinearAlgebra
using Random

Γ = meshsphere(1.0, 0.025)
RT = raviartthomas(Γ)

h = # minimum edge length of Γ
λ = 10h
k = 2π / λ
op = Maxwell3D.singlelayer(; wavenumber=k)

Random.seed!(1)
testtree  = KMeansTree(RT.pos, 2; minvalues=100)
Random.seed!(1)
trialtree = KMeansTree(RT.pos, 2; minvalues=100)
tree = H2Trees.BlockTree(testtree, trialtree)

tol = 1e-3

h2mat = PetrovGalerkinNCA(
    op, RT, RT, tree;
    testcompressor = TopDown(; factorization=ACA()),
    trialcompressor = TopDown(; factorization=ACA()),
    scheduler = DynamicScheduler(),
    verbose = true,
)

A = assemble(op, RT, RT)
x  = rand(ComplexF64, length(RT))
xt = rand(ComplexF64, length(RT))
println("forward error:    ", norm(h2mat * x  - A * x)  / norm(A * x))
println("transpose error:  ", norm(transpose(h2mat) * xt - transpose(A) * xt) / norm(transpose(A) * xt))
```

## Wideband EFIE

For problems where $k$ is large enough that some interactions are in the high-frequency regime, use the wideband admissibility condition together with BottomUp compression.

```julia
isnear = isnearwideband(k; ηhf=5.0, γ=1.0)

h2mat_wb = PetrovGalerkinNCA(
    op, RT, RT, tree;
    isnear = isnear,
    testcompressor = BottomUp(;
        factorization=iACA(
            MaximumValue(),
            TreeMimicryPivoting(RT.pos, RT.pos, H2Trees.trialtree(tree)),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    trialcompressor = BottomUp(;
        factorization=iACA(
            TreeMimicryPivoting(RT.pos, RT.pos, H2Trees.testtree(tree)),
            MaximumValue(),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    scheduler = DynamicScheduler(),
    verbose = true,
)
```
