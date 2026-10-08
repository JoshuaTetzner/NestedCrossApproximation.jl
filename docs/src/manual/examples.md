# [Application Examples](@id examples)

NestedCrossApproximation.jl currently has no convenience `assemble` interface.
The block tree, pivoting strategies, compressors, and
[`PetrovGalerkinNCA`](@ref NestedCrossApproximation.PetrovGalerkinNCA) construction
must be supplied explicitly. The example below keeps these steps visible and only
wraps them in a local function because the same construction is needed for the
primal and dual operators.

The tested BEAST integration currently covers the EFIE single-layer operator.
Double-layer operators are not currently part of the tested
NestedCrossApproximation workflow.

## Calderón-Preconditioned EFIE

The electric field integral equation is a first-kind equation. For an iterative
solution, this example applies a multiplicative Calderón preconditioner. Let $X$ be
the Raviart--Thomas space, $Y$ the Buffa--Christiansen space, and $N_{XY}$ their
duality matrix. The compressed primal and dual EFIE matrices $T_{XX}$ and $T_{YY}$
form the preconditioner

```math
P = N_{XY}^{-\mathrm{T}} T_{YY} N_{XY}^{-1}.
```

The resulting system is $P T_{XX} u = P e$. Both single-layer matrices are stored
as directional $\mathcal{H}^2$-matrices.

The sphere below has several thousand unknowns and is resolved with approximately
ten elements per wavelength. Unlike a very small demonstration mesh, it produces a
nonempty directional far field and meaningful compression.

```julia
using NestedCrossApproximation: PetrovGalerkinNCA, BottomUp
using NestedCrossApproximation: isnearwideband, storage
using AdaptiveCrossApproximation
using BEAST
using CompScienceMeshes
using H2Trees
using LinearAlgebra
using ParallelKMeans
using Random
using StaticArrays

h = 0.08
Γ = meshsphere(1.0, h)
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)

λ = 10h
k = 2pi / λ
T = Maxwell3D.singlelayer(; wavenumber=k)

E = Maxwell3D.planewave(;
    direction=ẑ,
    polarization=x̂,
    wavenumber=k,
)
e = assemble((n × E) × n, X)
```

### Directional Compression

For the EFIE, geometric positions alone do not represent the vector basis functions
reliably. `EFIEDirectionalFilter` adds their edge and surface-normal orientations to
tree-mimicry pivoting [[5]](@ref refs). The filter advances through orientation
groups in phases and therefore uses `PhaseExtrapolator` as its convergence
criterion.

The following helper is local to this example; it is not a package-level
convenience interface.

```julia
function efie_nca(
    operator,
    space,
    k;
    tol=1e-3,
    maxrank=80,
    minvalues=100,
    seed=1,
)
    # Independent builders with the same seed produce matching test and trial trees.
    testbuilder = KMeansTreeBuilder(;
        numberofclusters=2,
        minvalues=minvalues,
        splitterkwargs=(; rng=MersenneTwister(seed)),
    )
    trialbuilder = KMeansTreeBuilder(;
        numberofclusters=2,
        minvalues=minvalues,
        splitterkwargs=(; rng=MersenneTwister(seed)),
    )
    tree = BlockTree(
        KMeansTree(space.pos; builder=testbuilder),
        KMeansTree(space.pos; builder=trialbuilder),
    )

    testtree = H2Trees.testtree(tree)
    trialtree = H2Trees.trialtree(tree)

    testpivoting = TreeMimicryPivoting(
        space.pos,
        space.pos,
        trialtree,
        EFIEDirectionalFilter(space, trialtree),
    )
    trialpivoting = TreeMimicryPivoting(
        space.pos,
        space.pos,
        testtree,
        EFIEDirectionalFilter(space, testtree),
    )
    convergence = PhaseExtrapolator(tol)

    return PetrovGalerkinNCA(
        operator,
        space,
        space,
        tree;
        isnear=isnearwideband(k; ηhf=1.0, γ=1.0),
        maxrank=maxrank,
        testcompressor=BottomUp(;
            factorization=IACA(
                MaximumValue(),
                testpivoting,
                convergence,
            ),
        ),
        trialcompressor=BottomUp(;
            factorization=IACA(
                trialpivoting,
                MaximumValue(),
                convergence,
            ),
        ),
        verbose=true,
    )
end

Txx = efie_nca(T, X, k)
Tyy = efie_nca(T, Y, k)

primal_storage_gb = storage(Txx)
dual_storage_gb = storage(Tyy)
```

`Txx` and `Tyy` use separate block trees because the RT and BC spaces have
different positions. The two cluster trees inside each block tree use the same seed
and splitting parameters so that the Galerkin test and trial sides are partitioned
consistently.

### Calderón-Preconditioned Solve

The surface-cross-product operator supplies the duality matrix. Its inverse and
transposed inverse are applied iteratively; they remain linear maps and can therefore
be composed directly with the compressed matrices.

```julia
Nxy = assemble(BEAST.NCross(), X, Y)
Nxy_inverse = BEAST.GMRESSolver(
    Nxy;
    reltol=1e-8,
    maxiter=1500,
    restart=1500,
    verbose=false,
)
Nxy_transpose_inverse = BEAST.GMRESSolver(
    transpose(Nxy);
    reltol=1e-8,
    maxiter=1500,
    restart=1500,
    verbose=false,
)

P = Nxy_transpose_inverse * Tyy * Nxy_inverse

u, stats = solve(
    BEAST.GMRES(P * Txx; rtol=1e-4, memory=50),
    P * e,
)
@assert stats.solved "Calderón-preconditioned GMRES did not converge"
```

The solution coefficients can be used with the ordinary BEAST postprocessing
functions. For example, a bistatic far-field cut is evaluated by

```julia
angles = range(0, pi; length=181)
directions = [point(sin(θ), 0, cos(θ)) for θ in angles]
farfield = potential(
    MWFarField3D(; wavenumber=k),
    directions,
    u,
    X,
)
```
