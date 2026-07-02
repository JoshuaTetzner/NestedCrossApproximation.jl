using BEAST
using CompScienceMeshes
using ParallelKMeans
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using Random
using LinearAlgebra
using OhMyThreads
using Test
using StaticArrays
Random.seed!(1)
##
h = 0.025
λ = 0.25
k = 2 * pi / λ

Γ = meshsphere(1.0, h)
RT = raviartthomas(Γ)
println("Size RT ", length(RT))

op = Maxwell3D.singlelayer(; wavenumber=k)
testtree = KMeansTree(RT.pos, 2; minvalues=100)
trialtree = KMeansTree(RT.pos, 2; minvalues=100)
#testtree = TwoNTree(RT.pos, 2 / 2^10; minvalues=200)
#trialtree = TwoNTree(RT.pos, 2 / 2^10; minvalues=200)
tree = H2Trees.BlockTree(testtree, trialtree)
isnear = NestedCrossApproximation.isnearwideband(k; ηhf=5)

## topdown + ACA
tol = 1e-5
@time h2mat_mimicry = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    RT,
    RT,
    tree;
    isnear=isnear,
    testcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            MaximumValue(),
            MimicryPivoting(RT.pos, RT.pos),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    trialcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            MimicryPivoting(RT.pos, RT.pos),
            MaximumValue(),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    scheduler=DynamicScheduler(),
);

@time h2mat_treemimicry = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    RT,
    RT,
    tree;
    isnear=isnear,
    testcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            MaximumValue(),
            TreeMimicryPivoting(RT.pos, RT.pos, trialtree),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    trialcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            TreeMimicryPivoting(RT.pos, RT.pos, testtree),
            MaximumValue(),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    scheduler=DynamicScheduler(),
);
##
x = rand(ComplexF64, length(RT))
A = assemble(op, RT, RT);

norm(h2mat_mimicry * x - A * x) / norm(A * x)
norm(h2mat_treemimicry * x - A * x) / norm(A * x)
##

size.(h2mat_mimicry.couplingmatrices.blocks)
