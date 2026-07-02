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

##
h = 0.0005
Γ = meshcircle(1.0, h)
space = lagrangec0d1(Γ)
println("Size RT ", length(space))
λ = 10h
k = 2 * pi / λ
##
op = Helmholtz2D.singlelayer(; wavenumber=k)
Random.seed!(1)
testtree = KMeansTree(
    space.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
)
Random.seed!(1)
trialtree = KMeansTree(
    space.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
)
#testtree = TwoNTree(tRT.pos, 2 / 2^10; minvalues=200)
#trialtree = TwoNTree(sRT.pos, 2 / 2^10; minvalues=200)

tree = H2Trees.BlockTree(testtree, trialtree)
isnear = NestedCrossApproximation.isnearwideband(k; ηhf=5.0, γ=1.0);
##
quadstrat = BEAST.DoubleNumSauterQstrat(2, 3, 0, 4, 3, 4)#BEAST.defaultquadstrat(op, space, space)
farquadstrat = BEAST.DoubleNumQStrat(3, 3)
x = rand(ComplexF64, length(space))
@time A = assemble(op, space, space; quadstrat=quadstrat);

##
tol = 1e-3
@time h2mat = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    space,
    space,
    tree;
    isnear=isnear,
    testcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            MaximumValue(),
            TreeMimicryPivoting(space.pos, space.pos, H2Trees.trialtree(tree)),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    trialcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            TreeMimicryPivoting(space.pos, space.pos, H2Trees.testtree(tree)),
            MaximumValue(),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    matrixdata=quadstrat,
    farmatrixdata=farquadstrat,
    scheduler=DynamicScheduler(),
);
##
hmat = AdaptiveCrossApproximation.HMatrix(
    op,
    space,
    space,
    tree;
    isnear=isnear,
    maxrank=100,
    spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
    tol=1e-2 * tol,
    scheduler=DynamicScheduler(),
    nearmatrixdata=quadstrat,
    farmatrixdata=farquadstrat,
)
##
norm(h2mat * x - A * x) / norm(A * x)
xt = rand(ComplexF64, length(space))
norm(transpose(h2mat) * xt - transpose(A) * xt) / norm(transpose(A) * xt)
norm(adjoint(h2mat) * xt - adjoint(A) * xt) / norm(adjoint(A) * xt)
##
farh2mat = NestedCrossApproximation.farmatrix(h2mat)
farhmat = AdaptiveCrossApproximation.farmatrix(hmat)

norm(farhmat * x - farh2mat * x) / norm(farhmat * x)

##

estimate_reldifference(h2mat, A; tol=1e-4)
estimate_reldifference(hmat, A; tol=1e-4)
estimate_reldifference(h2mat, hmat; tol=1e-4)
estimate_reldifference(farh2mat, farhmat; tol=1e-4)
