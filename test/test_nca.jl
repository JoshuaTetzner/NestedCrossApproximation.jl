using BEAST
using CompScienceMeshes
using ParallelKMeans
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using Random
using LinearAlgebra
using OhMyThreads
using StaticArrays

Random.seed!(1)
##
λ = 4.0
k = 2 * pi / λ

Γ = meshicosphere(40, 1.0)
Γ1 = Γ# meshrectangle(1.0, 1.0, 0.08)
Γ2 = Γ# translate(Γ1, SVector(0.0, 0.0, 1.925))
tRT = raviartthomas(Γ1)
sRT = raviartthomas(Γ2)
println("Size RT ", length(tRT))

op = Maxwell3D.singlelayer(; wavenumber=k)

tree = H2Trees.BlockTree(
    KMeansTree(tRT.pos, 2; minvalues=150), KMeansTree(sRT.pos, 2; minvalues=150)
)
##
x = rand(ComplexF64, length(sRT))
A = assemble(op, tRT, sRT);
##
tfar, sfar = NestedCrossApproximation.fardata(tree, NestedCrossApproximation.isnear())
NestedCrossApproximation.fars(tfar)
NestedCrossApproximation.farptr(tfar)

for level in H2Trees.levels(tree.testcluster)
    println("Level ", level)
    for t in H2Trees.LevelIterator(tree.testcluster, level)
        println("Test cluster ", t)
        println("Fars: ", NestedCrossApproximation.fars(tfar, t))
    end
end

## topdown + ACA
@time hmat_topdown_aca = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    tRT,
    sRT,
    tree;
    testcompressor=NestedCrossApproximation.TopDown(; factorization=ACA()),
    trialcompressor=NestedCrossApproximation.TopDown(; factorization=ACA()),
);
##
y_topdown_aca = hmat_topdown_aca * x;
println(norm(A * x - y_topdown_aca) / norm(A * x))
## topdown + iACA mimicrypivoting
tol = 1.0e-3
@time hmat_topdown_iaca_mimicry = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    tRT,
    sRT,
    tree;
    testcompressor=NestedCrossApproximation.TopDown(;
        factorization=iACA(
            MaximumValue(),
            MimicryPivoting(tRT.pos, sRT.pos),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    trialcompressor=NestedCrossApproximation.TopDown(;
        factorization=iACA(
            MimicryPivoting(sRT.pos, tRT.pos),
            MaximumValue(),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    scheduler=DynamicScheduler(),
);
y_topdown_iaca_mimicry = hmat_topdown_iaca_mimicry * x;
#println(norm(A * x - y_topdown_iaca_mimicry) / norm(A * x))
## bottomup + mimicrypivoting
tol = 1.0e-3
@time hmat_bottomup_mimicry = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    tRT,
    sRT,
    tree;
    testcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            MaximumValue(),
            MimicryPivoting(tRT.pos, sRT.pos),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    trialcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            MimicryPivoting(sRT.pos, tRT.pos),
            MaximumValue(),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
);
y_bottomup_mimicry = hmat_bottomup_mimicry * x;
#println(norm(A * x - y_bottomup_mimicry) / norm(A * x))
## bottomup + treemimicrypivoting
@time hmat_bottomup_treemimicry = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    tRT,
    sRT,
    tree;
    testcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            MaximumValue(),
            TreeMimicryPivoting(tRT.pos, sRT.pos, H2Trees.trialtree(tree)),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    trialcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            TreeMimicryPivoting(sRT.pos, tRT.pos, H2Trees.testtree(tree)),
            MaximumValue(),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
);
y_bottomup_treemimicry = hmat_bottomup_treemimicry * x;
println(norm(A * x - y_bottomup_treemimicry) / norm(A * x))
