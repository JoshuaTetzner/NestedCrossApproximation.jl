using BEAST
using CompScienceMeshes
using ParallelKMeans
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using Random
using LinearAlgebra
using OhMyThreads

Random.seed!(1)
##
λ = 1.0
k = 2 * pi / λ

Γ = meshicosphere(100, 1.0)
RT = raviartthomas(Γ)
op = Maxwell3D.singlelayer(; wavenumber=k)

tree = H2Trees.BlockTree(
    KMeansTree(RT.pos, 2; minvalues=100), KMeansTree(RT.pos, 2; minvalues=100)
)
##
x = rand(ComplexF64, length(RT))
A = assemble(op, RT, RT);
## topdown + ACA
@time hmat_topdown_aca = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    RT,
    RT,
    tree;
    testcompressor=NestedCrossApproximation.TopDown(; factorization=ACA()),
    trialcompressor=NestedCrossApproximation.TopDown(; factorization=ACA()),
);
y_topdown_aca = hmat_topdown_aca * x;
println(norm(A * x - y_topdown_aca) / norm(A * x))
## topdown + iACA mimicrypivoting
@time hmat_topdown_iaca_mimicry = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    RT,
    RT,
    tree;
    testcompressor=NestedCrossApproximation.TopDown(;
        factorization=iACA(
            MaximumValue(),
            MimicryPivoting(RT.pos, RT.pos),
            FNormExtrapolator(iFNormEstimator(1.0e-3)),
        ),
    ),
    trialcompressor=NestedCrossApproximation.TopDown(;
        factorization=iACA(
            MimicryPivoting(RT.pos, RT.pos),
            MaximumValue(),
            FNormExtrapolator(iFNormEstimator(1.0e-3)),
        ),
    ),
);
##
y_topdown_iaca_mimicry = hmat_topdown_iaca_mimicry * x;
println(norm(A * x - y_topdown_iaca_mimicry) / norm(A * x))
## bottomup + mimicrypivoting
@time hmat_bottomup_mimicry = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    RT,
    RT,
    tree;
    testcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            MaximumValue(),
            MimicryPivoting(RT.pos, RT.pos),
            FNormExtrapolator(iFNormEstimator(1.0e-4)),
        ),
    ),
    trialcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            MimicryPivoting(RT.pos, RT.pos),
            MaximumValue(),
            FNormExtrapolator(iFNormEstimator(1.0e-4)),
        ),
    ),
);
#y_bottomup_mimicry = hmat_bottomup_mimicry * x;
#println(norm(A * x - y_bottomup_mimicry) / norm(A * x))
## bottomup + treemimicrypivoting
@time hmat_bottomup_treemimicry = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    RT,
    RT,
    tree;
    testcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            MaximumValue(),
            TreeMimicryPivoting(RT.pos, RT.pos, H2Trees.trialtree(tree)),
            FNormExtrapolator(iFNormEstimator(1.0e-4)),
        ),
    ),
    trialcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            TreeMimicryPivoting(RT.pos, RT.pos, H2Trees.testtree(tree)),
            MaximumValue(),
            FNormExtrapolator(iFNormEstimator(1.0e-4)),
        ),
    ),
);
y_bottomup_treemimicry = hmat_bottomup_treemimicry * x;
println(norm(A * x - y_bottomup_treemimicry) / norm(A * x))
