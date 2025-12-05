using BEAST
using CompScienceMeshes
using NestedCrossApproximation
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using LinearAlgebra
##

λ = 2
k = 2 * π / λ
Γ = meshicosphere(40, 1.0)
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
length(space)
##
ttree = H2Trees.KMeansTree(space.pos, 2; minvalues=100)
tree = BlockTree(ttree, ttree)

function myisnear(treea, treeb, nodea, nodeb; η=1.0)
    ths = H2Trees.radius(treea, nodea)
    shs = H2Trees.radius(treeb, nodeb)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)

    (2 * max(ths, shs) <= η * max(dist, 0.0)) ? (return false) : (return true)
end
##
testcompressor = NestedCrossApproximation.BottomUpCompressor(
    iACA(
        MaximumValue(),
        TreeMimicryPivoting(space.pos, space.pos, tree.trialcluster),
        FNormExtrapolator(iFNormEstimator(1e-4)),
    ),
    nothing,
)
trialcompressor = NestedCrossApproximation.BottomUpCompressor(
    iACA(
        TreeMimicryPivoting(space.pos, space.pos, tree.testcluster),
        MaximumValue(),
        FNormExtrapolator(iFNormEstimator(1e-4)),
    ),
    nothing,
)

@time h2 = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    space,
    space,
    tree;
    testcompressor=testcompressor,
    trialcompressor=trialcompressor,
    ntasks=Threads.nthreads(),
    isnear=myisnear,
);

##
x = rand(ComplexF64, size(h2, 2))
@time h2 * x;
@time A * x;
##
A = assemble(op, space, space)

norm(h2 * x - A * x) / norm(A * x)
norm(transpose(h2) * x - transpose(A) * x) / norm(transpose(A) * x)
norm(adjoint(h2) * x - adjoint(A) * x) / norm(adjoint(A) * x)
