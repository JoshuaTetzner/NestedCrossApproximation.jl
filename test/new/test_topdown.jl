using BEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using NestedCrossApproximation
using CompScienceMeshes
using LinearAlgebra
##

λ = 0.6
k = 2 * pi / λ
Γ = meshsphere(1.0, 0.03)
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
length(space)

ttree = H2Trees.KMeansTree(space.pos, 2; minvalues=100)
tree = BlockTree(ttree, ttree)

function myisnear(treea, treeb, nodea, nodeb; η=1.0)
    ths = H2Trees.radius(treea, nodea)
    shs = H2Trees.radius(treeb, nodeb)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)

    (2 * max(ths, shs) <= η * max(dist, 0.0)) ? (return false) : (return true)
end
##
testcompressor = NestedCrossApproximation.TopDownCompressor(
    iACA(
        MaximumValue(),
        MimicryPivoting(space.pos, space.pos),
        FNormExtrapolator(iFNormEstimator(1e-4)),
    ),
    nothing,
)
trialcompressor = NestedCrossApproximation.TopDownCompressor(
    iACA(
        MimicryPivoting(space.pos, space.pos),
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
iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> myisnear(tree))(tree)
##
x = rand(ComplexF64, size(h2, 2))
h2 * x
A = assemble(op, space, space)

norm(h2 * x - A * x) / norm(A * x)
##

##
y = zeros(ComplexF64, size(h2, 2))
@time LinearAlgebra.mul!(y, h2, x);

y2 = zeros(ComplexF64, size(h2, 2))
@time NestedCrossApproximation.mul2(y2, h2, x);
using Base.Threads
@threads for x in pairs(h2.nestedtestbases)
    println("x")
end
