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

# Assembles the same EFIE (Maxwell singlelayer) operator with three different
# NestedCrossApproximation compression paths and compares each against the dense
# reference matrix. Swap testcompressor/trialcompressor/isnear below to try other
# configurations.

function edgeinfo(m)
    edges = skeleton(m, 1)
    _edgelength = Float64[]
    for (_, e) in enumerate(edges)
        push!(_edgelength, norm(diff(vertices(chart(edges, e)))))
    end
    return minimum(_edgelength),
    sum(_edgelength) / length(_edgelength),
    maximum(_edgelength)
end

function comparematvec(h2mat, A, x, xt)
    println("forward:   ", norm(h2mat * x - A * x) / norm(A * x))
    println(
        "transpose: ",
        norm(transpose(h2mat) * xt - transpose(A) * xt) / norm(transpose(A) * xt),
    )
    return println(
        "adjoint:   ", norm(adjoint(h2mat) * xt - adjoint(A) * xt) / norm(adjoint(A) * xt)
    )
end

##
Γ = meshsphere(1.0, 0.025)
RT = raviartthomas(Γ)
println("Size RT ", length(RT))

h = edgeinfo(Γ)[3]
λ = 50h
k = 2 * pi / λ
op = Maxwell3D.singlelayer(; wavenumber=k)

Random.seed!(1)
testtree = KMeansTree(RT.pos, 2; minvalues=100)
Random.seed!(1)
trialtree = KMeansTree(RT.pos, 2; minvalues=100)
tree = H2Trees.BlockTree(testtree, trialtree)

x = rand(ComplexF64, length(RT))
xt = rand(ComplexF64, length(RT))
tol = 1e-3
##
@time A = assemble(op, RT, RT; threading=:cellcoloring);

## Path 1: TopDown + ACA
@time h2mat_topdown = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    RT,
    RT,
    tree;
    testcompressor=NestedCrossApproximation.TopDown(; factorization=ACA()),
    trialcompressor=NestedCrossApproximation.TopDown(; factorization=ACA()),
    scheduler=DynamicScheduler(),
    verbose=true,
);
comparematvec(h2mat_topdown, A, x, xt)

## Path 2: BottomUp + iACA (TreeMimicryPivoting)
@time h2mat_bottomup = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    RT,
    RT,
    tree;
    testcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            MaximumValue(),
            TreeMimicryPivoting(RT.pos, RT.pos, H2Trees.trialtree(tree)),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    trialcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            TreeMimicryPivoting(RT.pos, RT.pos, H2Trees.testtree(tree)),
            MaximumValue(),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    scheduler=DynamicScheduler(),
    verbose=true,
);
comparematvec(h2mat_bottomup, A, x, xt)

## Path 3: BottomUp + iACA (TreeMimicryPivoting), wideband (mixed LF/HF) admissibility
isnear = NestedCrossApproximation.isnearwideband(k; ηhf=5.0, γ=1.0)
@time h2mat_wideband = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    RT,
    RT,
    tree;
    isnear=isnear,
    testcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            MaximumValue(),
            TreeMimicryPivoting(RT.pos, RT.pos, H2Trees.trialtree(tree)),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    trialcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            TreeMimicryPivoting(RT.pos, RT.pos, H2Trees.testtree(tree)),
            MaximumValue(),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    scheduler=DynamicScheduler(),
    verbose=true,
);
comparematvec(h2mat_wideband, A, x, xt)
