using BEAST
using FastBEAST
using ClusterTrees
using CompScienceMeshes
using LinearAlgebra
using NestedCrossApproximation

##
Γ = meshsphere(1.0, 0.08)
λ = 2.0
k = 2 * pi / λ
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
tree = create_tree(space.pos, KMeansTreeOptions(; nmin=100))

x = rand(length(space.pos))

# Full
@time A = assemble(op, space, space);

##
# Standard NCA
@time G_NCA = GalerkinNCA(op, space; tree=tree, maxrank=100, multithreading=false);
##
comp = NestedCrossApproximation.TopDownCompressor(;
    representor=ChebyshevRep(1e-3, 1.0, space.pos)
)
##
@time PG_NCA = PetrovGalerkinNCA(
    op,
    space,
    space;
    testtree=tree,
    trialtree=tree,
    #testcompressor=comp,
    #trialcompressor=comp,
    maxrank=100,
    multithreading=true,
);
println(norm(A * x - G_NCA * x) / norm(A * x))
println(norm(A * x - PG_NCA * x) / norm(A * x))
##
# iACA
comp = NestedCrossApproximation.TopDownCompressor(iACA(space.pos), nothing);
@time G_iACA = GalerkinNCA(
    op, space; compressor=comp, tree=tree, multithreading=true, maxrank=100
);

testcomp = NestedCrossApproximation.TopDownCompressor(iACA(space.pos), nothing);
trialcomp = NestedCrossApproximation.TopDownCompressor(
    NestedCrossApproximation.iACA(
        space.pos;
        rowpivoting=NestedCrossApproximation.IACAPivoting(space.pos),
        columnpivoting=FastBEAST.LRF.MaximumValue(),
    ),
    nothing,
)
@time PG_iACA = PetrovGalerkinNCA(
    op,
    space,
    space;
    testtree=tree,
    trialtree=tree,
    testcompressor=testcomp,
    trialcompressor=trialcomp,
    multithreading=true,
    maxrank=100,
);
println(norm(A * x - G_iACA * x) / norm(A * x))
println(norm(A * x - PG_iACA * x) / norm(A * x))
##
fm = fullmat(PG_iACA)
norm(A - fm) / norm(A)
storage(PG_iACA)
