using FastBEAST
using BEAST
using NestedCrossApproximation
using CompScienceMeshes
using ClusterTrees
using StaticArrays
using Test


Γ=meshsphere(1.0, 0.04)

op = Helmholtz3D.singlelayer()
space = lagrangecxd0(Γ)
fct(x) = 1/x

piv = NestedCrossApproximation.MyPivoting(fct, space.pos)
##
compressor1 = NestedCrossApproximation.PCA2Options(
    NestedCrossApproximation.MaximumValue(),
    NestedCrossApproximation.MyPivoting(fct, space.pos),
    30,
    10^-4
);


##
tree = create_tree(space.pos, KMeansTreeOptions(nmin=50));
##

@time h2mat1 = NestedCrossApproximation.GalerkinNCA2(
    op, space, tree=tree, compressor=compressor1, multithreading=true
);
##
compressor = NestedCrossApproximation.PCAOptions(
    NestedCrossApproximation.PCAPivoting(fct, space.pos),
    NestedCrossApproximation.PCAPivoting(fct, space.pos),
    maxrank=30,
    tol=10^-4
);

@time h2mat = NestedCrossApproximation.GalerkinNCA(
    op, space, tree=tree, compressor=compressor, multithreading=true
);

##
x = rand(size(h2mat1, 2))
##
using LinearAlgebra
M = assemble(op, space, space)
norm(M*x-h2mat*x)/norm(M*x)
norm(M*x-h2mat1*x)/norm(M*x)
##
@time hmat = HM.assemble(op, space, tree=tree, multithreading=true);

##
A = zeros(10, 10)

norm(A) == 0.0

##
