using FastBEAST
using BEAST
using NestedCrossApproximation
using CompScienceMeshes

Γ=meshsphere(1.0, 0.06)

op = Helmholtz3D.singlelayer()
space = lagrangecxd0(Γ)
fct(x::F) where F = F(1/x)


@views blkasm = BEAST.blockassembler(op, space, space, quadstrat=BEAST.DoubleNumQStrat(2, 3))
 
@views function assembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    blkasm(tdata, sdata, store)
end
##
compressor1 = NestedCrossApproximation.PCA2Options(
    NestedCrossApproximation.MaximumValue(),
    NestedCrossApproximation.MyPivoting(fct, space.pos),
    30,
    10^-4
);
compressor = NestedCrossApproximation.PCAOptions(
    NestedCrossApproximation.PCAPivoting(fct, space.pos),
    NestedCrossApproximation.PCAPivoting(fct, space.pos),
    maxrank=25,
    tol=10^-4
);

##
tree = create_tree(space.pos, KMeansTreeOptions(nmin=50));

##
using ClusterTrees

blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
nears, fars = FastBEAST.computeinteractions(blktree,  η=1.0)
@time cpivots = NestedCrossApproximation.pivotselection(tree, fars, compressor1.cpivstrat;
    maxrank=40, multithreading=true
);
##

@time h2mat1 = NestedCrossApproximation.GalerkinNCA2(
    op, space, tree=tree, compressor=compressor1, multithreading=true, tol=1e-4
);

@time h2mat = NestedCrossApproximation.GalerkinNCA(
    op, space, tree=tree, compressor=compressor, multithreading=true
);

##

using LinearAlgebra
x = rand(size(h2mat1, 2))

M = assemble(op, space, space)
##

norm(M*x-h2mat*x)/norm(M*x)
norm(M*x-h2mat1*x)/norm(M*x)