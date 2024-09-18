using FastBEAST
using BEAST 
using NestedCrossApproximation
using CompScienceMeshes

λ = 3
k = 2 * π / λ
Γ = meshsphere(1.0, 0.08)
op = Maxwell3D.singlelayer(wavenumber=k)
space = raviartthomas(Γ)

##
@time A = assemble(op, space, space);
##
compressor = NestedCrossApproximation.PCAOptions(
    NestedCrossApproximation.PCAPivoting(space.pos),
    NestedCrossApproximation.PCAPivoting(space.pos),
    tol=1e-4,
    maxrank=150
);
    
tree = create_tree(space.pos, KMeansTreeOptions(nmin=500, nchildren=2))

println("PCA")
@time h2mat = NestedCrossApproximation.GalerkinNCA(
    op, space, compressor=compressor, tree=tree, η=1.0, verbose=false
);
##
@time hmat = FastBEAST.HM.assemble(
    op, 
    space,
    space;
    tree=tree,
    testtree=tree,
    trialtree=tree,
    compressor=FastBEAST.ACAOptions(tol=1e-4),
    multithreading=true
);
##
@time hmat2 = FastBEAST.hassemble(
    op, 
    space,
    space;
    treeoptions=KMeansTreeOptions(nmin=50, nchildren=2),
    compressor=FastBEAST.ACAOptions(tol=1e-4),
    multithreading=true
)
##
#FastBEAST.storage(hmat)
##
using LinearAlgebra
x = rand(Float64, size(A, 1))
@time y1 = A*x;
@time y2 = hmat*x;
@time y3 = h2mat*x;  
##
norm(y3-y1)/norm(y1)
norm(y2-y1)/norm(y1)
