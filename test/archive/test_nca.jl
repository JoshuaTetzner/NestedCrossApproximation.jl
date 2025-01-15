using BEAST
using FastBEAST
using ClusterTrees
using CompScienceMeshes
using NestedCrossApproximation

##

Γ = meshsphere(1.0, 0.05)
op = Helmholtz3D.singlelayer(wavenumber=1.0)
cxd0 = lagrangecxd0(Γ);
#A = assemble(op, cxd0, cxd0)
##
#h2mat = NestedCrossApproximation.PetrovGalerkinNCA(op, cxd0, cxd0)
compressor = NestedCrossApproximation.PCAOptions(
    NestedCrossApproximation.PCAPivoting(space.pos),
    NestedCrossApproximation.PCAPivoting(space.pos),
    tol=10^-3
);
tree = create_tree(cxd0.pos, KMeansTreeOptions(nmin=100))
@time h2mat1 = NestedCrossApproximation.GalerkinNCA(op, cxd0, tree=tree, compressor=compressor);
@time h2mat2 = NestedCrossApproximation.GalerkinNCA(op, cxd0, tree=tree);
@time h2mat3 = NestedCrossApproximation.PetrovGalerkinNCA(op, cxd0,cxd0, compressor=compressor);
@time h2mat4 = NestedCrossApproximation.PetrovGalerkinNCA(op, cxd0,cxd0);
##
x = rand(ComplexF64,12576)
#A*x
println(h2mat1.momentcollection[16])
##
@time h2mat1*x;
@time h2mat2*x;
@time h2mat3*x;
@time h2mat4*x;
##
adjoint(A)*x
@time adjoint(h2mat1)*x
@time adjoint(h2mat4)*x

transpose(assemble(op, cxd0, cxd0))*x


##
a = rand(ComplexF64, 10, 10)
adjoint(a)-transpose(conj(a))

##
storage(h2mat1)
storage(h2mat2)

x2 = lowrankmatrix(h2mat1, scalartype(op))

estimate_reldifference(h2mat1, h2mat2, tol=1e-4)

##
a = rand(10)
for i in reverse(eachindex(a))
    println(i)
end