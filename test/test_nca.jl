using BEAST
using FastBEAST
using ClusterTrees
using CompScienceMeshes
using NestedCrossApproximation

##

Γ = meshsphere(1.0, 0.1)
op = Helmholtz3D.singlelayer(wavenumber=1.0)
cxd0 = lagrangecxd0(Γ);
A = assemble(op, cxd0, cxd0)
##
#h2mat = NestedCrossApproximation.PetrovGalerkinNCA(op, cxd0, cxd0)
compressor = NestedCrossApproximation.PCAOptions(
    NestedCrossApproximation.PCAPivoting(cxd0.pos),
    NestedCrossApproximation.PCAPivoting(cxd0.pos),
    tol=10^-4
);
@time h2mat1 = NestedCrossApproximation.GalerkinNCA(op, cxd0, compressor=compressor);
@time h2mat2 = NestedCrossApproximation.GalerkinNCA(op, cxd0);
@time h2mat3 = NestedCrossApproximation.PetrovGalerkinNCA(op, cxd0,cxd0, compressor=compressor);
@time h2mat4 = NestedCrossApproximation.PetrovGalerkinNCA(op, cxd0,cxd0);
##
x = rand(ComplexF64,3214)
A*x
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