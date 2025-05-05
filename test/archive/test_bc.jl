using BEAST
using CompScienceMeshes
using StaticArrays

Γ = meshsphere(1.0, 0.1)

λ = 10
k = 2 * pi / λ
op = Maxwell3D.singlelayer(; wavenumber=k)
Y = raviartthomas(Γ)
X = buffachristiansen(Γ)
X.pos
##
@time A = assemble(op, Y, Y);
@time A = assemble(op, X, X);

@time hmat = HM.assemble(op, X, X);

##
testcomp = NestedCrossApproximation.TopDownCompressor(iACA(X.pos), nothing)
trialcomp = NestedCrossApproximation.TopDownCompressor(
    NestedCrossApproximation.iACA(
        X.pos;
        rowpivoting=NestedCrossApproximation.IACAPivoting(X.pos),
        columnpivoting=FastBEAST.LRF.MaximumValue(),
    ),
    nothing,
)

@time h2mat = PetrovGalerkinNCA(
    op,
    X,
    X;
    testcompressor=testcomp,
    trialcompressor=trialcomp,
    multithreading=true,
    maxrank=100,
)
