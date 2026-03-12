using NestedCrossApproximation
using BEAST
using FastBEAST
using CompScienceMeshes
using ClusterTrees
using LinearAlgebra
using Test

##

Γ = meshsphere(1.0, 0.08)
op = Helmholtz3D.singlelayer()
space = lagrangecxd0(Γ)

tree = create_tree(space.pos, KMeansTreeOptions(; nmin=50))
testcomp = NestedCrossApproximation.TopDownCompressor(iACA(space.pos), nothing)
trialcomp = NestedCrossApproximation.TopDownCompressor(
    NestedCrossApproximation.iACA(
        space.pos;
        rowpivoting=NestedCrossApproximation.IACAPivoting(space.pos),
        columnpivoting=FastBEAST.LRF.MaximumValue(),
    ),
    nothing,
)
A = assemble(op, space, space)
##

@time h2mat, tp, sp = PetrovGalerkinNCA(
    op,
    space,
    space;
    testtree=tree,
    trialtree=tree,
    testcompressor=testcomp,
    trialcompressor=trialcomp,
    multithreading=false,
)

##

for basis in h2mat.nestedtestbases
    @test basis[2].τ == h2mat.nestedtrialbases[basis[1]].σ
    @test basis[2].σ == h2mat.nestedtrialbases[basis[1]].τ
    if (
        norm(basis[2].T - transpose(h2mat.nestedtrialbases[basis[1]].T)) / norm(basis[2].T)
    ) > 1
        println(basis[1])
    end
end

t = h2mat.nestedtestbases[244].σ
s = h2mat.nestedtrialbases[244].τ

tp[244]

for idx in [244, 144, 88]
    tbasis = h2mat.nestedtestbases[idx]
    tref = A[tbasis.τ, tbasis.σ] * A[tp[idx][1], tp[idx][2]]^-1
    println("testbasis: ", norm(tref - tbasis.T) / norm(tref))

    sbasis = h2mat.nestedtrialbases[idx]
    sref = A[sp[idx][1], sp[idx][2]]^-1 * A[sbasis.τ, sbasis.σ]
    println("trialbasis: ", norm(sref - sbasis.T) / norm(sref))
    @test tp[idx][1] == sp[idx][2]
    @test tp[idx][2] == sp[idx][1]
    println(
        norm(A[tp[idx][1], tp[idx][2]] - transpose(A[sp[idx][1], sp[idx][2]])) /
        norm(A[tp[idx][1], tp[idx][2]]),
    )
    println(
        norm(A[tbasis.τ, tbasis.σ] - transpose(A[sbasis.τ, sbasis.σ])) /
        norm(A[tbasis.τ, tbasis.σ]),
    )

    println(norm(tbasis.T - adjoint(sbasis.T)) / norm(tbasis.T))
    println(norm(tref - transpose(sref)) / norm(tref))
end

tp[244][1]
sp[244][2]
