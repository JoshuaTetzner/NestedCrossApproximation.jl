using BEAST
using BlockSparseMatrices
using NestedCrossApproximation
using CompScienceMeshes
using LinearAlgebra
using H2Trees
##

λ = 0.5
k = 2 * pi / λ
ηₗ = 1.0
ηₕ = 4.0
Γ = meshsphere(1.0, 0.05)
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)

tree = TwoNTree(space, space, 0.1)

nearassembler = NestedCrossApproximation.BlockBEASTNearInteractionsAssembler{scalartype(op)}(
    op, space, space, BEAST.DoubleNumQStrat(2, 3)
)

function isnear(k, treea, treeb, nodea, nodeb; ηₗ=1.0, ηₕ=4.0)
    ths = H2Trees.halfsize(treea, nodea)
    shs = H2Trees.halfsize(treeb, nodeb)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)
    if k / pi * 4 * min(ths, shs) <= 1
        (2 * max(ths, shs) <= ηₗ * max(dist, 0.0)) ? (return true) : (return false)
    else
        (4 * k * max(ths^2, shs^2) <= ηₕ * max(dist, 0.0)) ? (return true) : (return false)
    end
end

function myisnear(treea, treeb, nodea, nodeb)
    return nodea == nodeb
end

@time nears = nearassembler(tree, H2Trees.isnear)
nears
##
values, nearvalues = H2Trees.nearinteractions(
    tree; isnear=myisnear, extractselfvalues=false
)

@show Base.get_extension(NestedCrossApproximation, :NCABlockSparse)
