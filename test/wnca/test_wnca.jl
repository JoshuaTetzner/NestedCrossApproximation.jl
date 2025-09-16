using BEAST
using NestedCrossApproximation
using CompScienceMeshes
using H2Trees
using LinearAlgebra

λ = 1.0
k = 2 * pi / λ
Γ = meshsphere(1.0, 0.05)

op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
ttree = H2Trees.TwoNTree(space, 0.0; minvalues=200)
tree = BlockTree(ttree, ttree)
##
function isnear(k, treea, treeb, nodea, nodeb; ηₗ=1.0, ηₕ=4.0)
    ths = H2Trees.halfsize(treea, nodea) * sqrt(3)
    shs = H2Trees.halfsize(treeb, nodeb) * sqrt(3)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)
    if k / pi * 4 * min(ths, shs) <= 1
        (2 * max(ths, shs) <= ηₗ * max(dist, 0.0)) ? (return false) : (return true)
    else
        (4 * k * max(ths^2, shs^2) <= ηₕ * max(dist, 0.0)) ? (return false) : (return true)
    end
end
issnear(treea, treeb, nodea, nodeb) = isnear(k, treea, treeb, nodea, nodeb)
wnca = NestedCrossApproximation.PetrovGalerkinWNCA(op, space, space, tree; isnear=issnear)

##
islf = NestedCrossApproximation.islf(NestedCrossApproximation.wavenumber(op))
dtree = NestedCrossApproximation.𝒟tree(H2Trees.halfsize(tree.testcluster), imag(op.gamma))
values, nearvalues, fars, dirs, lfvalues, lffarvalues = NestedCrossApproximation.directionalitneractions(
    tree, dtree, islf, issnear
)

lfvalues
maxcols = maximum(length.(Iterators.flatten(lffarvalues)))

lffarvalues == []

isempty(lffarvalues)