using BEAST
using NestedCrossApproximation
using CompScienceMeshes
using H2Trees
using AdaptiveCrossApproximation
using StaticArrays
using LinearAlgebra

λ = 0.2
k = 2 * pi / λ
Γ = meshrectangle(1.0, 1.0, 0.01)
Γ2 = translate(Γ, SVector(1.0, 0.0, 0.0))

op = Maxwell3D.singlelayer(; wavenumber=k)
spaceX = raviartthomas(Γ)
spaceY = raviartthomas(Γ2)
ttree = H2Trees.TwoNTree(spaceX, 0.0; minvalues=200)
stree = H2Trees.TwoNTree(spaceY, 0.0; minvalues=200)
tree = BlockTree(ttree, stree)
function isnear(k, treea, treeb, nodea, nodeb; ηₗ=1.0, ηₕ=4.0)
    if nodea == 113
        #println(nodeb)
        ths = H2Trees.halfsize(treea, nodea) * sqrt(3)
        shs = H2Trees.halfsize(treeb, nodeb) * sqrt(3)
        dist =
            norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)
        if k / pi * 4 * min(ths, shs) <= 1
            (2 * max(ths, shs) <= ηₗ * max(dist, 0.0)) ? (return false) : (return true)
        else
            if (4 * k * max(ths^2, shs^2) <= ηₕ * max(dist, 0.0))
                #println(nodeb)
                (return false)
            else
                (return true)
            end
        end
    else
        ths = H2Trees.halfsize(treea, nodea) * sqrt(3)
        shs = H2Trees.halfsize(treeb, nodeb) * sqrt(3)
        dist =
            norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)
        if k / pi * 4 * min(ths, shs) <= 1
            (2 * max(ths, shs) <= ηₗ * max(dist, 0.0)) ? (return false) : (return true)
        else
            if (4 * k * max(ths^2, shs^2) <= ηₕ * max(dist, 0.0))
                (return false)
            else
                (return true)
            end
        end
    end
end
issnear(treea, treeb, nodea, nodeb) = isnear(k, treea, treeb, nodea, nodeb)

islf = NestedCrossApproximation.islf(NestedCrossApproximation.wavenumber(op))
dtree = NestedCrossApproximation.𝒟tree(H2Trees.halfsize(tree.testcluster), imag(op.gamma))
values, nearvalues, fars, dirs, lfvalues, lffarvalues = NestedCrossApproximation.directionalitneractions(
    tree, dtree, islf, issnear
);

tdirfars = NestedCrossApproximation.testfars(dtree, tree, fars, dirs, islf)
tdirfars[4][113]
##
iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> myisnear)(tree)
for s in H2Trees.WellSeparatedIterator(tree, 113)
    if issnear(tree.testcluster, tree.trialcluster, 113, s)
        println("Node:", s)
    end
end

##
ct = tree.testcluster.nodes[113].data.center
hs = tree.testcluster.nodes[113].data.halfsize
for s in tdirfars[4][113][1367]
    #println(s, ", ", norm(ct - tree.trialcluster.nodes[s].data.center))
    println(issnear(tree.testcluster, tree.trialcluster, 113, s))
end
#iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> issnear)(tree)
iterator
