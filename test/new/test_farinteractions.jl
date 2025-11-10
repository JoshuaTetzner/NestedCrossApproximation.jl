using CompScienceMeshes
using H2Trees
using NestedCrossApproximation

Γ = meshsphere(1.0, 0.05)
ttree = TwoNTree(Γ.vertices, 0.1)
tree = BlockTree(ttree, ttree)

##
λ = 0.5
dtree = NestedCrossApproximation.𝒟tree(1.0, 2 * pi / λ)
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
myisnear(treea, treeb, nodea, nodeb) = isnear(2 * pi / λ, treea, treeb, nodea, nodeb)
##
tree
a, aa, b, bb = NestedCrossApproximation.directionalfarinteractions(
    tree, dtree; isnear=myisnear
)

##
dtree = NestedCrossApproximation.𝒟tree(1.0, 2 * pi / λ)
