using BEAST
using H2Trees
##

Γ = meshsphere(1.0, 0.03)
space = raviartthomas(Γ)
tree = H2Trees.TwoNTree(space, 0.0; minvalues=200);

##

ct, hs = H2Trees.boundingbox(space.pos)
hs
H2Trees.halfsize(tree)

##
@time dtree = NestedCrossApproximation.𝒟tree(
    H2Trees.halfsize(tree.testcluster), imag(op.gamma)
)
dtree.level
dtree.nodes[1]
iss = NestedCrossApproximation.islf(k)
iss(tree, 3)

max(0, floor(log2(k * hs * 2)) + 1)
iss(tree, 2)

maxlevel = max(0)

function maxlevel(islf::Function)
    level = 0
    while !islf(tree, level + 1)
        println(level)
        level += 1
    end
    return level
end

maxlevel(iss)
