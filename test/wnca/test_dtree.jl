using BEAST
using H2Trees
using NestedCrossApproximation
using CompScienceMeshes

##
λ = 0.8
k = 2 * pi / λ
##
Γ = meshsphere(1.0, 0.02; generator=:gmsh)
space = raviartthomas(Γ)

ttree = H2Trees.TwoNTree(space, 0.0; minvalues=200)
tree = BlockTree(ttree, ttree)
@time dtree = NestedCrossApproximation.𝒟tree(H2Trees.halfsize(tree.testcluster), k)
##
islf = NestedCrossApproximation.islf(k)
function isnear(treea, treeb, nodea, nodeb)
    return NestedCrossApproximation.isnear(k, treea, treeb, nodea, nodeb; ηₗ=1.0, ηₕ=4.0)
end

values, nearvalues = H2Trees.nearinteractions(tree; isnear=isnear, extractselfvalues=false)

##
@time valuess, nearvalues, fars, dirs, lfvalues, lffarvalues = NestedCrossApproximation.directionalitneractions(
    tree, dtree, islf, isnear
);
##
##
maximum(length.(Iterators.flatten(lffarvalues)))
maximum(length.(lfvalues))
length.(lfvalues)
##
lambda = 1
k = 2 * pi / lambda
hs = 1.0
tester = SVector(-1.0, 0.0, 0.0)
@time tree = 𝒟tree(hs, k)

direction(tester, tree, 1)

tree.nodes[6]
##
for node in eachindex(tree.nodes)
    if tree.nodes[node].level == 3
        println(tree.nodes[node].parent, ": ", node)
    end
end
##
##
tree.children[1].children[1].children

##
lambda = 1
k = 2 * pi / lambda
hs = 1

l = floor(log2(k * hs))

hs / 8 * k

##
a = [1, 3, 4, 2, 2, 3, 2, 1, 5, 1]
b = Vector(1:10)

function dirsort(a, b)
    return [b[findall(x -> x == dir, a)] for dir in unique(a)]
end

@time dirsort(a, b)
@time findall(x -> x == 1, a)
##
comb = [(rand(1:100), rand(1:10000)) for i in 1:10000]

function getvalues(comb)
    return [rand(1:100, 100) for i in findall(x -> x[1] == 10, comb)]
end

function getvalues2(comb)
    vals = Int[]
    for i in comb
        i[1] == 10 && append!(vals, rand(1:100, 100))
    end
    return vals
end
##
@time getvalues(comb);
@time getvalues2(comb);

using CompScienceMeshes
using LinearAlgebra
using PlotlyJS
using StaticArrays
# vertices: v₁, v₂, v₃, ...
curve(t) = SVector((9 / 20 - (1 / 9)cos(5t)) * cos(t), (9 / 20 - (1 / 9)cos(5t)) * sin(t))

# Detect closed curve
msh = meshcurve(curve, 0.1; tend=Float64(2π))
verts = msh.vertices

# segments: (start_index, end_index)
segs = msh.faces

xs = Float64[]
ys = Float64[]

for (i, j) in segs
    vi = verts[i]
    vj = verts[j]
    push!(xs, vi[1], vj[1], NaN)  # NaN separates segments
    push!(ys, vi[2], vj[2], NaN)
end

plt = plot(
    scatter(;
        x    = xs,
        y    = ys,
        mode = "lines+markers",  # markers show vertices
        name = "segments",
    ),
    Layout(;
        xaxis = attr(; scaleanchor="y"),
        yaxis = attr(; scaleratio=1),
        title = "Piecewise linear curve",
    ),
)

display(plt)

PlotlyJS.plot(wireframe(msh))
