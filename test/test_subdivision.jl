using CompScienceMeshes
using ParallelKMeans
using H2Trees
using NestedCrossApproximation
using LinearAlgebra: dot
using StaticArrays: SVector
using Test
using Plots
plotlyjs()

##

m = meshcuboid(1.0, 1.0, 1.0, 0.02)
pts = raviartthomas(m).pos
#push!(pts, SVector(0.0, 0.0, 0.1))
Random.seed!(1)
ttree = KMeansTree(pts, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere)
#ttree = TwoNTree(pts, 0.0; minvalues=100)
#stree = TwoNTree(pts, 0.0; minvalues=100)
Random.seed!(1)
stree = KMeansTree(pts, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere)
tree = BlockTree(ttree, stree)

##
h = edgeinfo(m)[3]
λ = 10h
k = 2 * pi / λ

isnear = NestedCrossApproximation.isnearwideband(k; ηhf=5.0, γ=1.0);
testdata, trialdata = NestedCrossApproximation.fardata(tree, isnear)

node = 45#H2Trees.leaves(ttree, 1)
H2Trees.center(ttree, node)

nears = Int[]
for leaf in H2Trees.leaves(stree, 1)
    if isnear(ttree, stree, node, leaf)
        push!(nears, leaf)
    end
end

##

pnode = H2Trees.parent(ttree, node)
ppnode = H2Trees.parent(ttree, pnode)
pppnode = H2Trees.parent(ttree, ppnode)
fars = NestedCrossApproximation.fars(testdata, node)
pts = [H2Trees.center(stree, node) for node in fars]
npts = [H2Trees.center(stree, node) for node in nears]
scatter(
    getindex.(npts, 1), getindex.(npts, 2), getindex.(npts, 3); markersize=3, color=:red
)
refcts = H2Trees.center(ttree, node)
refctsp = H2Trees.center(ttree, pnode)
refctspp = H2Trees.center(ttree, ppnode)
scatter!([refcts[1]], [refcts[2]], [refcts[3]]; markersize=4, color=:blue)
refcts = H2Trees.center(ttree, node)
scatter!([refctsp[1]], [refctsp[2]], [refctsp[3]]; markersize=4, label="p")
refcts = H2Trees.center(ttree, node)
scatter!([refctspp[1]], [refctspp[2]], [refctspp[3]]; markersize=4, label="pp")

for dir in NestedCrossApproximation.dirs(testdata, node)[10:10]
    dirfar = NestedCrossApproximation.dirfars(testdata, node, dir)
    println("n dirfars $(length(dirfar))")
    dirfarfield = NestedCrossApproximation.dirfarfield(ttree, testdata, node, dir)
    println("n dirfarfield $(length(dirfarfield))")
    dpts = [H2Trees.center(stree, node) for node in dirfarfield]
    scatter!(
        getindex.(dpts, 1),
        getindex.(dpts, 2),
        getindex.(dpts, 3);
        markersize=3,
        label="Dir $dir",
    )
end
##
display(plot!())
