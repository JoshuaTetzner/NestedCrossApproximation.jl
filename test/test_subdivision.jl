using CompScienceMeshes
using ParallelKMeans
using H2Trees
using NestedCrossApproximation
using LinearAlgebra: dot
using StaticArrays: SVector
using Test

##
dirs = NestedCrossApproximation.sphericalfibonaccipoints(10)
mydir = SVector(-1.0, 1.0, 0.0)
nearest = NestedCrossApproximation._nearest_direction(dirs, mydir)
scatter(
    getindex.(dirs, 1), getindex.(dirs, 2), getindex.(dirs, 3); markersize=2, color=:green
)

##

pts = meshrectangle(1.0, 1.0, 0.005).vertices
#push!(pts, SVector(0.0, 0.0, 0.1))
ttree = KMeansTree(pts, 2; minvalues=200, updateradii=H2Trees.unsafemaxradiusboundingsphere)
#ttree = TwoNTree(pts, 0.0; minvalues=100)
#stree = TwoNTree(pts, 0.0; minvalues=100)
stree = KMeansTree(pts, 2; minvalues=200, updateradii=H2Trees.unsafemaxradiusboundingsphere)
tree = BlockTree(ttree, stree)

##
λ = 0.2
k = 2 * pi / λ

isnear = NestedCrossApproximation.isnearwideband(k)
testdata, trialdata = NestedCrossApproximation.fardata(tree, isnear)

node = H2Trees.leaves(ttree, 1)[3]
H2Trees.center(ttree, node)

nears = Int[]
for leaf in H2Trees.leaves(stree, 1)
    if isnear(ttree, stree, node, leaf)
        push!(nears, leaf)
    end
end

##
fars = NestedCrossApproximation.fars(testdata, node)
dir = NestedCrossApproximation.dirs(testdata, node)[7]
dirfar = NestedCrossApproximation.dirfars(testdata, node, dir)
dirfarfield = NestedCrossApproximation.dirfarfield(ttree, testdata, node, dir)

pts = [H2Trees.center(stree, node) for node in fars]
dpts = [H2Trees.center(stree, node) for node in dirfarfield]
npts = [H2Trees.center(stree, node) for node in nears]
refcts = H2Trees.center(ttree, node)
##
using Plots
plotlyjs()

scatter(getindex.(pts, 1), getindex.(pts, 2), getindex.(pts, 3); markersize=2, color=:green)

scatter!(
    getindex.(npts, 1), getindex.(npts, 2), getindex.(npts, 3); markersize=2, color=:red
)
scatter!([refcts[1]], [refcts[2]], [refcts[3]]; markersize=4, color=:blue)
scatter!(
    getindex.(dpts, 1), getindex.(dpts, 2), getindex.(dpts, 3); markersize=2, color=:orange
)
