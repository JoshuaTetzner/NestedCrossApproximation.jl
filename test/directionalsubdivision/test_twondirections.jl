using CompScienceMeshes
using H2Trees
using NestedCrossApproximation
using LinearAlgebra: dot
using StaticArrays: SVector
using Test

pts = meshicosphere(40, 1.0).vertices
pts2 = translate(meshicosphere(40, 1.0), SVector(1.0, 0.0, 0.0)).vertices
ttree = TwoNTree(pts, 2 / 2^10; minvalues=10)
stree = TwoNTree(pts2, 2 / 2^10; minvalues=10)
tree = BlockTree(ttree, stree)

##
λ = 20.0
k = 2 * pi / λ

isnear = NestedCrossApproximation.isnearwideband(k)
testdata, trialdata = NestedCrossApproximation.fardata(tree, isnear)

@test all((testdata.dirs .== 0))
@test all((trialdata.dirs .== 0))
##

λ = 0.4
k = 2π / λ

isnear = NestedCrossApproximation.isnearwideband(k)
testdata, trialdata = NestedCrossApproximation.fardata(tree, isnear)

@test all((testdata.dirs .!== 0))
@test all((trialdata.dirs .!== 0))
##
length(testdata.fdata.fars)
length(testdata.fdata.farptr)
length(testdata.dirptr)
testdata.dirs

##

for node in eachindex(ttree.nodes)
    fars = NestedCrossApproximation.fars(testdata, node)
    dirs = NestedCrossApproximation.dirs(testdata, node)
    sfars = Int[]

    @test length(dirs) == length(unique(dirs))
    for dir in dirs
        dirfars = NestedCrossApproximation.dirfars(testdata, node, dir)
        dirfarfield = NestedCrossApproximation.dirfarfield(ttree, testdata, node, dir)
        @test dirfarfield[1:length(dirfars)] == dirfars
        append!(sfars, dirfars)
    end

    @test length(sfars) == length(unique(sfars))

    if node > 1
        pnode = H2Trees.parent(ttree, node)
        dirmap = NestedCrossApproximation.dirmap(testdata, node)
        pdirs = NestedCrossApproximation.dirs(testdata, pnode)
        @test length(dirmap) == length(pdirs)
        for dir in dirmap
            @test dir in Vector(1:length(dirs))
        end
    end
end
