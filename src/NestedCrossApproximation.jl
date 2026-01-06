module NestedCrossApproximation

using AdaptiveCrossApproximation
using BlockSparseMatrices
using LinearMaps
using LinearAlgebra
using H2Trees
import H2Trees: testtree, trialtree, levels, LevelIterator, numberofnodes, center
import H2Trees: parent, ChildIterator, firstchild, ParentUpwardsIterator, BoundingBallTree
using OhMyThreads
using StaticArrays

include("abstractkernelmatrix/abstractkernelmatrix.jl")
include("abstractkernelmatrix/beastkernelmatrix.jl")
include("nearinteractions.jl")

include("NCA/AbstractNCA.jl")
#include("WidebandNCA/directionaltree.jl")
include("directionalfarinteractions/farinteractions.jl")
include("directionalfarinteractions/utilities.jl")
include("directionalfarinteractions/abstractdirections.jl")
include("directionalfarinteractions/twondirections.jl")
include("directionalfarinteractions/boundingballdirections.jl")

#include("matrixblocks.jl")
include("coupling.jl")
include("bases.jl")
include("transfermatrices.jl")
include("compressor.jl")
#include("blockcompressor.jl")
include("farinteractions.jl")
include("bottomupcompressor.jl")
include("topdowncompressor.jl")
include("buffer.jl")

include("NCA/PetrovGalerkinNCA.jl")
include("MV/PetrovGalerkinNCA.jl")

#include("WidebandNCA/farinteractions.jl")
#include("WidebandNCA/directionaltrees/utilities.jl")

include("WidebandNCA/topdowncompressor.jl")
#include("WidebandNCA/bottomupcompressor.jl")
include("WidebandNCA/PetrovGalerkinWNCA.jl")
include("utils.jl")
end
