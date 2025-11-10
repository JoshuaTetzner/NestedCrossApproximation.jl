module NestedCrossApproximation

using AdaptiveCrossApproximation
using BlockSparseMatrices
using LinearMaps
using LinearAlgebra
using H2Trees
import H2Trees: testtree, trialtree, levels, LevelIterator, numberofnodes, center
import H2Trees: parent, ChildIterator, firstchild
using OhMyThreads
using StaticArrays

include("abstractkernelmatrix/abstractkernelmatrix.jl")
include("abstractkernelmatrix/beastkernelmatrix.jl")

include("NCA/AbstractNCA.jl")

include("matrixblocks.jl")
include("coupling.jl")
include("bases.jl")
include("compressor.jl")
include("blockcompressor.jl")
include("farinteractions.jl")
include("bottomupcompressor.jl")
include("topdowncompressor.jl")
include("buffer.jl")
include("NCA/PetrovGalerkinNCA.jl")
include("MV/PetrovGalerkinNCA.jl")
include("WidebandNCA/directionaltree.jl")
#include("WidebandNCA/farinteractions.jl")
include("WidebandNCA/topdowncompressor.jl")
include("WidebandNCA/PetrovGalerkinWNCA.jl")
end
