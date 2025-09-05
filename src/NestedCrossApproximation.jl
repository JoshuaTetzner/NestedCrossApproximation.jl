module NestedCrossApproximation

using AdaptiveCrossApproximation
using BlockSparseMatrices
using LinearMaps
using LinearAlgebra
using H2Trees
using OhMyThreads
using StaticArrays

include("abstractkernelmatrix/abstractkernelmatrix.jl")
include("abstractkernelmatrix/beastkernelmatrix.jl")

include("NCA/AbstractNCA.jl")

include("matrixblocks.jl")
include("coupling.jl")
include("bases.jl")
include("compressor.jl")
include("topdowncompressor2.jl")
include("buffer.jl")
include("NCA/PetrovGalerkinNCA.jl")
include("MV/PetrovGalerkinNCA.jl")
include("WNCA/directionaltree.jl")
include("WNCA/PetrovGalerkinWNCA.jl")
#=
using BEAST
using Base.Threads
using ClusterTrees
using FastBEAST
import FastBEAST.NminClusterTrees.NminTree
using LinearAlgebra
using LinearMaps
using StaticArrays
using Statistics
using H2Trees
using ThreadsX
using AdaptiveCrossApproximation

include("incompletefactorization/convergence.jl")
include("incompletefactorization/pivoting.jl")
include("incompletefactorization/incompleteaca.jl")

export iACA
export IACAPivoting

include("NCA/AbstractNCA.jl")
include("representor.jl")
include("lowrankfactorization.jl")
include("buffer.jl")
include("x2x.jl")
include("topdowncompressor.jl")
include("buttomupcompressor.jl")
include("zhaocompressor.jl")
include("i2o.jl")

include("NCA/GalerkinNCA.jl")
include("NCA/PetrovGalerkinNCA.jl")
include("MV/GalerkinNCA.jl")
include("MV/PetrovGalerkinNCA.jl")
include("utils.jl")

include("nearinteractions/abstractnearinteraction.jl")
include("nearinteractions/blocknearinteractions.jl")

include("matrixblocks.jl")

include("WBNCA/abstractkernel.jl")
include("WBNCA/compressor.jl")
include("WBNCA/tree.jl")
include("WBNCA/dtree.jl")
include("WBNCA/directionalcompressor.jl")
include("WBNCA/coupling.jl")
include("WBNCA/WBNCA.jl")

export PetrovGalerkinNCA
export GalerkinNCA
export ChebyshevRep
=#
end
