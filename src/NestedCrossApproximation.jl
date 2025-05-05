module NestedCrossApproximation

using BEAST
using Base.Threads
using ClusterTrees
using FastBEAST
import FastBEAST.NminClusterTrees.NminTree
using LinearAlgebra
using LinearMaps
using StaticArrays
using Statistics
using ThreadsX

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

export PetrovGalerkinNCA
export GalerkinNCA
export ChebyshevRep

end
