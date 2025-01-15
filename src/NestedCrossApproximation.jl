module NestedCrossApproximation

using FLoops
using BEAST
using ClusterTrees
using FastBEAST
import FastBEAST.NminClusterTrees.NminTree
using LinearAlgebra
using LinearMaps
using StaticArrays
using Statistics
using SparseArrays
using Base.Threads

include("NCA/AbstractNCA.jl")
include("pca/pca_utils.jl")
include("pca/pca.jl")
include("pca/pca2.jl")
include("pca/pivoting.jl")
include("pca/compressor.jl")
include("compressor.jl")
include("pivotselection.jl")
include("i2o.jl")
include("moments.jl")
include("NCA/GalerkinNCA.jl")
include("mv/GalerkinNCA.jl")
include("NCA/PetrovGalerkinNCA.jl")
include("mv/PetrovGalerkinNCA.jl")
include("utils.jl")
include("buttomupgalerkin.jl")
include("NCA/GalerkinNCA_BA.jl")

include("flexNCA/incompletefactorization/convergence.jl")
include("flexNCA/incompletefactorization/pivoting.jl")
include("flexNCA/incompletefactorization/incompleteaca.jl")
include("flexNCA/representor.jl")
include("flexNCA/moments.jl")
include("flexNCA/translations.jl")
include("flexNCA/compressor.jl")
include("flexNCA/buffer.jl")
include("flexNCA/farinteractions.jl")
include("NCA/NCA.jl")

export row_pivot_selection
export column_pivot_selection
export build_test_bases
export build_test_bases
export assemble_couplingmatrices
export storage

end
