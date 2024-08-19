module NestedCrossApproximation

using FLoops
using BEAST
using ClusterTrees
using FastBEAST
using LinearAlgebra
using LinearMaps
using StaticArrays
using Statistics
using SparseArrays

include("NCA/AbstractNCA.jl")
include("pca/pca_utils.jl")
include("pca/pca.jl")
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


export row_pivot_selection
export column_pivot_selection
export build_test_bases
export build_test_bases
export assemble_couplingmatrices

end 