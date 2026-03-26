module NestedCrossApproximation

using AdaptiveCrossApproximation
using BlockSparseMatrices
using H2Trees
using LinearMaps
using LinearAlgebra
using OhMyThreads
using StaticArrays

#include("kernelmatrix/abstractkernelmatrix.jl")
#include("kernelmatrix/beastkernelmatrix.jl")
#include("kernelmatrix/pointmatrix.jl")

include("nearinteractions.jl")
include("farinteractions.jl")

include("directionalsubdivision/abstractdirections.jl")
include("directionalsubdivision/twondirections.jl")
include("directionalsubdivision/boundingballdirections.jl")

include("utils.jl")
include("nestedbasis.jl")
include("transfermatrices.jl")
include("topdown.jl")
include("bottomup.jl")
include("buffer.jl")
include("factorization.jl")
include("couplingmatrices.jl")

include("nca/abstractnca.jl")
include("nca/petrovgalerkinnca.jl")

#import H2Trees: testtree, trialtree, levels, LevelIterator, numberofnodes, center
#import H2Trees: parent, ChildIterator, firstchild, ParentUpwardsIterator, BoundingBallTree
#=
struct H2MatrixBlock{I,K}
    Z::Matrix{K}
    row_basis::I
    col_basis::I
end

struct H2BasisBlock{I,K}
    T::Union{Vector{Matrix{K}},Matrix{K}}
    τ::Vector{I}
    σ::Vector{I}
    children::Vector{I}
end

include("abstractkernelmatrix/abstractkernelmatrix.jl")
include("abstractkernelmatrix/beastkernelmatrix.jl")

#include("NCA/AbstractNCA.jl")
#include("WidebandNCA/directionaltree.jl")

include("directionalfarinteractions/utilities.jl")
include("directionalfarinteractions/abstractdirfars.jl")

include("farinteractions.jl")
include("nearinteractions.jl")
include("directionalfarinteractions/twondirfars.jl")
include("directionalfarinteractions/boundingballdirfars.jl")

#include("matrixblocks.jl")
include("coupling.jl")
include("bases.jl")
include("transfermatrices.jl")
include("compressor.jl")
#include("blockcompressor.jl")
#include("farinteractions.jl")
include("bottomupcompressor.jl")
include("topdowncompressor.jl")
include("buffer.jl")

include("NCA/PetrovGalerkinNCA.jl")
include("MV/PetrovGalerkinNCA.jl")

#include("WidebandNCA/farinteractions.jl")
#include("WidebandNCA/directionaltrees/utilities.jl")

include("WidebandNCA/topdowncompressor.jl")
include("WidebandNCA/compressor.jl")
include("WidebandNCA/bottomupcompressor.jl")
include("WidebandNCA/PetrovGalerkinWNCA.jl")
include("utils.jl")
=#
end
