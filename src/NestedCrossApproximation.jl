module NestedCrossApproximation

using AdaptiveCrossApproximation
using BlockSparseMatrices
using H2Trees
using LinearMaps
using LinearAlgebra
using OhMyThreads
using StaticArrays

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
end
