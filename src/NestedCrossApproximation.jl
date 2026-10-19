module NestedCrossApproximation

using AdaptiveCrossApproximation
using BlockSparseMatrices
using H2Trees
using LinearMaps
using LinearAlgebra
using OhMyThreads
using ProgressMeter
using StaticArrays

include("admissibility/nearinteractions.jl")
include("admissibility/farinteractions.jl")

include("admissibility/directionalsubdivision/abstractdirections.jl")
include("admissibility/directionalsubdivision/twondirections.jl")
include("admissibility/directionalsubdivision/boundingballdirections.jl")

include("utils/utils.jl")
include("nestedbasis.jl")
include("transfermatrices.jl")

include("compressors/abstractcompressor.jl")
include("compressors/topdown.jl")
include("compressors/bottomup.jl")
include("compressors/batchedbottomup.jl")
# buffer.jl dispatches on ::TopDown/::BottomUp, so it has to follow them.
include("compressors/buffer.jl")

# Same reason: workload.jl holds every progress-bar dry run, and its `_basesworkload`
# methods annotate the compressor types above.
include("utils/workload.jl")

# representor.jl defines MimicryRep/MimicryRepFunctor, which pivots.jl annotates.
include("pivoting/representor.jl")
include("pivoting/pivots.jl")

include("couplingmatrices.jl")

include("nca/abstractnca.jl")
include("nca/aggregate.jl")
include("nca/couple.jl")
include("nca/disaggregate.jl")
include("nca/petrovgalerkinnca.jl")
end
