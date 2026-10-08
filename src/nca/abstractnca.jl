struct BasisTraversalPlan
    nodes::Vector{Int}
end

struct DirBasisTraversalPlan
    nodes::Vector{Int}
    idxtodiridxptr::Vector{Int} # position in nodes → range in diridx
    diridx::Vector{Int}
end

"""
    BasisStore{T,P}

Leaf bases of one side of an H²-matrix.

`blocks[i]` is the basis of the cluster `plan.nodes[i]`, of size `(points, rank)` on the test
side and `(rank, points)` on the trial side. With a directional plan a cluster holds one
basis per direction.

# Fields

  - `plan::P`: clusters (and directions) the blocks belong to
  - `blocks::Vector{Matrix{T}}`: basis matrices
"""
struct BasisStore{T,P}
    plan::P
    blocks::Vector{Matrix{T}}

    function BasisStore{T}(plan, blocks) where {T}
        return new{T,typeof(plan)}(plan, blocks)
    end
end

struct TransferTraversalPlan
    leveltoidcsptr::Vector{Int} # level → range in levelidcs
    levelidcs::Vector{Int}      # parent of the transfer blocks, level by level
    idxtochdsptr::Vector{Int}   # position in levelidcs → range in chdsidcs
    chdsidcs::Vector{Int}       # child of each transfer block
end

struct DirTransferTraversalPlan
    leveltoidcsptr::Vector{Int}
    levelidcs::Vector{Int}
    idxtochdsptr::Vector{Int}
    chdsidcs::Vector{Int}
    leveltogroupsptr::Vector{Int} # level → range in groupsptr
    groupsptr::Vector{Int}        # group (the directions of one cluster) → range in levelidcs
end

"""
    TransferStore{T,P}

Transfer matrices of one side of an H²-matrix, ordered level by level.

Each block connects a cluster and one of its children, and is of size
`(child rank, parent rank)` on the test side and `(parent rank, child rank)` on the trial
side. With a directional plan the blocks connect (cluster, direction) pairs.

# Fields

  - `plan::P`: levels, parents and children the blocks belong to
  - `blocks::Vector{Matrix{T}}`: transfer matrices
"""
struct TransferStore{T,P}
    plan::P
    blocks::Vector{Matrix{T}}

    function TransferStore{T}(plan, blocks) where {T}
        return new{T,typeof(plan)}(plan, blocks)
    end
end

struct CouplingTraversalPlan
    nodetofarnodesptr::Vector{Int} # test node → range in farnodes
    farnodes::Vector{Int}          # trial node of each block
end

struct DirCouplingTraversalPlan
    nodetodiridxptr::Vector{Int} # test node → range in tidcs and sidcs
    tidcs::Vector{Int}           # test diridx of each block
    sidcs::Vector{Int}           # trial diridx of each block
end

"""
    CouplingStore{T,P}

Coupling matrices of an H²-matrix.

`blocks[i]` holds the matrix entries at the pivots of the `i`-th admissible pair of a test
and a trial cluster, `A[testpivots, trialpivots]`.

# Fields

  - `plan::P`: test and trial cluster (or direction index) of each block
  - `blocks::Vector{Matrix{T}}`: coupling matrices
"""
struct CouplingStore{T,P}
    plan::P
    blocks::Vector{Matrix{T}}

    function CouplingStore{T}(plan, blocks) where {T}
        return new{T,typeof(plan)}(plan, blocks)
    end
end

struct CoefficientPlan
    globalidxtocoeffptr::Vector{Int} # node (or diridx) → range in the coefficient vector
end

function plan_from_pivots(pivots::AbstractVector)
    nvals = length(pivots)
    globalidxtocoeffptr = Vector{Int}(undef, nvals + 1)
    globalidxtocoeffptr[1] = 1
    @inbounds for val in 1:nvals
        rank = isassigned(pivots, val) ? length(pivots[val]) : 0
        globalidxtocoeffptr[val + 1] = globalidxtocoeffptr[val] + rank
    end
    return CoefficientPlan(globalidxtocoeffptr)
end

# One method per `isnear` type keeps the type of `fardata(tree, isnear)` inferable.
"""
    assemblefars(tree, isnear, testcompressor, trialcompressor, farmatrix;
        scheduler, maxrank, verbose=false)

Assemble the far field of an H²-matrix.

Determines the admissible cluster pairs of `tree`, builds the nested bases of both sides
with [`testbases`](@ref) and [`trialbases`](@ref), and evaluates the coupling matrices at
the resulting pivots.

# Returns

A named tuple with `nestedtestbases`, `testtransfermats`, `testpivots`, `nestedtrialbases`,
`trialtransfermats`, `trialpivots`, and `couplingmatrices`.
"""
function assemblefars(
    tree,
    isnear::IsNearWidebandFunctor,
    testcompressor,
    trialcompressor,
    farmatrix;
    scheduler,
    maxrank,
    verbose::Bool=false,
)
    testfardata, trialfardata = fardata(tree, isnear)
    return _buildbases_and_coupling(
        testcompressor,
        trialcompressor,
        farmatrix,
        tree,
        testfardata,
        trialfardata;
        scheduler=scheduler,
        maxrank=maxrank,
        verbose=verbose,
    )
end

function assemblefars(
    tree,
    isnear::IsNearFunctor,
    testcompressor,
    trialcompressor,
    farmatrix;
    scheduler,
    maxrank,
    verbose::Bool=false,
)
    testfardata, trialfardata = fardata(tree, isnear)
    return _buildbases_and_coupling(
        testcompressor,
        trialcompressor,
        farmatrix,
        tree,
        testfardata,
        trialfardata;
        scheduler=scheduler,
        maxrank=maxrank,
        verbose=verbose,
    )
end

function _buildbases_and_coupling(
    testcompressor,
    trialcompressor,
    farmatrix,
    tree,
    testfardata::FarData,
    trialfardata::FarData;
    scheduler,
    maxrank,
    verbose::Bool=false,
)
    basespbar = _basesprogress(
        testcompressor, trialcompressor, tree, testfardata, trialfardata, verbose
    )
    nestedtestbases, testtransfermats, testpivots = testbases(
        testcompressor,
        farmatrix,
        tree,
        testfardata;
        scheduler=scheduler,
        maxrank=maxrank,
        pbar=basespbar,
    )
    nestedtrialbases, trialtransfermats, trialpivots = trialbases(
        trialcompressor,
        farmatrix,
        tree,
        trialfardata;
        scheduler=scheduler,
        maxrank=maxrank,
        pbar=basespbar,
    )
    finish!(basespbar)

    couplingpbar = _couplingprogress(testfardata, verbose)
    couplingmatrices = assemble_couplingstore(
        farmatrix,
        testpivots,
        trialpivots,
        testfardata,
        trialfardata;
        scheduler=scheduler,
        pbar=couplingpbar,
    )
    finish!(couplingpbar)
    return (;
        nestedtestbases,
        testtransfermats,
        testpivots,
        nestedtrialbases,
        trialtransfermats,
        trialpivots,
        couplingmatrices,
    )
end

function _buildbases_and_coupling(
    testcompressor,
    trialcompressor,
    farmatrix,
    tree,
    testfardata::DirectionalData,
    trialfardata::DirectionalData;
    scheduler,
    maxrank,
    verbose::Bool=false,
)
    _check_wideband_compressor(testcompressor, :test)
    _check_wideband_compressor(trialcompressor, :trial)

    basespbar = _basesprogress(
        testcompressor, trialcompressor, tree, testfardata, trialfardata, verbose
    )
    nestedtestbases, testtransfermats, testpivots = testbases(
        testcompressor,
        farmatrix,
        tree,
        testfardata;
        scheduler=scheduler,
        maxrank=maxrank,
        pbar=basespbar,
    )
    nestedtrialbases, trialtransfermats, trialpivots = trialbases(
        trialcompressor,
        farmatrix,
        tree,
        trialfardata;
        scheduler=scheduler,
        maxrank=maxrank,
        pbar=basespbar,
    )
    finish!(basespbar)

    couplingpbar = _couplingprogress(testfardata, verbose)
    couplingmatrices = assemble_couplingstore(
        farmatrix,
        testpivots,
        trialpivots,
        testfardata,
        trialfardata;
        scheduler=scheduler,
        pbar=couplingpbar,
    )
    finish!(couplingpbar)
    return (;
        nestedtestbases,
        testtransfermats,
        testpivots,
        nestedtrialbases,
        trialtransfermats,
        trialpivots,
        couplingmatrices,
    )
end

_idop(A) = A
