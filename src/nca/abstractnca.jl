struct BasisTraversalPlan
    nodes::Vector{Int}
end

struct DirBasisTraversalPlan
    nodes::Vector{Int}
    idxtodiridxptr::Vector{Int} # length(nodes)+1; position i of nodes → range in diridx
    diridx::Vector{Int}
end

struct BasisStore{T,P}
    plan::P
    blocks::Vector{Matrix{T}}

    function BasisStore{T}(plan, blocks) where {T}
        return new{T,typeof(plan)}(plan, blocks)
    end
end

struct TransferTraversalPlan
    leveltoidcsptr::Vector{Int} # length(levels)+1; level → range in levelidcs
    levelidcs::Vector{Int}      # node or diridx active at that level, flattened across levels
    idxtochdsptr::Vector{Int}   # length(levelidcs)+1; position in levelidcs → range in chdsidcs
    chdsidcs::Vector{Int}       # children's node or diridx ids, paired 1:1 with the store's blocks
end

struct DirTransferTraversalPlan
    leveltoidcsptr::Vector{Int}    # same role as TransferTraversalPlan
    levelidcs::Vector{Int}
    idxtochdsptr::Vector{Int}
    chdsidcs::Vector{Int}
    leveltogroupsptr::Vector{Int}  # length(levels)+1; level → range of groups in groupsptr
    groupsptr::Vector{Int}         # groupsptr[g]:groupsptr[g+1]-1 = levelidcs range for group g
end

struct TransferStore{T,P}
    plan::P
    blocks::Vector{Matrix{T}}

    function TransferStore{T}(plan, blocks) where {T}
        return new{T,typeof(plan)}(plan, blocks)
    end
end

struct CouplingTraversalPlan
    nodetofarnodesptr::Vector{Int} # length of test nodes + 1
    farnodes::Vector{Int}          # length of blocks -> trial node idx
end

struct DirCouplingTraversalPlan
    nodetodiridxptr::Vector{Int} # length of test nodes + 1; reuses the raw per-node far-pair ptr
    tidcs::Vector{Int}           # length of blocks -> test diridx
    sidcs::Vector{Int}           # length of blocks -> trial diridx
end

struct CouplingStore{T,P}
    plan::P
    blocks::Vector{Matrix{T}}

    function CouplingStore{T}(plan, blocks) where {T}
        return new{T,typeof(plan)}(plan, blocks)
    end
end

struct CoefficientPlan
    globalidxtocoeffptr::Vector{Int} # node or diridx (reused generically for both) → range in xhat/yhat/coeffs
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

# Split in two rather than one Union-typed `isnear`: fardata(tree, isnear) would then
# return Union{Tuple{FarData,FarData},Tuple{DirectionalData,DirectionalData}}, and JET
# union-splits that into the two impossible mixed pairs.
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
