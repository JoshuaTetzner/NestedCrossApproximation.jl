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

# Directional variant of TransferTraversalPlan: carries spatial-node grouping pointers so
# that _disaggregate_coefficients! can parallelize over spatial nodes (safe: different
# spatial nodes own disjoint child-direction ranges) while processing each node's direction
# indices serially within one task (avoiding the write-write race that occurs when sibling
# direction-tree nodes at the same cluster level share the same child direction index).
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

function plan_from_pivots(pivots::AbstractVector, nvals::Int=length(pivots))
    globalidxtocoeffptr = Vector{Int}(undef, nvals + 1)
    globalidxtocoeffptr[1] = 1
    @inbounds for val in 1:nvals
        rank = isassigned(pivots, val) ? length(pivots[val]) : 0
        globalidxtocoeffptr[val + 1] = globalidxtocoeffptr[val] + rank
    end
    return CoefficientPlan(globalidxtocoeffptr)
end

# `fardata(tree, isnear)` returns either a (FarData,FarData) or a
# (DirectionalData,DirectionalData) pair depending on isnear's concrete type
# (IsNearFunctor vs IsNearWidebandFunctor). Dispatching `assemblefars` on
# isnear's concrete type, rather than branching on `isnear isa
# IsNearWidebandFunctor` inside one method, keeps every method body — and the
# `_buildbases_and_coupling` methods each one calls — locked to a single
# concrete fardata type from the start. A shared `where {T<:Union{...}}`
# bound would not do this: JET's package-level analysis checks declared
# signatures rather than per-call instantiations, so it would still see T as
# that Union and report the (impossible) cross-combinations as ambiguous.
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

# Bases (ACA/iACA pivoting, one bar) and coupling (one kernel-block evaluation per
# far pair, a separate bar) have unrelated per-unit costs, so they get separate
# progress bars rather than sharing one linear scale where neither end is weighted
# correctly. `_farworkload` is a cheap dry run that counts only clusters (or, for
# DirectionalData, cluster/direction pairs) that will actually do work, so levels
# with no interactions contribute nothing; both bars advance at per-cluster /
# per-coupling-block granularity (not per-level / not in one final batch).
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
    basesworkload =
        _basesworkload(testcompressor, testtree(tree), testfardata) +
        _basesworkload(trialcompressor, trialtree(tree), trialfardata)
    basespbar = Progress(basesworkload; desc="Assembling far bases: ", enabled=verbose)
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

    couplingpbar = Progress(
        length(farnodes(testfardata)); desc="Assembling far coupling: ", enabled=verbose
    )
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
    basesworkload =
        _basesworkload(testcompressor, testtree(tree), testfardata) +
        _basesworkload(trialcompressor, trialtree(tree), trialfardata)
    basespbar = Progress(basesworkload; desc="Assembling far bases: ", enabled=verbose)
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

    couplingpbar = Progress(
        length(farnodes(testfardata)); desc="Assembling far coupling: ", enabled=verbose
    )
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

function _project_to_coefficients!(
    coeffs::AbstractVector,
    x::AbstractVector,
    basis_store::BasisStore{T,<:BasisTraversalPlan},
    plan,
    valuesfun,
    basisop,
    scheduler,
) where {T}
    @tasks for i in eachindex(basis_store.plan.nodes)
        @set scheduler = scheduler
        node = basis_store.plan.nodes[i]
        ptr0 = plan.globalidxtocoeffptr[node]
        ptr1 = plan.globalidxtocoeffptr[node + 1] - 1
        coeff_node = @view coeffs[ptr0:ptr1]
        x_node = @view x[valuesfun(node)]
        mul!(coeff_node, basisop(basis_store.blocks[i]), x_node, true, false)
    end
end

function _project_to_coefficients!(
    coeffs::AbstractVector,
    x::AbstractVector,
    basis_store::BasisStore{T,<:DirBasisTraversalPlan},
    plan,
    valuesfun,
    basisop,
    scheduler,
) where {T}
    @tasks for i in eachindex(basis_store.plan.nodes)
        @set scheduler = scheduler
        node = basis_store.plan.nodes[i]
        x_node = @view x[valuesfun(node)]
        idxtodiridxptr = basis_store.plan.idxtodiridxptr
        for ldiridx in idxtodiridxptr[i]:(idxtodiridxptr[i + 1] - 1)
            diridx = basis_store.plan.diridx[ldiridx]
            coeffptr = plan.globalidxtocoeffptr
            coeff_node = @view coeffs[coeffptr[diridx]:(coeffptr[diridx + 1] - 1)]
            mul!(coeff_node, basisop(basis_store.blocks[ldiridx]), x_node, true, false)
        end
    end
end

function _aggregate_coefficients!(
    xhat::AbstractVector, transfer_store, plan, transferop, scheduler
)
    leveltoidcsptr = transfer_store.plan.leveltoidcsptr
    for level in (length(leveltoidcsptr) - 1):-1:1
        @tasks for idx in leveltoidcsptr[level]:(leveltoidcsptr[level + 1] - 1)
            @set scheduler = scheduler

            globalidx = transfer_store.plan.levelidcs[idx]
            coeffptr = plan.globalidxtocoeffptr
            idxxhat = view(xhat, coeffptr[globalidx]:(coeffptr[globalidx + 1] - 1))

            idxtochdsptr = transfer_store.plan.idxtochdsptr
            chdsidcs = transfer_store.plan.chdsidcs
            for cidx in idxtochdsptr[idx]:(idxtochdsptr[idx + 1] - 1)
                childidx = chdsidcs[cidx]
                childxhat = view(xhat, coeffptr[childidx]:(coeffptr[childidx + 1] - 1))
                @views idxxhat .+= transferop(transfer_store.blocks[cidx]) * childxhat
            end
        end
    end
end

function _couple_forward!(
    yhat::AbstractVector,
    xhat::AbstractVector,
    coupling_store,
    testplan,
    trialplan,
    couplingop,
    scheduler,
)
    nodetofarnodesptr = coupling_store.plan.nodetofarnodesptr
    @tasks for tidx in 1:(length(nodetofarnodesptr) - 1)
        @set scheduler = scheduler
        ptrstart = nodetofarnodesptr[tidx]
        ptrend = nodetofarnodesptr[tidx + 1] - 1
        testcoeffptr = testplan.globalidxtocoeffptr
        trialcoeffptr = trialplan.globalidxtocoeffptr
        for cidx in ptrstart:ptrend
            sidx = coupling_store.plan.farnodes[cidx]
            yhat_test = @view yhat[testcoeffptr[tidx]:(testcoeffptr[tidx + 1] - 1)]
            xhat_trial = @view xhat[trialcoeffptr[sidx]:(trialcoeffptr[sidx + 1] - 1)]
            @views yhat_test .+= couplingop(coupling_store.blocks[cidx]) * xhat_trial
        end
    end
end

function _couple_forward!(
    yhat::AbstractVector,
    xhat::AbstractVector,
    coupling_store::CouplingStore{T,<:DirCouplingTraversalPlan},
    testplan,
    trialplan,
    couplingop,
    scheduler,
) where {T}
    nodetodiridxptr = coupling_store.plan.nodetodiridxptr
    @tasks for t in 1:(length(nodetodiridxptr) - 1)
        @set scheduler = scheduler
        ptrstart = nodetodiridxptr[t]
        ptrend = nodetodiridxptr[t + 1] - 1
        testcoeffptr = testplan.globalidxtocoeffptr
        trialcoeffptr = trialplan.globalidxtocoeffptr
        for idx in ptrstart:ptrend
            tidx = coupling_store.plan.tidcs[idx]
            sidx = coupling_store.plan.sidcs[idx]
            yhat_test = @view yhat[testcoeffptr[tidx]:(testcoeffptr[tidx + 1] - 1)]
            xhat_trial = @view xhat[trialcoeffptr[sidx]:(trialcoeffptr[sidx + 1] - 1)]
            @views yhat_test .+= couplingop(coupling_store.blocks[idx]) * xhat_trial
        end
    end
end

function _couple_reverse!(
    yhat::AbstractVector,
    xhat::AbstractVector,
    coupling_store,
    trialplan,
    testplan,
    couplingop,
    scheduler,
)
    nodetofarnodesptr = coupling_store.plan.nodetofarnodesptr
    trialcoeffptr = trialplan.globalidxtocoeffptr
    testcoeffptr = testplan.globalidxtocoeffptr
    for tidx in 1:(length(nodetofarnodesptr) - 1)
        ptrstart = nodetofarnodesptr[tidx]
        ptrend = nodetofarnodesptr[tidx + 1] - 1
        for cidx in ptrstart:ptrend
            sidx = coupling_store.plan.farnodes[cidx]
            yhat_trial = @view yhat[trialcoeffptr[sidx]:(trialcoeffptr[sidx + 1] - 1)]
            xhat_test = @view xhat[testcoeffptr[tidx]:(testcoeffptr[tidx + 1] - 1)]
            @views yhat_trial .+= couplingop(coupling_store.blocks[cidx]) * xhat_test
        end
    end
end

function _couple_reverse!(
    yhat::AbstractVector,
    xhat::AbstractVector,
    coupling_store::CouplingStore{T,<:DirCouplingTraversalPlan},
    trialplan,
    testplan,
    couplingop,
    scheduler,
) where {T}
    nodetodiridxptr = coupling_store.plan.nodetodiridxptr
    trialcoeffptr = trialplan.globalidxtocoeffptr
    testcoeffptr = testplan.globalidxtocoeffptr
    for t in 1:(length(nodetodiridxptr) - 1)
        ptrstart = nodetodiridxptr[t]
        ptrend = nodetodiridxptr[t + 1] - 1
        for idx in ptrstart:ptrend
            tidx = coupling_store.plan.tidcs[idx]
            sidx = coupling_store.plan.sidcs[idx]
            yhat_trial = @view yhat[trialcoeffptr[sidx]:(trialcoeffptr[sidx + 1] - 1)]
            xhat_test = @view xhat[testcoeffptr[tidx]:(testcoeffptr[tidx + 1] - 1)]
            @views yhat_trial .+= couplingop(coupling_store.blocks[idx]) * xhat_test
        end
    end
end

function _disaggregate_coefficients!(
    yhat::AbstractVector, transfer_store, plan, transferop, scheduler
)
    leveltoidcsptr = transfer_store.plan.leveltoidcsptr
    for level in 1:(length(leveltoidcsptr) - 1)
        @tasks for idx in leveltoidcsptr[level]:(leveltoidcsptr[level + 1] - 1)
            @set scheduler = scheduler

            globalidx = transfer_store.plan.levelidcs[idx]
            coeffptr = plan.globalidxtocoeffptr
            idxyhat = view(yhat, coeffptr[globalidx]:(coeffptr[globalidx + 1] - 1))
            idxtochdsptr = transfer_store.plan.idxtochdsptr
            chdsidcs = transfer_store.plan.chdsidcs
            for cidx in idxtochdsptr[idx]:(idxtochdsptr[idx + 1] - 1)
                childidx = chdsidcs[cidx]
                childyhat = view(yhat, coeffptr[childidx]:(coeffptr[childidx + 1] - 1))
                @views childyhat .+= transferop(transfer_store.blocks[cidx]) * idxyhat
            end
        end
    end
end

# Directional disaggregation: parallelise over spatial-node groups (safe — different
# spatial nodes own disjoint child-direction index ranges), process each node's
# direction indices serially so sibling direction-tree nodes that share a common child
# direction accumulate without a write-write race.
function _disaggregate_coefficients!(
    yhat::AbstractVector,
    transfer_store::TransferStore{T,<:DirTransferTraversalPlan},
    plan,
    transferop,
    scheduler,
) where {T}
    leveltoidcsptr = transfer_store.plan.leveltoidcsptr
    leveltogroupsptr = transfer_store.plan.leveltogroupsptr
    groupsptr = transfer_store.plan.groupsptr
    for level in 1:(length(leveltoidcsptr) - 1)
        @tasks for grp in leveltogroupsptr[level]:(leveltogroupsptr[level + 1] - 1)
            @set scheduler = scheduler
            coeffptr = plan.globalidxtocoeffptr
            idxtochdsptr = transfer_store.plan.idxtochdsptr
            chdsidcs = transfer_store.plan.chdsidcs
            for idx in groupsptr[grp]:(groupsptr[grp + 1] - 1)
                globalidx = transfer_store.plan.levelidcs[idx]
                idxyhat = view(yhat, coeffptr[globalidx]:(coeffptr[globalidx + 1] - 1))
                for cidx in idxtochdsptr[idx]:(idxtochdsptr[idx + 1] - 1)
                    childidx = chdsidcs[cidx]
                    childyhat = view(yhat, coeffptr[childidx]:(coeffptr[childidx + 1] - 1))
                    @views childyhat .+= transferop(transfer_store.blocks[cidx]) * idxyhat
                end
            end
        end
    end
end

#=
function _disaggregate_coefficients!(
    yhat::AbstractVector, transfer_store, plan, transferop, scheduler
)
    return _propagate_coefficients!(
        yhat,
        transfer_store,
        plan,
        transferop,
        scheduler;
        reverse_levels=false,
        reset_parent=false,
    )
end=#

function _project_to_output!(
    y::AbstractVector,
    yhat::AbstractVector,
    basis_store,
    plan,
    valuesfun,
    basisop,
    scheduler,
)
    @tasks for i in eachindex(basis_store.plan.nodes)
        @set scheduler = scheduler
        node = basis_store.plan.nodes[i]
        ptr0 = plan.globalidxtocoeffptr[node]
        ptr1 = plan.globalidxtocoeffptr[node + 1] - 1
        ynode = @view y[valuesfun(node)]
        yhat_node = @view yhat[ptr0:ptr1]
        @views ynode .+= basisop(basis_store.blocks[i]) * yhat_node
        #mul!(y_node, basisop(basis_store.blocks[i]), yhat_node)
    end
end

function _project_to_output!(
    y::AbstractVector,
    yhat::AbstractVector,
    basis_store::BasisStore{T,<:DirBasisTraversalPlan},
    plan,
    valuesfun,
    basisop,
    scheduler,
) where {T}
    @tasks for i in eachindex(basis_store.plan.nodes)
        node = basis_store.plan.nodes[i]
        @set scheduler = scheduler
        idxtodiridxptr = basis_store.plan.idxtodiridxptr
        coeffptr = plan.globalidxtocoeffptr
        for ldiridx in idxtodiridxptr[i]:(idxtodiridxptr[i + 1] - 1)
            diridx = basis_store.plan.diridx[ldiridx]
            y_node = @view y[valuesfun(node)]
            yhat_node = @view yhat[coeffptr[diridx]:(coeffptr[diridx + 1] - 1)]
            @views y_node .+= basisop(basis_store.blocks[ldiridx]) * yhat_node
        end
    end
end

function admissiblelevel(tree::TwoNTree, fardata::DirectionalData, isnear)
    hffarlev = 0
    lffarlev = 0
    for level in H2Trees.levels(tree)
        if isnear.islf(tree, level)
            for node in LevelIterator(tree, level)
                !isempty(farnodes(fardata, node)) && (hffarlev += 1; break)
            end
        else
            for node in LevelIterator(tree, level)
                !isempty(farnodes(fardata, node)) && (lffarlev += 1; break)
            end
        end
    end
    return max(1, max(hffarlev, lffarlev))
end

function admissiblelevel(tree, fardata, isnear)
    farlev = 0
    for level in H2Trees.levels(tree)
        for node in LevelIterator(tree, level)
            !isempty(farnodes(fardata, node)) && (farlev += 1; break)
        end
    end
    return max(1, farlev)
end

function admissiblelevel(tree::H2Trees.BoundingBallTree, fardata::DirectionalData, isnear)
    hffarlev = 0
    lffarlev = 0
    for level in H2Trees.levels(tree)
        ishffarlev = false
        islffarlev = false
        for node in LevelIterator(tree, level)
            if !isempty(farnodes(fardata, node))
                if isnear.islf(tree, node)
                    !ishffarlev && (hffarlev += 1)
                    ishffarlev = true
                else
                    !islffarlev && (lffarlev += 1)
                    islffarlev = true
                end
            end
            ishffarlev && islffarlev && break
        end
    end
    return max(1, max(hffarlev, lffarlev))
end

function tolerance!(
    lrf::AdaptiveCrossApproximation.ACA{RP,CP,CC}, denominator::F
) where {RP,CP,CC<:FNormEstimator,F}
    return lrf.convergence.tol = lrf.convergence.tol / denominator
end

function tolerance!(
    lrf::AdaptiveCrossApproximation.iACA{RP,CP,CC}, denominator::F
) where {RP,CP,CC<:FNormExtrapolator,F}
    return lrf.convergence.estimator.tol = lrf.convergence.estimator.tol / denominator
end
