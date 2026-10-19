# Middle of the matvec: apply the coupling blocks between admissible cluster pairs.
# Forward is parallel over test clusters; reverse (transpose/adjoint) is serial because
# several test clusters accumulate into the same trial slot.

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
