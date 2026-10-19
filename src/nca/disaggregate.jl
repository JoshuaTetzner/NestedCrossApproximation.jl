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
                mul!(
                    childyhat, transferop(transfer_store.blocks[cidx]), idxyhat, true, true
                )
            end
        end
    end
end

# Parallel over the clusters, serial over the directions of a cluster: these may share
# a child direction.
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
                    mul!(
                        childyhat,
                        transferop(transfer_store.blocks[cidx]),
                        idxyhat,
                        true,
                        true,
                    )
                end
            end
        end
    end
end

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
        mul!(ynode, basisop(basis_store.blocks[i]), yhat_node, true, true)
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
            mul!(y_node, basisop(basis_store.blocks[ldiridx]), yhat_node, true, true)
        end
    end
end
