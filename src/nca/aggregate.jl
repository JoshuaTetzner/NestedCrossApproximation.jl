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
                mul!(
                    idxxhat, transferop(transfer_store.blocks[cidx]), childxhat, true, true
                )
            end
        end
    end
end
