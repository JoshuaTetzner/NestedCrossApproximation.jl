struct BasisTraversalPlan
    nodes::Vector{Int}
end

struct BasisStore{T}
    plan::BasisTraversalPlan
    blocks::Vector{Matrix{T}}
end

struct TransferTraversalPlan
    level_ptr::Vector{Int}
    level_nodes::Vector{Int}
    node_ptr::Vector{Int}
    edge_child::Vector{Int}
end

struct TransferStore{T}
    plan::TransferTraversalPlan
    blocks::Vector{Matrix{T}}
end

struct CouplingTraversalPlan
    test_ptr::Vector{Int}
    trial_node::Vector{Int}
end

struct CouplingStore{T}
    plan::CouplingTraversalPlan
    blocks::Vector{Matrix{T}}
end

struct CoefficientPlan
    ptr::Vector{Int}
end

function plan_from_pivots(pivots::AbstractVector, nnodes::Int=length(pivots))
    ptr = Vector{Int}(undef, nnodes + 1)
    ptr[1] = 1
    @inbounds for node in 1:nnodes
        rank = isassigned(pivots, node) ? length(pivots[node]) : 0
        ptr[node + 1] = ptr[node] + rank
    end
    return CoefficientPlan(ptr)
end

_idop(A) = A

function _project_to_coefficients!(
    coeffs::AbstractVector,
    x::AbstractVector,
    basis_store,
    plan,
    valuesfun,
    basisop,
    scheduler,
)
    @tasks for i in eachindex(basis_store.plan.nodes)
        @set scheduler = scheduler
        node = basis_store.plan.nodes[i]
        ptr0 = plan.ptr[node]
        ptr1 = plan.ptr[node + 1] - 1
        coeff_node = @view coeffs[ptr0:ptr1]
        x_node = @view x[valuesfun(node)]
        mul!(coeff_node, basisop(basis_store.blocks[i]), x_node)
    end
end

function _aggregate_coefficients!(
    coeffs::AbstractVector, transfer_store, plan, transferop, scheduler
)
    return _propagate_coefficients!(
        coeffs,
        transfer_store,
        plan,
        transferop,
        scheduler;
        reverse_levels=true,
        reset_parent=true,
    )
end

function _propagate_coefficients!(
    coeffs::AbstractVector,
    transfer_store,
    plan,
    transferop,
    scheduler;
    reverse_levels::Bool,
    reset_parent::Bool,
)
    if reverse_levels
        level_range = (length(transfer_store.plan.level_ptr) - 1):-1:1
    else
        level_range = 1:(length(transfer_store.plan.level_ptr) - 1)
    end

    for level in level_range
        level_first = transfer_store.plan.level_ptr[level]
        level_last = transfer_store.plan.level_ptr[level + 1] - 1
        @tasks for nodeidx in level_first:level_last
            @set scheduler = scheduler
            parent = transfer_store.plan.level_nodes[nodeidx]
            pptr0 = plan.ptr[parent]
            pptr1 = plan.ptr[parent + 1] - 1
            coeff_parent = @view coeffs[pptr0:pptr1]
            if reverse_levels && reset_parent
                fill!(coeff_parent, zero(eltype(coeff_parent)))
            end

            edge_first = transfer_store.plan.node_ptr[nodeidx]
            edge_last = transfer_store.plan.node_ptr[nodeidx + 1] - 1
            if reverse_levels
                first_edge = true
                for transfer_idx in edge_first:edge_last
                    child = transfer_store.plan.edge_child[transfer_idx]
                    cptr0 = plan.ptr[child]
                    cptr1 = plan.ptr[child + 1] - 1
                    coeff_child = @view coeffs[cptr0:cptr1]
                    mul!(
                        coeff_parent,
                        transferop(transfer_store.blocks[transfer_idx]),
                        coeff_child,
                        true,
                        !first_edge,
                    )
                    first_edge = false
                end
            else
                for transfer_idx in edge_first:edge_last
                    child = transfer_store.plan.edge_child[transfer_idx]
                    cptr0 = plan.ptr[child]
                    cptr1 = plan.ptr[child + 1] - 1
                    coeff_child = @view coeffs[cptr0:cptr1]
                    mul!(
                        coeff_child,
                        transferop(transfer_store.blocks[transfer_idx]),
                        coeff_parent,
                        true,
                        true,
                    )
                end
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
    @tasks for testnode in 1:(length(coupling_store.plan.test_ptr) - 1)
        @set scheduler = scheduler
        coupling_first = coupling_store.plan.test_ptr[testnode]
        coupling_last = coupling_store.plan.test_ptr[testnode + 1] - 1
        for coupling_idx in coupling_first:coupling_last
            trialnode = coupling_store.plan.trial_node[coupling_idx]
            trialptr0 = trialplan.ptr[trialnode]
            trialptr1 = trialplan.ptr[trialnode + 1] - 1
            testptr0 = testplan.ptr[testnode]
            testptr1 = testplan.ptr[testnode + 1] - 1
            yhat_test = @view yhat[testptr0:testptr1]
            xhat_trial = @view xhat[trialptr0:trialptr1]
            mul!(
                yhat_test,
                couplingop(coupling_store.blocks[coupling_idx]),
                xhat_trial,
                true,
                true,
            )
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
    for testnode in 1:(length(coupling_store.plan.test_ptr) - 1)
        coupling_first = coupling_store.plan.test_ptr[testnode]
        coupling_last = coupling_store.plan.test_ptr[testnode + 1] - 1
        testptr0 = testplan.ptr[testnode]
        testptr1 = testplan.ptr[testnode + 1] - 1
        xhat_test = @view xhat[testptr0:testptr1]
        for coupling_idx in coupling_first:coupling_last
            trialnode = coupling_store.plan.trial_node[coupling_idx]
            trialptr0 = trialplan.ptr[trialnode]
            trialptr1 = trialplan.ptr[trialnode + 1] - 1
            yhat_trial = @view yhat[trialptr0:trialptr1]
            mul!(
                yhat_trial,
                couplingop(coupling_store.blocks[coupling_idx]),
                xhat_test,
                true,
                true,
            )
        end
    end
end

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
        ptr0 = plan.ptr[node]
        ptr1 = plan.ptr[node + 1] - 1
        y_node = @view y[valuesfun(node)]
        yhat_node = @view yhat[ptr0:ptr1]
        mul!(y_node, basisop(basis_store.blocks[i]), yhat_node)
    end
end
