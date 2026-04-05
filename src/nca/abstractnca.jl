struct BasisTraversalPlan
    nodes::Vector{Int}
end

struct DirBasisTraversalPlan
    nodes::Vector{Int}
    diridxptr::Vector{Int}
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
    level_ptr::Vector{Int}
    level_nodes::Vector{Int}
    child_ptr::Vector{Int}
    child_nodes::Vector{Int}
end

struct DirTransferTraversalPlan
    level_ptr::Vector{Int}
    level_dirs::Vector{Int}
    child_ptr::Vector{Int}
    child_dirs::Vector{Int}
end

struct TransferStore{T,P}
    plan::P
    blocks::Vector{Matrix{T}}

    function TransferStore{T}(plan, blocks) where {T}
        return new{T,typeof(plan)}(plan, blocks)
    end
end

struct CouplingTraversalPlan
    ptr::Vector{Int} # length of test nodes + 1
    idcs::Vector{Int} # length of blocks -> trial node idx
end

struct DirCouplingTraversalPlan
    ptr::Vector{Int} # length of test nodes + 1
    tidcs::Vector{Int} # length of blocks -> test node idx
    sidcs::Vector{Int} # length of blocks -> trial node idx
end

struct CouplingStore{T,P}
    plan::P
    blocks::Vector{Matrix{T}}

    function CouplingStore{T}(plan, blocks) where {T}
        return new{T,typeof(plan)}(plan, blocks)
    end
end

struct CoefficientPlan
    ptr::Vector{Int}
end

function plan_from_pivots(pivots::AbstractVector, nvals::Int=length(pivots))
    ptr = Vector{Int}(undef, nvals + 1)
    ptr[1] = 1
    @inbounds for val in 1:nvals
        rank = isassigned(pivots, val) ? length(pivots[val]) : 0
        ptr[val + 1] = ptr[val] + rank
    end
    return CoefficientPlan(ptr)
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
        ptr0 = plan.ptr[node]
        ptr1 = plan.ptr[node + 1] - 1
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
        for ldiridx in basis_store.plan.diridxptr[i]:(basis_store.plan.diridxptr[i + 1] - 1)
            diridx = basis_store.plan.diridx[ldiridx]
            coeff_node = @view coeffs[plan.ptr[diridx]:(plan.ptr[diridx + 1] - 1)]
            mul!(coeff_node, basisop(basis_store.blocks[ldiridx]), x_node, true, false)
        end
    end
end

function _aggregate_coefficients!(
    xhat::AbstractVector, transfer_store, plan, transferop, scheduler
)
    for level in (length(transfer_store.plan.level_ptr) - 1):-1:1
        @tasks for idx in
                   transfer_store.plan.level_ptr[level]:(transfer_store.plan.level_ptr[level + 1] - 1)
            @set scheduler = scheduler

            globalidx = transfer_store.plan.level_nodes[idx]
            idxxhat = view(xhat, plan.ptr[globalidx]:(plan.ptr[globalidx + 1] - 1))

            child_ptr = transfer_store.plan.child_ptr
            child_idcs = transfer_store.plan.child_nodes
            for cidx in child_ptr[idx]:(child_ptr[idx + 1] - 1)
                childidx = child_idcs[cidx]
                childxhat = view(xhat, plan.ptr[childidx]:(plan.ptr[childidx + 1] - 1))
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
    @tasks for tidx in 1:(length(coupling_store.plan.ptr) - 1)
        @set scheduler = scheduler
        ptrstart = coupling_store.plan.ptr[tidx]
        ptrend = coupling_store.plan.ptr[tidx + 1] - 1
        for cidx in ptrstart:ptrend
            sidx = coupling_store.plan.idcs[cidx]
            yhat_test = @view yhat[testplan.ptr[tidx]:(testplan.ptr[tidx + 1] - 1)]
            xhat_trial = @view xhat[trialplan.ptr[sidx]:(trialplan.ptr[sidx + 1] - 1)]
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
    @tasks for t in 1:(length(coupling_store.plan.ptr) - 1)
        @set scheduler = scheduler
        ptrstart = coupling_store.plan.ptr[t]
        ptrend = coupling_store.plan.ptr[t + 1] - 1
        for idx in ptrstart:ptrend
            tidx = coupling_store.plan.tidcs[idx]
            sidx = coupling_store.plan.sidcs[idx]
            yhat_test = @view yhat[testplan.ptr[tidx]:(testplan.ptr[tidx + 1] - 1)]
            xhat_trial = @view xhat[trialplan.ptr[sidx]:(trialplan.ptr[sidx + 1] - 1)]
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
    for tidx in 1:(length(coupling_store.plan.ptr) - 1)
        ptrstart = coupling_store.plan.ptr[tidx]
        ptrend = coupling_store.plan.ptr[tidx + 1] - 1
        for cidx in ptrstart:ptrend
            sidx = coupling_store.plan.idcs[cidx]
            yhat_trial = @view yhat[trialplan.ptr[sidx]:(trialplan.ptr[sidx + 1] - 1)]
            xhat_test = @view xhat[testplan.ptr[tidx]:(testplan.ptr[tidx + 1] - 1)]
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
    for t in 1:(length(coupling_store.plan.ptr) - 1)
        ptrstart = coupling_store.plan.ptr[t]
        ptrend = coupling_store.plan.ptr[t + 1] - 1
        for idx in ptrstart:ptrend
            tidx = coupling_store.plan.tidcs[idx]
            sidx = coupling_store.plan.sidcs[idx]
            yhat_trial = @view yhat[trialplan.ptr[sidx]:(trialplan.ptr[sidx + 1] - 1)]
            xhat_test = @view xhat[testplan.ptr[tidx]:(testplan.ptr[tidx + 1] - 1)]
            @views yhat_trial .+= couplingop(coupling_store.blocks[idx]) * xhat_test
        end
    end
end

function _disaggregate_coefficients!(
    yhat::AbstractVector, transfer_store, plan, transferop, scheduler
)
    for level in 1:(length(transfer_store.plan.level_ptr) - 1)
        @tasks for idx in
                   transfer_store.plan.level_ptr[level]:(transfer_store.plan.level_ptr[level + 1] - 1)
            @set scheduler = scheduler

            localidx = transfer_store.plan.level_nodes[idx]
            idxyhat = view(yhat, plan.ptr[localidx]:(plan.ptr[localidx + 1] - 1))
            child_ptr = transfer_store.plan.child_ptr
            child_nodes = transfer_store.plan.child_nodes
            for cidx in child_ptr[idx]:(child_ptr[idx + 1] - 1)
                childidx = child_nodes[cidx]
                childyhat = view(yhat, plan.ptr[childidx]:(plan.ptr[childidx + 1] - 1))
                @views childyhat .+= transferop(transfer_store.blocks[cidx]) * idxyhat
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
        ptr0 = plan.ptr[node]
        ptr1 = plan.ptr[node + 1] - 1
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
        for ldiridx in basis_store.plan.diridxptr[i]:(basis_store.plan.diridxptr[i + 1] - 1)
            diridx = basis_store.plan.diridx[ldiridx]
            y_node = @view y[valuesfun(node)]
            yhat_node = @view yhat[plan.ptr[diridx]:(plan.ptr[diridx + 1] - 1)]
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
                !isempty(fars(fardata, node)) && (hffarlev += 1; break)
            end
        else
            for node in LevelIterator(tree, level)
                !isempty(fars(fardata, node)) && (lffarlev += 1; break)
            end
        end
    end
    return max(1, max(hffarlev, lffarlev))
end

function admissiblelevel(tree, fardata, isnear)
    farlev = 0
    for level in H2Trees.levels(tree)
        for node in LevelIterator(tree, level)
            !isempty(fars(fardata, node)) && (farlev += 1; break)
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
            if !isempty(fars(fardata, node))
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
    println("New tol: ", lrf.convergence.tol / denominator)
    return lrf.convergence.tol = lrf.convergence.tol / denominator
end

function tolerance!(
    lrf::AdaptiveCrossApproximation.iACA{RP,CP,CC}, denominator::F
) where {RP,CP,CC<:FNormExtrapolator,F}
    println("New tol: ", lrf.convergence.estimator.tol / denominator)
    return lrf.convergence.estimator.tol = lrf.convergence.estimator.tol / denominator
end
