function _build_transfer_storage(
    transfer::Vector{Vector{Matrix{T}}}, leveltransfernodes::Vector{Vector{Int}}, tree
) where {T}
    leveltoidcsptr = Vector{Int}(undef, length(leveltransfernodes) + 1)
    leveltoidcsptr[1] = 1
    for level in 1:length(leveltransfernodes)
        leveltoidcsptr[level + 1] =
            leveltoidcsptr[level] + length(leveltransfernodes[level])
    end
    levelidcs = collect(Iterators.flatten(leveltransfernodes))

    idxtochdsptr = Vector{Int}(undef, length(levelidcs) + 1)
    idxtochdsptr[1] = 1
    for nodeidx in eachindex(levelidcs)
        idxtochdsptr[nodeidx + 1] =
            idxtochdsptr[nodeidx] + length(transfer[levelidcs[nodeidx]])
    end

    transferbocks = Vector{Matrix{T}}(undef, idxtochdsptr[end] - 1)
    chdsidcs = Vector{Int}(undef, idxtochdsptr[end] - 1)
    edgeidx = 1
    for levelptridx in 1:(length(leveltoidcsptr) - 1)
        for nodeidx in leveltoidcsptr[levelptridx]:(leveltoidcsptr[levelptridx + 1] - 1)
            node = levelidcs[nodeidx]
            children = collect(H2Trees.ChildIterator(tree, node))
            for childidx in eachindex(transfer[node])
                transferbocks[edgeidx] = transfer[node][childidx]
                chdsidcs[edgeidx] = children[childidx]
                edgeidx += 1
            end
        end
    end

    return TransferStore{T}(
        TransferTraversalPlan(leveltoidcsptr, levelidcs, idxtochdsptr, chdsidcs),
        transferbocks,
    )
end

function _build_dirtransfer_storage(
    transfer::Vector{Vector{Matrix{T}}},
    transferdirs::Vector{Vector{Int}},
    leveltransferdirs::Vector{Vector{Int}},
    leveltransfergroupsizes::Vector{Vector{Int}},
) where {T}
    leveltoidcsptr = Vector{Int}(undef, length(leveltransferdirs) + 1)
    leveltoidcsptr[1] = 1
    for level in 1:length(leveltransferdirs)
        leveltoidcsptr[level + 1] = leveltoidcsptr[level] + length(leveltransferdirs[level])
    end
    levelidcs = collect(Iterators.flatten(leveltransferdirs))

    idxtochdsptr = Vector{Int}(undef, length(levelidcs) + 1)
    idxtochdsptr[1] = 1
    for diridx in eachindex(levelidcs)
        idxtochdsptr[diridx + 1] =
            idxtochdsptr[diridx] + length(transfer[levelidcs[diridx]])
    end

    transferblocks = Vector{Matrix{T}}(undef, idxtochdsptr[end] - 1)
    chdsidcs = Vector{Int}(undef, idxtochdsptr[end] - 1)
    edgeidx = 1
    for levelptridx in 1:(length(leveltoidcsptr) - 1)
        for diridx in leveltoidcsptr[levelptridx]:(leveltoidcsptr[levelptridx + 1] - 1)
            dir = levelidcs[diridx]
            for childidx in eachindex(transfer[dir])
                transferblocks[edgeidx] = transfer[dir][childidx]
                chdsidcs[edgeidx] = transferdirs[dir][childidx]
                edgeidx += 1
            end
        end
    end

    ngroups = sum(length, leveltransfergroupsizes)
    leveltogroupsptr = Vector{Int}(undef, length(leveltransferdirs) + 1)
    groupsptr = Vector{Int}(undef, ngroups + 1)
    leveltogroupsptr[1] = 1
    grp = 1
    groupsptr[1] = 1
    for level in 1:length(leveltransferdirs)
        leveltogroupsptr[level + 1] =
            leveltogroupsptr[level] + length(leveltransfergroupsizes[level])
        for gsz in leveltransfergroupsizes[level]
            groupsptr[grp + 1] = groupsptr[grp] + gsz
            grp += 1
        end
    end

    return TransferStore{T}(
        DirTransferTraversalPlan(
            leveltoidcsptr, levelidcs, idxtochdsptr, chdsidcs, leveltogroupsptr, groupsptr
        ),
        transferblocks,
    )
end

function testtransfermatrices!(
    tree, transfer, levelnodes, pivots, buf; scheduler=DynamicScheduler()
)
    @tasks for nodeidx in eachindex(levelnodes)
        @set scheduler = scheduler
        node = levelnodes[nodeidx]
        children = collect(H2Trees.ChildIterator(tree, node))
        nodetransfers = Vector{Matrix{eltype(buf)}}(undef, length(children))
        @inbounds for j in eachindex(children)
            child = children[j]
            nodetransfers[j] = testtransfer(pivots[node], pivots[child], buf)
        end
        transfer[node] = nodetransfers
    end
    return nothing
end

function testtransfer(rows, childrows, buf)
    n = length(rows)
    transfer = buf[childrows, 1:n]
    return rdiv!(transfer, _crossfactorization(buf[rows, 1:n], :test_transfer_matrix))
end

function trialtransfermatrices!(
    tree, transfer, levelnodes, pivots, buf; scheduler=DynamicScheduler()
)
    @tasks for nodeidx in eachindex(levelnodes)
        @set scheduler = scheduler
        node = levelnodes[nodeidx]
        children = collect(H2Trees.ChildIterator(tree, node))
        nodetransfers = Vector{Matrix{eltype(buf)}}(undef, length(children))
        @inbounds for j in eachindex(children)
            child = children[j]
            nodetransfers[j] = trialtransfer(pivots[node], pivots[child], buf)
        end
        transfer[node] = nodetransfers
    end
    return nothing
end

function trialtransfer(cols, childcols, buf)
    n = length(cols)
    transfer = buf[1:n, childcols]
    return ldiv!(_crossfactorization(buf[1:n, cols], :trial_transfer_matrix), transfer)
end
