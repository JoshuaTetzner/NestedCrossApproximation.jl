import H2Trees: levels, LevelIterator

mutable struct BottomUp{LowRankFactorizationType,RepresentorType}
    factorization::LowRankFactorizationType
    representor::RepresentorType

    function BottomUp(factorization, representor)
        factorization isa AdaptiveCrossApproximation.ACA &&
            error("BottomUp with ACA is not permitted.")
        return new{typeof(factorization),typeof(representor)}(factorization, representor)
    end
end

function BottomUp(; factorization=AdaptiveCrossApproximation.ACA(), representor=nothing)
    return BottomUp(factorization, representor)
end

function (compressor::BottomUp)(
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::FarData,
    buffer::Tuple{K,Channel{K}};
    scheduler=DynamicScheduler(),
    maxrank=40,
) where {T,K<:Matrix{T}}
    factorizationpool = factorization_channel(
        compressor; scheduler=scheduler, maxrank=maxrank
    )

    tpivots = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(testtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(testtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(testtree(tree)))]

    for level in reverse(levels(testtree(tree)))
        testclusters = collect(LevelIterator(testtree(tree), level))
        @tasks for t in testclusters
            @set scheduler = scheduler
            Ft = farfield(testtree(tree), fardata, t)
            if !isempty(Ft)
                tvalues = Int[]
                if isleaf(testtree(tree), t)
                    append!(tvalues, H2Trees.values(testtree(tree), t))
                else
                    for child in H2Trees.ChildIterator(testtree(tree), t)
                        append!(tvalues, tpivots[child])
                    end
                end
                tvalues = H2Trees.values(testtree(tree), t)
                adaptedFt = adapt_farfield_indices(compressor, trialtree(tree), Ft)
                factorization = take!(factorizationpool)
                try
                    tpivots[t], _ = compute_test_pivots!(
                        factorization,
                        nothing,
                        farmatrix,
                        tree,
                        tvalues,
                        adaptedFt,
                        buffer[1],
                        buffer[2];
                        maxrank=maxrank,
                    )
                finally
                    put!(factorizationpool, factorization)
                end
                if isleaf(testtree(tree), t)
                    bases[t] = nestedtestbasis(tvalues, tpivots[t], buffer[1])
                else
                    children = collect(H2Trees.ChildIterator(testtree(tree), t))
                    nodetransfers = Vector{Matrix{T}}(undef, length(children))
                    @inbounds for j in eachindex(children)
                        child = children[j]
                        nodetransfers[j] = testtransfer(
                            tpivots[t], tpivots[child], buffer[1]
                        )
                    end
                    transfer[t] = nodetransfers
                end
            end
        end

        sizehint!(level_transfer_nodes[level], length(testclusters))
        @inbounds for t in testclusters
            if !isleaf(testtree(tree), t) && isassigned(transfer, t)
                push!(level_transfer_nodes[level], t)
            end
        end
    end

    basis_store = _build_basis_storage(bases)
    transfer_store = _build_transfer_storage(transfer, level_transfer_nodes, testtree(tree))
    return basis_store, transfer_store, tpivots
end

function (compressor::BottomUp)(
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::FarData,
    buffer::Tuple{Channel{K},K};
    scheduler=DynamicScheduler(),
    maxrank=40,
) where {T,K<:Matrix{T}}
    factorizationpool = factorization_channel(
        compressor; scheduler=scheduler, maxrank=maxrank
    )

    spivots = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(trialtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(trialtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(trialtree(tree)))]

    for level in reverse(levels(trialtree(tree)))
        trialclusters = collect(LevelIterator(trialtree(tree), level))
        @tasks for s in trialclusters
            @set scheduler = scheduler
            Fs = farfield(trialtree(tree), fardata, s)
            if !isempty(Fs)
                svalues = Int[]
                if isleaf(trialtree(tree), s)
                    append!(svalues, H2Trees.values(trialtree(tree), s))
                else
                    for child in H2Trees.ChildIterator(trialtree(tree), s)
                        append!(svalues, spivots[child])
                    end
                end
                adaptedFs = adapt_farfield_indices(compressor, testtree(tree), Fs)
                factorization = take!(factorizationpool)
                try
                    _, spivots[s] = compute_trial_pivots!(
                        factorization,
                        nothing,
                        farmatrix,
                        tree,
                        adaptedFs,
                        svalues,
                        buffer[1],
                        buffer[2];
                        maxrank=maxrank,
                    )
                finally
                    put!(factorizationpool, factorization)
                end

                if isleaf(trialtree(tree), s)
                    bases[s] = nestedtrialbasis(svalues, spivots[s], buffer[2])
                else
                    children = collect(H2Trees.ChildIterator(trialtree(tree), s))
                    nodetransfers = Vector{Matrix{T}}(undef, length(children))
                    @inbounds for j in eachindex(children)
                        child = children[j]
                        nodetransfers[j] = trialtransfer(
                            spivots[s], spivots[child], buffer[2]
                        )
                    end
                    transfer[s] = nodetransfers
                end
            end
        end

        sizehint!(level_transfer_nodes[level], length(trialclusters))
        @inbounds for s in trialclusters
            if !isleaf(trialtree(tree), s) && isassigned(transfer, s)
                push!(level_transfer_nodes[level], s)
            end
        end
    end

    basis_store = _build_basis_storage(bases)
    transfer_store = _build_transfer_storage(
        transfer, level_transfer_nodes, trialtree(tree)
    )

    return basis_store, transfer_store, spivots
end

# wideband nca

function (compressor::BottomUp)(
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::DirectionalData,
    buffer::Tuple{K,Channel{K}};
    scheduler=DynamicScheduler(),
    maxrank=40,
) where {T,K<:Matrix{T}}
    factorizationpool = factorization_channel(
        compressor; scheduler=scheduler, maxrank=maxrank
    )

    tpivots = Vector{Vector{Int}}(undef, length(fardata.dirs))
    bases = Vector{Matrix{T}}(undef, length(fardata.dirs))
    basescounter = zeros(Int, length(testtree(tree).nodes))
    transfer = Vector{Vector{Matrix{T}}}(undef, length(fardata.dirs))
    transfercounter = zeros(Int, length(testtree(tree).nodes))
    transferdirs = Vector{Vector{Int}}(undef, length(fardata.dirs))
    level_transfer_diridcs = [Int[] for _ in 1:length(levels(testtree(tree)))]

    for level in reverse(levels(testtree(tree)))
        testclusters = collect(LevelIterator(testtree(tree), level))
        @tasks for t in testclusters
            @set scheduler = scheduler
            for (localdiridx, diridx) in enumerate(dirrange(fardata, t))
                Ft = dirfarfield(testtree(tree), fardata, t, fardata.dirs[diridx])
                if !isempty(Ft)
                    tvalues = Int[]
                    if isbasisnode(testtree(tree), fardata, t, fardata.dirs[diridx])
                        append!(tvalues, H2Trees.values(testtree(tree), t))
                    else
                        for children in H2Trees.ChildIterator(testtree(tree), t)
                            cdiridx = dirmap(fardata, child)[localdiridx]
                            cglobalidx = dirrange(fardata, child)[cdiridx]
                            append!(tvalues, tpivots[cglobalidx])
                        end
                    end
                    adaptedFt = adapt_farfield_indices(compressor, trialtree(tree), Ft)
                    factorization = take!(factorizationpool)
                    try
                        tpivots[diridx], _ = compute_test_pivots!(
                            factorization,
                            nothing,
                            farmatrix,
                            tree,
                            tvalues,
                            adaptedFt,
                            buffer[1],
                            buffer[2];
                            maxrank=maxrank,
                        )
                    finally
                        put!(factorizationpool, factorization)
                    end
                    if isbasisnode(testtree(tree), fardata, t, fardata.dirs[diridx])
                        bases[diridx] = nestedtestbasis(tvalues, tpivots[diridx], buffer[1])
                        basescounter[t] += 1
                    else
                        transfercounter[t] += 1
                        children = collect(H2Trees.ChildIterator(testtree(tree), t))

                        nodetransfers = Vector{Matrix{T}}(undef, length(children))
                        nodechilddirs = Vector{Int}(undef, length(children))
                        @inbounds for j in eachindex(children)
                            child = children[j]
                            cdiridx = dirmap(fardata, child)[localdiridx]
                            cglobalidx = dirrange(fardata, child)[cdiridx]
                            nodetransfers[j] = testtransfer(
                                tpivots[diridx], tpivots[cglobalidx], buffer[1]
                            )
                            nodechilddirs[j] = cglobalidx
                        end
                        transferdirs[diridx] = nodechilddirs
                        transfer[diridx] = nodetransfers
                    end
                end
            end
        end

        sizehint!(level_transfer_diridcs[level], sum(transfercounter[testclusters]))
        @inbounds for t in testclusters
            for diridx in dirrange(fardata, t)
                if isassigned(transfer, diridx)
                    push!(level_transfer_diridcs[level], diridx)
                end
            end
        end
    end

    basis_store = _build_dirbasis_storage(bases, basescounter)
    transfer_store = _build_dirtransfer_storage(
        transfer, transferdirs, level_transfer_diridcs, testtree(tree)
    )

    return basis_store, transfer_store, tpivots
end

function (compressor::BottomUp)(
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::DirectionalData,
    buffer::Tuple{Channel{K},K};
    scheduler=DynamicScheduler(),
    maxrank=40,
) where {T,K<:Matrix{T}}
    factorizationpool = factorization_channel(
        compressor; scheduler=scheduler, maxrank=maxrank
    )

    spivots = Vector{Vector{Int}}(undef, length(fardata.dirs))
    bases = Vector{Matrix{T}}(undef, length(fardata.dirs))
    basescounter = zeros(Int, length(trialtree(tree).nodes))
    transfer = Vector{Vector{Matrix{T}}}(undef, length(fardata.dirs))
    transfercounter = zeros(Int, length(trialtree(tree).nodes))
    transferdirs = Vector{Vector{Int}}(undef, length(fardata.dirs))
    level_transfer_diridcs = [Int[] for _ in 1:length(levels(trialtree(tree)))]

    for level in reverse(levels(trialtree(tree)))
        trialclusters = collect(LevelIterator(trialtree(tree), level))
        @tasks for s in trialclusters
            @set scheduler = scheduler
            for (localdiridx, diridx) in enumerate(dirrange(fardata, s))
                Fs = dirfarfield(trialtree(tree), fardata, s, fardata.dirs[diridx])
                if !isempty(Fs)
                    svalues = Int[]
                    if isbasisnode(trialtree(tree), fardata, s, fardata.dirs[diridx])
                        append!(svalues, H2Trees.values(trialtree(tree), s))
                    else
                        for children in H2Trees.ChildIterator(trialtree(tree), s)
                            cdiridx = dirmap(fardata, child)[localdiridx]
                            cglobalidx = dirrange(fardata, child)[cdiridx]
                            append!(svalues, spivots[cglobalidx])
                        end
                    end

                    adaptedFs = adapt_farfield_indices(compressor, testtree(tree), Fs)
                    factorization = take!(factorizationpool)
                    try
                        _, spivots[diridx] = compute_trial_pivots!(
                            factorization,
                            nothing,
                            farmatrix,
                            tree,
                            adaptedFs,
                            svalues,
                            buffer[1],
                            buffer[2];
                            maxrank=maxrank,
                        )
                    finally
                        put!(factorizationpool, factorization)
                    end

                    if isbasisnode(trialtree(tree), fardata, s, fardata.dirs[diridx])
                        bases[diridx] = nestedtrialbasis(
                            svalues, spivots[diridx], buffer[2]
                        )
                        basescounter[s] += 1
                    else
                        children = collect(H2Trees.ChildIterator(trialtree(tree), s))

                        nodetransfers = Vector{Matrix{T}}(undef, length(children))
                        nodechilddirs = Vector{Int}(undef, length(children))
                        @inbounds for j in eachindex(children)
                            child = children[j]
                            cdiridx = dirmap(fardata, child)[localdiridx]
                            cglobalidx = dirrange(fardata, child)[cdiridx]
                            nodetransfers[j] = trialtransfer(
                                spivots[diridx], spivots[cglobalidx], buffer[2]
                            )
                            nodechilddirs[j] = cglobalidx
                        end
                        transferdirs[diridx] = nodechilddirs
                        transfer[diridx] = nodetransfers
                        transfercounter[s] += 1
                    end
                end
            end
        end

        sizehint!(level_transfer_diridcs[level], sum(transfercounter[trialclusters]))
        @inbounds for s in trialclusters
            for diridx in dirrange(fardata, s)
                if !isbasisnode(trialtree(tree), fardata, s, fardata.dirs[diridx]) &&
                    isassigned(transfer, diridx)
                    push!(level_transfer_diridcs[level], diridx)
                end
            end
        end
    end

    basis_store = _build_dirbasis_storage(bases, basescounter)
    transfer_store = _build_dirtransfer_storage(
        transfer, transferdirs, level_transfer_diridcs, trialtree(tree)
    )

    return basis_store, transfer_store, spivots
end
