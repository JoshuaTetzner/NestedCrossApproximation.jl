import H2Trees: levels, LevelIterator

mutable struct BottomUp{LowRankFactorizationType,RepresentorType}
    factorization::LowRankFactorizationType
    representor::RepresentorType

    function BottomUp(factorization, representor)
        #factorization isa AdaptiveCrossApproximation.ACA &&
        #    error("BottomUp with ACA is not permitted.")
        return new{typeof(factorization),typeof(representor)}(factorization, representor)
    end
end

function BottomUp(; factorization=AdaptiveCrossApproximation.ACA(), representor=nothing)
    return BottomUp(factorization, representor)
end

function testbases(
    compressor::BottomUp,
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::FarData;
    scheduler=DynamicScheduler(),
    maxrank=40,
) where {T}
    buffer = testbuffer(compressor, farmatrix, maxrank)

    tpivots = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(testtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(testtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(testtree(tree)))]

    for level in reverse(levels(testtree(tree)))
        testclusters = collect(LevelIterator(testtree(tree), level))
        @tasks for t in testclusters
            @set scheduler = scheduler
            @local begin
                farbuffer = fartestbuffer(compressor.factorization, farmatrix, maxrank)
                factorization = _stateful_factorization(compressor.factorization, maxrank)
            end

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
                #tvalues = H2Trees.values(testtree(tree), t)
                #adaptedFt = adapt_farfield_indices(compressor, trialtree(tree), Ft)

                tpivots[t], _ = compute_test_pivots!(
                    factorization,
                    nothing,
                    farmatrix,
                    tree,
                    tvalues,
                    adaptedFt,
                    buffer,
                    farbuffer;
                    maxrank=maxrank,
                )

                if isleaf(testtree(tree), t)
                    bases[t] = nestedtestbasis(tvalues, tpivots[t], buffer)
                else
                    children = collect(H2Trees.ChildIterator(testtree(tree), t))
                    nodetransfers = Vector{Matrix{T}}(undef, length(children))
                    @inbounds for j in eachindex(children)
                        child = children[j]
                        nodetransfers[j] = testtransfer(tpivots[t], tpivots[child], buffer)
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

function trialbases(
    compressor::BottomUp,
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::FarData;
    scheduler=DynamicScheduler(),
    maxrank=40,
) where {T}
    buffer = trialbuffer(compressor, farmatrix, maxrank)

    spivots = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(trialtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(trialtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(trialtree(tree)))]

    for level in reverse(levels(trialtree(tree)))
        trialclusters = collect(LevelIterator(trialtree(tree), level))
        @tasks for s in trialclusters
            @set scheduler = scheduler
            @local begin
                farbuffer = fartrialbuffer(compressor.factorization, farmatrix, maxrank)
                factorization = _stateful_factorization(compressor.factorization, maxrank)
            end
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

                _, spivots[s] = compute_trial_pivots!(
                    factorization,
                    nothing,
                    farmatrix,
                    tree,
                    adaptedFs,
                    svalues,
                    farbuffer,
                    buffer;
                    maxrank=maxrank,
                )

                if isleaf(trialtree(tree), s)
                    bases[s] = nestedtrialbasis(svalues, spivots[s], buffer)
                else
                    children = collect(H2Trees.ChildIterator(trialtree(tree), s))
                    nodetransfers = Vector{Matrix{T}}(undef, length(children))
                    @inbounds for j in eachindex(children)
                        child = children[j]
                        nodetransfers[j] = trialtransfer(spivots[s], spivots[child], buffer)
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

function testbases(
    compressor::BottomUp,
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::DirectionalData;
    scheduler=DynamicScheduler(),
    maxrank=40,
) where {T}
    buffer = testbuffer(compressor, farmatrix, maxrank)

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
            #println("t: $t")
            @set scheduler = scheduler
            @local begin
                farbuffer = fartestbuffer(compressor.factorization, farmatrix, maxrank)
                factorization = _stateful_factorization(compressor.factorization, maxrank)
                #factorization = _stateful_factorization(
                #    compressor.factorization, farmatrix, maxrank
                #)
                representor = _stateful_representor(compressor.representor, maxrank)
            end
            for (localdiridx, diridx) in enumerate(dirrange(fardata, t))
                Ft = dirfarfield(testtree(tree), fardata, t, fardata.dirs[diridx])
                if !isempty(Ft)
                    tvalues = Int[]
                    if isbasisnode(testtree(tree), fardata, t, fardata.dirs[diridx])
                        append!(tvalues, H2Trees.values(testtree(tree), t))
                    else
                        for child in H2Trees.ChildIterator(testtree(tree), t)
                            #println("node: $t, child: $child, localdiridx: $localdiridx") #= --- IGNORE ===#
                            cdiridx = dirmap(fardata, child)[localdiridx]
                            cglobalidx = dirrange(fardata, child)[cdiridx]
                            append!(tvalues, tpivots[cglobalidx])
                        end
                    end

                    tpivots[diridx], _ = compute_test_pivots!(
                        factorization,
                        representor,
                        farmatrix,
                        tree,
                        tvalues,
                        Ft,
                        buffer,
                        farbuffer;
                        maxrank=maxrank,
                    )
                    if isbasisnode(testtree(tree), fardata, t, fardata.dirs[diridx])
                        bases[diridx] = nestedtestbasis(tvalues, tpivots[diridx], buffer)
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
                                tpivots[diridx], tpivots[cglobalidx], buffer
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

function trialbases(
    compressor::BottomUp,
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::DirectionalData;
    scheduler=DynamicScheduler(),
    maxrank=40,
) where {T}
    buffer = trialbuffer(compressor, farmatrix, maxrank)

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
            @local begin
                farbuffer = fartrialbuffer(compressor.factorization, farmatrix, maxrank)
                factorization = _stateful_factorization(compressor.factorization, maxrank)
                #factorization = _stateful_factorization(
                #    compressor.factorization, farmatrix, maxrank
                #)
                representor = _stateful_representor(compressor.representor, maxrank)
            end
            for (localdiridx, diridx) in enumerate(dirrange(fardata, s))
                Fs = dirfarfield(trialtree(tree), fardata, s, fardata.dirs[diridx])
                if !isempty(Fs)
                    svalues = Int[]
                    if isbasisnode(trialtree(tree), fardata, s, fardata.dirs[diridx])
                        append!(svalues, H2Trees.values(trialtree(tree), s))
                    else
                        for child in H2Trees.ChildIterator(trialtree(tree), s)
                            cdiridx = dirmap(fardata, child)[localdiridx]
                            cglobalidx = dirrange(fardata, child)[cdiridx]
                            append!(svalues, spivots[cglobalidx])
                        end
                    end
                    #svalues = H2Trees.values(trialtree(tree), s)
                    #adaptedFs = adapt_farfield_indices(compressor, testtree(tree), Fs)

                    _, spivots[diridx] = compute_trial_pivots!(
                        factorization,
                        representor,
                        farmatrix,
                        tree,
                        Fs,
                        svalues,
                        farbuffer,
                        buffer;
                        maxrank=maxrank,
                    )

                    if isbasisnode(trialtree(tree), fardata, s, fardata.dirs[diridx])
                        bases[diridx] = nestedtrialbasis(svalues, spivots[diridx], buffer)
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
                                spivots[diridx], spivots[cglobalidx], buffer
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
