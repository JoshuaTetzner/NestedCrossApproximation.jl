import H2Trees: levels, LevelIterator

mutable struct BottomUp{LowRankFactorizationType,RepresentorType}
    factorization::LowRankFactorizationType
    representor::RepresentorType

    function BottomUp(factorization, representor)
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
    pbar=Progress(0; enabled=false),
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
                factorization = _build_stateful_factorization(
                    compressor.factorization, farmatrix, maxrank
                )
                representor = _stateful_representor(compressor.representor, maxrank)
            end

            Ft = farfield(testtree(tree), fardata, t)
            if !isempty(Ft)
                tvalues = Int[]
                if isleaf(testtree(tree), t)
                    append!(tvalues, H2Trees.values(testtree(tree), t))
                else
                    @inbounds for child in H2Trees.ChildIterator(testtree(tree), t)
                        append!(tvalues, tpivots[child])
                    end
                end

                tpivots[t], _ = compute_test_pivots!(
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
                next!(pbar)
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

# BottomUp: each active cluster costs about one coupling-matrix-equivalent
# (computing its pivots is comparable work to evaluating one coupling block), so
# the per-active-cluster dry run is already the right cost model — no per-level
# correction needed here, unlike TopDown.
_basesworkload(::BottomUp, tree, fardata::FarData) = _farworkload(tree, fardata)

function trialbases(
    compressor::BottomUp,
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::FarData;
    scheduler=DynamicScheduler(),
    maxrank=40,
    pbar=Progress(0; enabled=false),
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
                factorization = _build_stateful_factorization(
                    compressor.factorization, farmatrix, maxrank
                )
                representor = _stateful_representor(compressor.representor, maxrank)
            end
            Fs = farfield(trialtree(tree), fardata, s)
            if !isempty(Fs)
                svalues = Int[]
                if isleaf(trialtree(tree), s)
                    append!(svalues, H2Trees.values(trialtree(tree), s))
                else
                    @inbounds for child in H2Trees.ChildIterator(trialtree(tree), s)
                        append!(svalues, spivots[child])
                    end
                end

                _, spivots[s] = compute_trial_pivots!(
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
                next!(pbar)
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

# GPU
#=
function testbases(
    compressor::BottomUp,
    farmatrix::GPUBEASTKernelMatrix{T},
    tree::BlockTree,
    fardata::FarData;
    scheduler=DynamicScheduler(),
    maxrank=40,
) where {T}
    trialpivots = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    for level in reverse(levels(testtree(tree)))
        testclusters = collect(LevelIterator(testtree(tree), level))
        for t in testclusters
            Ft = farfield(testtree(tree), fardata, t)
            if !isempty(Ft)
                trialpivots[t] = trialpivots!(
                    factorization, tree, tvalues, Ft; maxrank=maxrank
                )
            end
        end
    end

    return trialpivots
end=#

# wideband nca

function testbases(
    compressor::BottomUp,
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::DirectionalData;
    scheduler=DynamicScheduler(),
    maxrank=40,
    pbar=Progress(0; enabled=false),
) where {T}
    buffer = testbuffer(compressor, farmatrix, maxrank)

    tpivots = Vector{Vector{Int}}(undef, length(fardata.dirs))
    bases = Vector{Matrix{T}}(undef, length(fardata.dirs))
    basescount = zeros(Int, length(testtree(tree).nodes))
    transfer = Vector{Vector{Matrix{T}}}(undef, length(fardata.dirs))
    transfercount = zeros(Int, length(testtree(tree).nodes))
    transferdirs = Vector{Vector{Int}}(undef, length(fardata.dirs))
    level_transfer_diridcs = [Int[] for _ in 1:length(levels(testtree(tree)))]
    level_transfer_groupsizes = [Int[] for _ in 1:length(levels(testtree(tree)))]
    nleaves = count(n -> isleaf(testtree(tree), n), 1:numberofnodes(testtree(tree)))
    for level in reverse(levels(testtree(tree)))
        testclusters = collect(LevelIterator(testtree(tree), level))
        hflevelstep = max(1, nleaves ÷ length(testclusters))
        @tasks for t in testclusters
            @set scheduler = scheduler
            @local begin
                farbuffer = fartestbuffer(compressor.factorization, farmatrix, maxrank)
                factorization = _build_stateful_factorization(
                    compressor.factorization, farmatrix, maxrank
                )
                representor = _stateful_representor(compressor.representor, maxrank)
            end
            for (localdiridx, diridx) in enumerate(dirrange(fardata, t))
                Ft = dirfarfield(testtree(tree), fardata, t, fardata.dirs[diridx])
                if !isempty(Ft)
                    tvalues = Int[]
                    if isbasisnode(testtree(tree), fardata, t, fardata.dirs[diridx])
                        append!(tvalues, H2Trees.values(testtree(tree), t))
                    else
                        @inbounds for child in H2Trees.ChildIterator(testtree(tree), t)
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
                        basescount[t] += 1
                    else
                        transfercount[t] += 1
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
                    next!(pbar; step=(fardata.dirs[diridx] == 0 ? 1 : hflevelstep))
                end
            end
        end

        sizehint!(level_transfer_diridcs[level], sum(transfercount[testclusters]))
        @inbounds for t in testclusters
            groupsize = 0
            for diridx in dirrange(fardata, t)
                if isassigned(transfer, diridx)
                    push!(level_transfer_diridcs[level], diridx)
                    groupsize += 1
                end
            end
            groupsize > 0 && push!(level_transfer_groupsizes[level], groupsize)
        end
    end

    basis_store = _build_dirbasis_storage(bases, basescount)
    transfer_store = _build_dirtransfer_storage(
        transfer, transferdirs, level_transfer_diridcs, level_transfer_groupsizes
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
    pbar=Progress(0; enabled=false),
) where {T}
    buffer = trialbuffer(compressor, farmatrix, maxrank)

    spivots = Vector{Vector{Int}}(undef, length(fardata.dirs))
    bases = Vector{Matrix{T}}(undef, length(fardata.dirs))
    basescount = zeros(Int, length(trialtree(tree).nodes))
    transfer = Vector{Vector{Matrix{T}}}(undef, length(fardata.dirs))
    transfercount = zeros(Int, length(trialtree(tree).nodes))
    transferdirs = Vector{Vector{Int}}(undef, length(fardata.dirs))
    level_transfer_diridcs = [Int[] for _ in 1:length(levels(trialtree(tree)))]
    level_transfer_groupsizes = [Int[] for _ in 1:length(levels(trialtree(tree)))]
    nleaves = count(n -> isleaf(trialtree(tree), n), 1:numberofnodes(trialtree(tree)))
    for level in reverse(levels(trialtree(tree)))
        trialclusters = collect(LevelIterator(trialtree(tree), level))
        hflevelstep = max(1, nleaves ÷ length(trialclusters))
        @tasks for s in trialclusters
            @set scheduler = scheduler
            @local begin
                farbuffer = fartrialbuffer(compressor.factorization, farmatrix, maxrank)
                factorization = _build_stateful_factorization(
                    compressor.factorization, farmatrix, maxrank
                )

                representor = _stateful_representor(compressor.representor, maxrank)
            end
            for (localdiridx, diridx) in enumerate(dirrange(fardata, s))
                Fs = dirfarfield(trialtree(tree), fardata, s, fardata.dirs[diridx])
                if !isempty(Fs)
                    svalues = Int[]
                    if isbasisnode(trialtree(tree), fardata, s, fardata.dirs[diridx])
                        append!(svalues, H2Trees.values(trialtree(tree), s))
                    else
                        @inbounds for child in H2Trees.ChildIterator(trialtree(tree), s)
                            cdiridx = dirmap(fardata, child)[localdiridx]
                            cglobalidx = dirrange(fardata, child)[cdiridx]
                            append!(svalues, spivots[cglobalidx])
                        end
                    end

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
                        basescount[s] += 1
                    else
                        transfercount[s] += 1
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
                    end
                    next!(pbar; step=(fardata.dirs[diridx] == 0 ? 1 : hflevelstep))
                end
            end
        end

        sizehint!(level_transfer_diridcs[level], sum(transfercount[trialclusters]))
        @inbounds for s in trialclusters
            groupsize = 0
            for diridx in dirrange(fardata, s)
                if isassigned(transfer, diridx)
                    push!(level_transfer_diridcs[level], diridx)
                    groupsize += 1
                end
            end
            groupsize > 0 && push!(level_transfer_groupsizes[level], groupsize)
        end
    end

    basis_store = _build_dirbasis_storage(bases, basescount)
    transfer_store = _build_dirtransfer_storage(
        transfer, transferdirs, level_transfer_diridcs, level_transfer_groupsizes
    )

    return basis_store, transfer_store, spivots
end

# BottomUp + DirectionalData: LF pairs (dir==0) keep weight 1 each; HF levels
# (dir!=0) follow the same per-level-constant model as TopDown — roughly
# nleaves × avg-HF-directions-per-cluster per active HF level. Mirrors the
# `fardata.dirs[diridx]==0 && next!(pbar)` + post-loop HF batch logic above.
function _basesworkload(::BottomUp, tree, fardata::DirectionalData)
    nleaves = count(n -> isleaf(tree, n), 1:numberofnodes(tree))
    workload = 0
    for level in levels(tree)
        leveliter = LevelIterator(tree, level)
        hflevelstep = max(1, nleaves ÷ length(leveliter))
        for t in leveliter
            for diridx in dirrange(fardata, t)
                dir = fardata.dirs[diridx]
                isempty(dirfarfield(tree, fardata, t, dir)) && continue
                workload += (dir == 0 ? 1 : hflevelstep)
            end
        end
    end
    return workload
end
