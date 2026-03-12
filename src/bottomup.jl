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
    tpivots = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(testtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(testtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(testtree(tree)))]
    level_child_counts = [Int[] for _ in 1:length(levels(testtree(tree)))]

    for level in reverse(levels(testtree(tree)))
        testclusters = collect(LevelIterator(testtree(tree), level))
        @tasks for t in testclusters
            @set scheduler = scheduler
            Ft = farfield(testtree(tree), fardata, t)
            if !isempty(Ft)
                tvalues = H2Trees.values(testtree(tree), t)
                adaptedFt = adapt_farfield_indices(compressor, trialtree(tree), Ft)
                tpivots[t], _ = compute_test_pivots!(
                    compressor.factorization,
                    nothing,
                    farmatrix,
                    tree,
                    tvalues,
                    adaptedFt,
                    buffer[1],
                    buffer[2];
                    maxrank=maxrank,
                )
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
        empty!(level_child_counts[level])
        sizehint!(level_child_counts[level], length(testclusters))
        @inbounds for t in testclusters
            if !isleaf(testtree(tree), t) && isassigned(transfer, t)
                push!(level_transfer_nodes[level], t)
                push!(level_child_counts[level], length(transfer[t]))
            end
        end
    end

    basis_store = _build_basis_store(bases)
    transfer_store = _build_transfer_store(
        transfer, level_transfer_nodes, level_child_counts, testtree(tree)
    )
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
    spivots = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(trialtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(trialtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(trialtree(tree)))]
    level_child_counts = [Int[] for _ in 1:length(levels(trialtree(tree)))]

    for level in reverse(levels(trialtree(tree)))
        trialclusters = collect(LevelIterator(trialtree(tree), level))
        @tasks for s in trialclusters
            @set scheduler = scheduler
            Fs = farfield(trialtree(tree), fardata, s)
            if !isempty(Fs)
                svalues = H2Trees.values(trialtree(tree), s)
                adaptedFs = adapt_farfield_indices(compressor, testtree(tree), Fs)
                _, spivots[s] = compute_trial_pivots!(
                    compressor.factorization,
                    nothing,
                    farmatrix,
                    tree,
                    adaptedFs,
                    svalues,
                    buffer[1],
                    buffer[2];
                    maxrank=maxrank,
                )
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
        empty!(level_child_counts[level])
        sizehint!(level_child_counts[level], length(trialclusters))
        @inbounds for s in trialclusters
            if !isleaf(trialtree(tree), s) && isassigned(transfer, s)
                push!(level_transfer_nodes[level], s)
                push!(level_child_counts[level], length(transfer[s]))
            end
        end
    end

    basis_store = _build_basis_store(bases)
    transfer_store = _build_transfer_store(
        transfer, level_transfer_nodes, level_child_counts, trialtree(tree)
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
    tpivots = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(testtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(testtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(testtree(tree)))]
    level_child_counts = [Int[] for _ in 1:length(levels(testtree(tree)))]

    for level in reverse(levels(testtree(tree)))
        testclusters = collect(LevelIterator(testtree(tree), level))
        @tasks for t in testclusters
            @set scheduler = scheduler
            Ft = farfield(testtree(tree), fardata, t)
            if !isempty(Ft)
                tvalues = H2Trees.values(testtree(tree), t)
                adaptedFt = adapt_farfield_indices(compressor, trialtree(tree), Ft)
                tpivots[t], _ = compute_test_pivots!(
                    compressor.factorization,
                    nothing,
                    farmatrix,
                    tree,
                    tvalues,
                    adaptedFt,
                    buffer[1],
                    buffer[2];
                    maxrank=maxrank,
                )
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
        empty!(level_child_counts[level])
        sizehint!(level_child_counts[level], length(testclusters))
        @inbounds for t in testclusters
            if !isleaf(testtree(tree), t) && isassigned(transfer, t)
                push!(level_transfer_nodes[level], t)
                push!(level_child_counts[level], length(transfer[t]))
            end
        end
    end

    basis_store = _build_basis_store(bases)
    transfer_store = _build_transfer_store(
        transfer, level_transfer_nodes, level_child_counts, testtree(tree)
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
    spivots = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(trialtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(trialtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(trialtree(tree)))]
    level_child_counts = [Int[] for _ in 1:length(levels(trialtree(tree)))]

    for level in reverse(levels(trialtree(tree)))
        trialclusters = collect(LevelIterator(trialtree(tree), level))
        @tasks for s in trialclusters
            @set scheduler = scheduler
            Fs = farfield(trialtree(tree), fardata, s)
            if !isempty(Fs)
                svalues = H2Trees.values(trialtree(tree), s)
                adaptedFs = adapt_farfield_indices(compressor, testtree(tree), Fs)
                _, spivots[s] = compute_trial_pivots!(
                    compressor.factorization,
                    nothing,
                    farmatrix,
                    tree,
                    adaptedFs,
                    svalues,
                    buffer[1],
                    buffer[2];
                    maxrank=maxrank,
                )
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
        empty!(level_child_counts[level])
        sizehint!(level_child_counts[level], length(trialclusters))
        @inbounds for s in trialclusters
            if !isleaf(trialtree(tree), s) && isassigned(transfer, s)
                push!(level_transfer_nodes[level], s)
                push!(level_child_counts[level], length(transfer[s]))
            end
        end
    end

    basis_store = _build_basis_store(bases)
    transfer_store = _build_transfer_store(
        transfer, level_transfer_nodes, level_child_counts, trialtree(tree)
    )
    return basis_store, transfer_store, spivots
end
