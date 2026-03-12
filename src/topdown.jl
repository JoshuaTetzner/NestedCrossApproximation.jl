import H2Trees: levels, LevelIterator

mutable struct TopDown{LowRankFactorizationType,RepresentorType}
    factorization::LowRankFactorizationType
    representor::RepresentorType

    function TopDown(factorization, representor)
        return new{typeof(factorization),typeof(representor)}(factorization, representor)
    end
end

function TopDown(; factorization=AdaptiveCrossApproximation.ACA(), representor=nothing)
    return TopDown(factorization, representor)
end

## testtree

function (compressor::TopDown)(
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::FarData,
    buffer::Tuple{Tuple{K,K},Channel{K}};
    scheduler=DynamicScheduler(),
    maxrank=40,
) where {T,K<:Matrix{T}}
    factorizationpool = factorization_channel(
        compressor; scheduler=scheduler, maxrank=maxrank
    )

    tpivots = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    Ftpivots = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(testtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(testtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(testtree(tree)))]
    level_child_counts = [Int[] for _ in 1:length(levels(testtree(tree)))]
    fp = farptr(fardata)
    fs = fars(fardata)
    for level in levels(testtree(tree))
        testclusters = collect(LevelIterator(testtree(tree), level))
        @tasks for t in testclusters
            @set scheduler = scheduler
            Ft = collect(fs[fp[t]:(fp[t + 1] - 1)])
            Ftvalues = H2Trees.values(trialtree(tree), Ft)
            isassigned(Ftpivots, H2Trees.parent(testtree(tree), t)) &&
                append!(Ftvalues, Ftpivots[H2Trees.parent(testtree(tree), t)])
            if !isempty(Ftvalues)
                tvalues = H2Trees.values(testtree(tree), t)
                factorization = take!(factorizationpool)
                try
                    tpivots[t], Ftpivots[t] = compute_test_pivots!(
                        factorization,
                        compressor.representor,
                        farmatrix,
                        tree,
                        tvalues,
                        Ftvalues,
                        buffer[1][bufferidx(level)],
                        buffer[2],
                    )
                finally
                    put!(factorizationpool, factorization)
                end
                if isleaf(H2Trees.testtree(tree), t)
                    bases[t] = nestedtestbasis(
                        tvalues, tpivots[t], buffer[1][bufferidx(level)]
                    )
                end
            end
        end
        sizehint!(level_transfer_nodes[level], length(testclusters))
        @inbounds for t in testclusters
            if !isleaf(H2Trees.testtree(tree), t) && isassigned(tpivots, t)
                push!(level_transfer_nodes[level], t)
            end
        end
        level != 1 && (
            level_child_counts[level - 1] = testtransfermatrices!(
                testtree(tree),
                transfer,
                level_transfer_nodes[level - 1],
                tpivots,
                buffer[1][bufferidx(level - 1)];
                scheduler=scheduler,
            )
        )
    end
    basis_store = _build_basis_store(bases)
    transfer_store = _build_transfer_store(
        transfer, level_transfer_nodes, level_child_counts, testtree(tree)
    )
    return basis_store, transfer_store, tpivots
end

function (compressor::TopDown)(
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::FarData,
    buffer::Tuple{Channel{K},Tuple{K,K}};
    scheduler=DynamicScheduler(),
    maxrank=40,
) where {T,K<:Matrix{T}}
    factorizationpool = factorization_channel(
        compressor; scheduler=scheduler, maxrank=maxrank
    )

    Fspivots = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    spivots = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(trialtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(trialtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(trialtree(tree)))]
    level_child_counts = [Int[] for _ in 1:length(levels(trialtree(tree)))]
    fp = farptr(fardata)
    fs = fars(fardata)
    for level in levels(trialtree(tree))
        trialclusters = collect(LevelIterator(trialtree(tree), level))
        @tasks for s in trialclusters
            @set scheduler = scheduler
            Fs = collect(fs[fp[s]:(fp[s + 1] - 1)])
            Fsvalues = H2Trees.values(testtree(tree), Fs)
            isassigned(Fspivots, H2Trees.parent(trialtree(tree), s)) &&
                append!(Fsvalues, Fspivots[H2Trees.parent(trialtree(tree), s)])
            if !isempty(Fsvalues)
                svalues = H2Trees.values(trialtree(tree), s)
                factorization = take!(factorizationpool)
                try
                    Fspivots[s], spivots[s] = compute_trial_pivots!(
                        factorization,
                        compressor.representor,
                        farmatrix,
                        tree,
                        Fsvalues,
                        svalues,
                        buffer[1],
                        buffer[2][bufferidx(level)];
                        maxrank=maxrank,
                    )
                finally
                    put!(factorizationpool, factorization)
                end
                if isleaf(H2Trees.trialtree(tree), s)
                    bases[s] = nestedtrialbasis(
                        svalues, spivots[s], buffer[2][bufferidx(level)]
                    )
                end
            end
        end
        sizehint!(level_transfer_nodes[level], length(trialclusters))
        @inbounds for s in trialclusters
            if !isleaf(H2Trees.trialtree(tree), s) && isassigned(spivots, s)
                push!(level_transfer_nodes[level], s)
            end
        end
        level != 1 && (
            level_child_counts[level - 1] = trialtransfermatrices!(
                trialtree(tree),
                transfer,
                level_transfer_nodes[level - 1],
                spivots,
                buffer[2][bufferidx(level - 1)];
                scheduler=scheduler,
            )
        )
    end
    basis_store = _build_basis_store(bases)
    transfer_store = _build_transfer_store(
        transfer, level_transfer_nodes, level_child_counts, trialtree(tree)
    )
    return basis_store, transfer_store, spivots
end
