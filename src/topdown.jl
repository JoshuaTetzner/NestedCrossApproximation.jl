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

function testbases(
    compressor::TopDown,
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::FarData;
    scheduler=DynamicScheduler(),
    maxrank=40,
    pbar=Progress(0; enabled=false),
) where {T}
    buffer = testbuffer(compressor, farmatrix, maxrank)

    tpivots = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    Ftpivots = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(testtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(testtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(testtree(tree)))]
    ptr = farptr(fardata)
    nodes = farnodes(fardata)
    nleaves = count(n -> isleaf(testtree(tree), n), 1:numberofnodes(testtree(tree)))
    for level in levels(testtree(tree))
        testclusters = collect(LevelIterator(testtree(tree), level))
        levelstep = max(1, nleaves ÷ length(testclusters))
        @tasks for t in testclusters
            @set scheduler = scheduler
            @local begin
                farbuffer = fartestbuffer(compressor.factorization, farmatrix, maxrank)
                factorization = _build_stateful_factorization(
                    compressor.factorization, farmatrix, maxrank
                )
                representor = _stateful_representor(compressor.representor, maxrank)
            end
            Ft = collect(nodes[ptr[t]:(ptr[t + 1] - 1)])
            Ftvalues = H2Trees.values(trialtree(tree), Ft)
            isassigned(Ftpivots, H2Trees.parent(testtree(tree), t)) &&
                append!(Ftvalues, Ftpivots[H2Trees.parent(testtree(tree), t)])
            if !isempty(Ftvalues)
                tvalues = H2Trees.values(testtree(tree), t)

                tpivots[t], Ftpivots[t] = compute_test_pivots!(
                    factorization,
                    representor,
                    farmatrix,
                    tree,
                    tvalues,
                    PreResolvedIndices(Ftvalues),
                    buffer[bufferidx(level)],
                    farbuffer;
                    maxrank=maxrank,
                )
                if isleaf(H2Trees.testtree(tree), t)
                    bases[t] = nestedtestbasis(
                        tvalues, tpivots[t], buffer[bufferidx(level)]
                    )
                end
                next!(pbar; step=levelstep)
            end
        end
        sizehint!(level_transfer_nodes[level], length(testclusters))
        @inbounds for t in testclusters
            if isassigned(tpivots, t)
                !isleaf(H2Trees.testtree(tree), t) && push!(level_transfer_nodes[level], t)
            end
        end
        level != 1 && testtransfermatrices!(
            testtree(tree),
            transfer,
            level_transfer_nodes[level - 1],
            tpivots,
            buffer[bufferidx(level - 1)];
            scheduler=scheduler,
        )
    end
    basis_store = _build_basis_storage(bases)
    transfer_store = _build_transfer_storage(transfer, level_transfer_nodes, testtree(tree))
    return basis_store, transfer_store, tpivots
end

function trialbases(
    compressor::TopDown,
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::FarData;
    scheduler=DynamicScheduler(),
    maxrank=40,
    pbar=Progress(0; enabled=false),
) where {T}
    buffer = trialbuffer(compressor, farmatrix, maxrank)

    Fspivots = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    spivots = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(trialtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(trialtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(trialtree(tree)))]
    ptr = farptr(fardata)
    nodes = farnodes(fardata)
    nleaves = count(n -> isleaf(trialtree(tree), n), 1:numberofnodes(trialtree(tree)))
    for level in levels(trialtree(tree))
        trialclusters = collect(LevelIterator(trialtree(tree), level))
        levelstep = max(1, nleaves ÷ length(trialclusters))
        @tasks for s in trialclusters
            @set scheduler = scheduler
            @local begin
                farbuffer = fartrialbuffer(compressor.factorization, farmatrix, maxrank)
                factorization = _build_stateful_factorization(
                    compressor.factorization, farmatrix, maxrank
                )
                representor = _stateful_representor(compressor.representor, maxrank)
            end
            Fs = collect(nodes[ptr[s]:(ptr[s + 1] - 1)])
            Fsvalues = H2Trees.values(testtree(tree), Fs)
            isassigned(Fspivots, H2Trees.parent(trialtree(tree), s)) &&
                append!(Fsvalues, Fspivots[H2Trees.parent(trialtree(tree), s)])
            if !isempty(Fsvalues)
                svalues = H2Trees.values(trialtree(tree), s)
                Fspivots[s], spivots[s] = compute_trial_pivots!(
                    factorization,
                    representor,
                    farmatrix,
                    tree,
                    PreResolvedIndices(Fsvalues),
                    svalues,
                    farbuffer,
                    buffer[bufferidx(level)];
                    maxrank=maxrank,
                )
                if isleaf(H2Trees.trialtree(tree), s)
                    bases[s] = nestedtrialbasis(
                        svalues, spivots[s], buffer[bufferidx(level)]
                    )
                end
                next!(pbar; step=levelstep)
            end
        end
        sizehint!(level_transfer_nodes[level], length(trialclusters))
        @inbounds for s in trialclusters
            if isassigned(spivots, s)
                !isleaf(H2Trees.trialtree(tree), s) && push!(level_transfer_nodes[level], s)
            end
        end
        level != 1 && trialtransfermatrices!(
            trialtree(tree),
            transfer,
            level_transfer_nodes[level - 1],
            spivots,
            buffer[bufferidx(level - 1)];
            scheduler=scheduler,
        )
    end
    basis_store = _build_basis_storage(bases)
    transfer_store = _build_transfer_storage(
        transfer, level_transfer_nodes, trialtree(tree)
    )
    return basis_store, transfer_store, spivots
end

# TopDown descends root-to-leaves: each active level has more clusters than the
# last, but each one is correspondingly cheaper (less work per cluster as Ftvalues
# narrows), so wall time per active level is roughly constant — about nleaves(tree)
# coupling-matrix-equivalents — rather than scaling with that level's cluster count.
# Mirrors the `levelactive`/`next!(pbar; step=nleaves)` runtime logic above so the
# dry-run total matches exactly what gets incremented.
function _basesworkload(::TopDown, tree, fardata::FarData)
    ptr = farptr(fardata)
    active = falses(numberofnodes(tree))
    nleaves = count(n -> isleaf(tree, n), 1:numberofnodes(tree))
    workload = 0
    for level in levels(tree)
        leveliter = LevelIterator(tree, level)
        levelstep = max(1, nleaves ÷ length(leveliter))
        for t in leveliter
            ownfar = ptr[t + 1] > ptr[t]
            parentactive = level == 1 ? false : active[H2Trees.parent(tree, t)]
            if ownfar || parentactive
                active[t] = true
                workload += levelstep
            end
        end
    end
    return workload
end
