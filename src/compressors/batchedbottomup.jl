import H2Trees: levels, LevelIterator

"""
    BatchedBottomUp{PivotSelectorType,CompressorType}

Compressor that constructs the nested bases from the leaves to the root and evaluates
the matrix entries of a whole level in one batch.

The construction has three phases:

 1. The far-field pivots of every cluster are selected from the geometry alone, without
    matrix entries.
 2. For each cluster of a level one dense block against its far-field pivots is evaluated.
    Its other indices are all points of a leaf, or the pivots of the children.
 3. The pivots of the cluster are selected in this block, and its basis or transfer
    matrices are computed from it.

# Fields

  - `pivotselector::PivotSelectorType`: geometric pivoting strategy for the first phase,
    e.g. `AdaptiveCrossApproximation.TreeMimicryPivoting`
  - `compressor::CompressorType`: factorization of a dense block for the third phase, e.g.
    a fully pivoted `ACA`

# Notes

The batched evaluation leaves the choice between CPU and GPU to the kernel matrix.
Directional H²-matrices require [`BottomUp`](@ref).
"""
mutable struct BatchedBottomUp{PivotSelectorType,CompressorType}
    pivotselector::PivotSelectorType
    compressor::CompressorType

    function BatchedBottomUp(pivotselector, compressor)
        return new{typeof(pivotselector),typeof(compressor)}(pivotselector, compressor)
    end
end

"""
    BatchedBottomUp(; pivotselector, compressor)

Build a [`BatchedBottomUp`](@ref) compressor from keyword arguments.

# Keyword Arguments

  - `pivotselector`: geometric pivoting strategy that selects the far-field pivots
  - `compressor`: factorization of a dense block
"""
function BatchedBottomUp(; pivotselector, compressor)
    return BatchedBottomUp(pivotselector, compressor)
end

# Phase 1 for one node: up to `maxrank` points out of the far nodes `idcs`.
function _selectfarpivots(
    pivotselector, refidcs::Vector{Int}, idcs::Vector{Int}, fartree, maxrank::Int
)
    n = min(maxrank, length(refidcs), sum(_clustersize(fartree, node) for node in idcs))
    n == 0 && return Int[]

    functor = pivotselector(refidcs, idcs, n)
    pivots = Vector{Int}(undef, n)
    pivots[1] = functor()
    for npivot in 2:n
        pivots[npivot] = functor(npivot)
    end
    return pivots
end

function testbases(
    compressor::BatchedBottomUp,
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::FarData;
    ntasks::Integer=Threads.nthreads(),
    maxrank=40,
    pbar=Progress(0; enabled=false),
) where {T}
    scheduler = _scheduler(ntasks)
    farpivots = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    @tasks for t in eachindex(farpivots)
        @set scheduler = scheduler
        Ft = farfield(testtree(tree), fardata, t)
        if !isempty(Ft)
            tvalues = H2Trees.values(testtree(tree), t)
            farpivots[t] = _selectfarpivots(
                compressor.pivotselector, tvalues, Ft, trialtree(tree), maxrank
            )
        end
    end

    tpivots = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(testtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(testtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(testtree(tree)))]

    for level in reverse(levels(testtree(tree)))
        testclusters = collect(LevelIterator(testtree(tree), level))
        activeclusters = filter(
            t -> isassigned(farpivots, t) && !isempty(farpivots[t]), testclusters
        )
        isempty(activeclusters) && continue

        # Rows of each block: the points of a leaf, the pivots of the children otherwise.
        # `childranges[i][j]` is the range of child j in `tvalues[i]`.
        tvalues = Vector{Vector{Int}}(undef, length(activeclusters))
        childranges = Vector{Vector{UnitRange{Int}}}(undef, length(activeclusters))
        nodechildren = Vector{Vector{Int}}(undef, length(activeclusters))
        for (i, t) in enumerate(activeclusters)
            if isleaf(testtree(tree), t)
                tvalues[i] = H2Trees.values(testtree(tree), t)
                childranges[i] = UnitRange{Int}[]
                nodechildren[i] = Int[]
            else
                children = collect(H2Trees.ChildIterator(testtree(tree), t))
                vals = Int[]
                ranges = Vector{UnitRange{Int}}(undef, length(children))
                for (j, child) in enumerate(children)
                    start = length(vals) + 1
                    append!(vals, tpivots[child])
                    ranges[j] = start:length(vals)
                end
                tvalues[i] = vals
                childranges[i] = ranges
                nodechildren[i] = children
            end
        end
        blockpivots = [farpivots[t] for t in activeclusters]

        # Phase 2: one block per active node.
        blocks = [
            zeros(T, length(tvalues[i]), length(blockpivots[i])) for
            i in eachindex(activeclusters)
        ]
        assemble_blocks(farmatrix, blocks, tvalues, blockpivots; ntasks=ntasks)

        # Phase 3: the compressor works in place, so each block is copied to `scratch`.
        scratch = Matrix{T}(undef, maximum(length, tvalues), maximum(length, blockpivots))
        for i in eachindex(activeclusters)
            t = activeclusters[i]
            m, n = size(blocks[i])
            Ascratch = view(scratch, 1:m, 1:n)
            copyto!(Ascratch, blocks[i])
            _, rowpivots, colpivots = compressor.compressor(Ascratch, maxrank)

            tpivots[t] = tvalues[i][rowpivots]
            localbuffer = blocks[i][:, colpivots]

            if isleaf(testtree(tree), t)
                bases[t] = nestedtestbasis(collect(1:m), rowpivots, localbuffer)
            else
                children = nodechildren[i]
                nodetransfers = Vector{Matrix{T}}(undef, length(children))
                for j in eachindex(children)
                    nodetransfers[j] = testtransfer(
                        rowpivots, childranges[i][j], localbuffer
                    )
                end
                transfer[t] = nodetransfers
            end
            next!(pbar)
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
    compressor::BatchedBottomUp,
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::FarData;
    ntasks::Integer=Threads.nthreads(),
    maxrank=40,
    pbar=Progress(0; enabled=false),
) where {T}
    scheduler = _scheduler(ntasks)
    farpivots = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    @tasks for s in eachindex(farpivots)
        @set scheduler = scheduler
        Fs = farfield(trialtree(tree), fardata, s)
        if !isempty(Fs)
            svalues = H2Trees.values(trialtree(tree), s)
            farpivots[s] = _selectfarpivots(
                compressor.pivotselector, svalues, Fs, testtree(tree), maxrank
            )
        end
    end

    spivots = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(trialtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(trialtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(trialtree(tree)))]

    for level in reverse(levels(trialtree(tree)))
        trialclusters = collect(LevelIterator(trialtree(tree), level))
        activeclusters = filter(
            s -> isassigned(farpivots, s) && !isempty(farpivots[s]), trialclusters
        )
        isempty(activeclusters) && continue

        # Columns of each block, as the rows in testbases.
        svalues = Vector{Vector{Int}}(undef, length(activeclusters))
        childranges = Vector{Vector{UnitRange{Int}}}(undef, length(activeclusters))
        nodechildren = Vector{Vector{Int}}(undef, length(activeclusters))
        for (i, s) in enumerate(activeclusters)
            if isleaf(trialtree(tree), s)
                svalues[i] = H2Trees.values(trialtree(tree), s)
                childranges[i] = UnitRange{Int}[]
                nodechildren[i] = Int[]
            else
                children = collect(H2Trees.ChildIterator(trialtree(tree), s))
                vals = Int[]
                ranges = Vector{UnitRange{Int}}(undef, length(children))
                for (j, child) in enumerate(children)
                    start = length(vals) + 1
                    append!(vals, spivots[child])
                    ranges[j] = start:length(vals)
                end
                svalues[i] = vals
                childranges[i] = ranges
                nodechildren[i] = children
            end
        end
        blockpivots = [farpivots[s] for s in activeclusters]

        # Phase 2: the far pivots are test indices, the rows of the block.
        blocks = [
            zeros(T, length(blockpivots[i]), length(svalues[i])) for
            i in eachindex(activeclusters)
        ]
        assemble_blocks(farmatrix, blocks, blockpivots, svalues; ntasks=ntasks)

        # Phase 3: as in testbases with rows and columns exchanged.
        scratch = Matrix{T}(undef, maximum(length, blockpivots), maximum(length, svalues))
        for i in eachindex(activeclusters)
            s = activeclusters[i]
            n, m = size(blocks[i])
            Ascratch = view(scratch, 1:n, 1:m)
            copyto!(Ascratch, blocks[i])
            _, rowpivots, colpivots = compressor.compressor(Ascratch, maxrank)

            spivots[s] = svalues[i][colpivots]
            localbuffer = blocks[i][rowpivots, :]

            if isleaf(trialtree(tree), s)
                bases[s] = nestedtrialbasis(collect(1:m), colpivots, localbuffer)
            else
                children = nodechildren[i]
                nodetransfers = Vector{Matrix{T}}(undef, length(children))
                for j in eachindex(children)
                    nodetransfers[j] = trialtransfer(
                        colpivots, childranges[i][j], localbuffer
                    )
                end
                transfer[s] = nodetransfers
            end
            next!(pbar)
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
