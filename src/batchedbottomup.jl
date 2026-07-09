import H2Trees: levels, LevelIterator, nodes

"""
    BatchedBottomUp{PivotSelectorType,CompressorType}

Bottom-up nested-basis compressor that processes a whole tree level as one
batch, unlike `BottomUp`'s single interleaved per-node ACA/iACA pass.

  - `pivotselector`: purely geometric far-field candidate selection, touching
    no matrix entries (e.g. `AdaptiveCrossApproximation.TreeMimicryPivoting`).
  - `compressor`: dense-block factorization run on the batch-assembled block
    (e.g. fully pivoted ACA).

No `representor` field: geometric selection isn't an optional pre-reduction
layered on an on-the-fly factorization (as in `BottomUp`) — it's what decides
what gets batch-evaluated in the first place. Backend-agnostic: only the
concrete `pivotselector`/`compressor` objects and the `farmatrix` type passed
to `testbases`/`trialbases` determine CPU vs GPU.
"""
mutable struct BatchedBottomUp{PivotSelectorType,CompressorType}
    pivotselector::PivotSelectorType
    compressor::CompressorType

    function BatchedBottomUp(pivotselector, compressor)
        return new{typeof(pivotselector),typeof(compressor)}(pivotselector, compressor)
    end
end

function BatchedBottomUp(; pivotselector, compressor)
    return BatchedBottomUp(pivotselector, compressor)
end

# Per level, leaf → root (like BottomUp):
#   Phase 1 — geometric far-field pivot selection (`pivotselector`). Runs over
#             all nodes at once rather than per level, since it only needs
#             far-side candidates. Own-side reference is always the full
#             leaf-level point set, even for non-leaf nodes — a known,
#             separate limitation from Phase 2/3 below (fixing it needs
#             restructuring this phase to run per level too).
#   Phase 2 — batched dense-block evaluation via `farmatrix`, one block per
#             active node. Own-side indices are the full leaf point set for
#             leaves, but the children's already-selected pivots for non-leaf
#             nodes (available since levels run leaf → root) — this is what
#             makes the result a genuine nested (H2-style) basis. Filling is
#             delegated to `assemble_blocks`, so CPU/GPU dispatch safety (e.g.
#             the forced-serial GPU path - concurrent CUDA launches from
#             multiple threads corrupt results without device/stream
#             coordination) lives there rather than being duplicated here.
#   Phase 3 — `compressor.compressor` (e.g. fully pivoted ACA) selects each
#             node's own pivots from the Phase 2 block, then builds its leaf
#             basis or transfer matrices via BottomUp's
#             `nestedtestbasis`/`testtransfer`/`nestedtrialbasis`/
#             `trialtransfer`, unchanged — those only index into a buffer, so
#             local block indices work the same as BottomUp's global dof
#             indices. `compressor.compressor` mutates its input in place and
#             `blocks` must survive for the slicing below, so one scratch
#             buffer is reused per level instead of copying every block.

# One node's Phase 1: up to `maxrank` far-field candidates from `idcs` (raw
# far-tree node ids), mimicking `refidcs` (point-level own indices). `fartree`
# only bounds how many point-level pivots the candidate nodes can provide.
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
    scheduler=DynamicScheduler(),
    maxrank=40,
    pbar=Progress(0; enabled=false),
) where {T}
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

        # Own-side reference indices for block assembly: leaves use the full
        # leaf-level point set, non-leaf nodes use the pivots already
        # selected for their children. `childranges[i][j]` records where
        # child j's rows land within `tvalues[i]`/`blocks[i]`'s local
        # indexing, since (unlike BottomUp's tree-wide buffer) a child's
        # global pivot indices aren't valid row indices into a node-local
        # block. `nodechildren[i]` is cached so the child order here matches
        # what Phase 3 consumes below.
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

        # Phase 2 — one block per active node (own tvalues × its farpivots).
        blocks = [
            zeros(T, length(tvalues[i]), length(blockpivots[i])) for
            i in eachindex(activeclusters)
        ]
        assemble_blocks(farmatrix, blocks, tvalues, blockpivots; scheduler=scheduler)

        # Phase 3 — select pivots, then build the leaf basis or transfer
        # matrices straight from the block above.
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
    scheduler=DynamicScheduler(),
    maxrank=40,
    pbar=Progress(0; enabled=false),
) where {T}
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

        # Own-side reference indices, mirroring testbases: leaves use the
        # full leaf-level point set, non-leaf nodes use the pivots already
        # selected for their children. `childranges[i][j]`/`nodechildren[i]`
        # play the same role as in testbases.
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

        # Phase 2 — one block per active node (own svalues × its farpivots).
        blocks = [
            zeros(T, length(svalues[i]), length(blockpivots[i])) for
            i in eachindex(activeclusters)
        ]
        assemble_blocks(farmatrix, blocks, svalues, blockpivots; scheduler=scheduler)

        # Phase 3 — mirrors testbases, but `nestedtrialbasis`/`trialtransfer`
        # expect a buffer shaped (far-pivot-rank × own-trial-dofs), the
        # transpose of `blocks[i]`'s own orientation, so the local slice is
        # transposed before use. `transpose`, not adjoint: `T` is typically
        # complex (BEM) and adjoint would wrongly conjugate the entries.
        scratch = Matrix{T}(undef, maximum(length, svalues), maximum(length, blockpivots))
        for i in eachindex(activeclusters)
            s = activeclusters[i]
            m, n = size(blocks[i])
            Ascratch = view(scratch, 1:m, 1:n)
            copyto!(Ascratch, blocks[i])
            _, rowpivots, colpivots = compressor.compressor(Ascratch, maxrank)

            spivots[s] = svalues[i][rowpivots]
            localbuffer = Matrix(transpose(blocks[i][:, colpivots]))

            if isleaf(trialtree(tree), s)
                bases[s] = nestedtrialbasis(collect(1:m), rowpivots, localbuffer)
            else
                children = nodechildren[i]
                nodetransfers = Vector{Matrix{T}}(undef, length(children))
                for j in eachindex(children)
                    nodetransfers[j] = trialtransfer(
                        rowpivots, childranges[i][j], localbuffer
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

# Same cost model as BottomUp: one active cluster is about one
# coupling-matrix-equivalent of work, whether it runs as a per-node task
# (BottomUp) or as a share of a level-wide batch (BatchedBottomUp).
_basesworkload(::BatchedBottomUp, tree, fardata::FarData) = _farworkload(tree, fardata)
