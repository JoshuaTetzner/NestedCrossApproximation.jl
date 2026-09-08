using H2Trees: H2Trees

struct DirectionalData{I<:Integer}
    fardata::FarData{I}           # farptr/farnodes bundled once via the generic FarData container
    nodetodirsptr::Vector{I}      # length nnodes+1; node → range in dirs/diridxtofarptr
    dirs::Vector{I}               # direction-tree node indices, grouped by cluster node
    diridxtofarptr::Vector{I}     # length nodetodirsptr[end]; diridx → range in farnodes
    nodetochdirptr::Vector{I}     # length nnodes+1; node (=child) → range in partochddirmap, one entry per parent direction
    partochddirmap::Vector{I}     # parent's local dir slot (key) → this node's own matching local dir slot (value, 0 if none)
    isnear                        # near-interaction classifier functor

    function DirectionalData(
        fardata::FarData{I},
        nodetodirsptr::Vector{I},
        dirs::Vector{I},
        diridxtofarptr::Vector{I},
        nodetochdirptr::Vector{I},
        partochddirmap::Vector{I},
        isnear,
    ) where {I<:Integer}
        return new{I}(
            fardata,
            nodetodirsptr,
            dirs,
            diridxtofarptr,
            nodetochdirptr,
            partochddirmap,
            isnear,
        )
    end
end

# Shared by both directionalfardata builders (TwoNTree, BoundingBallTree): flattens the
# per-node ragged dirs/dircount/partochddirmap into DirectionalData's CSR storage.
# `dircount[node]` holds raw per-direction far-node counts (same indexing as dirs[node]);
# this turns those counts into the cumulative diridxtofarptr.
function _finalize_directionaldata(
    fardata::FarData, dirs, dircount, partochddirmap, partochddircount, isnear
)
    ptr = farptr(fardata)
    nnodes = length(dirs)

    nodetodirsptr = Vector{Int}(undef, nnodes + 1)
    nodetochdirptr = Vector{Int}(undef, nnodes + 1)
    nodetodirsptr[1] = 1
    nodetochdirptr[1] = 1
    for node in 1:nnodes
        nodetodirsptr[node + 1] = nodetodirsptr[node] + length(dirs[node])
        nodetochdirptr[node + 1] = nodetochdirptr[node] + partochddircount[node]
    end

    linedirs = Vector{Int}(undef, nodetodirsptr[end] - 1)
    diridxtofarptr = Vector{Int}(undef, nodetodirsptr[end])
    linepartochddirmap = Vector{Int}(undef, nodetochdirptr[end] - 1)

    for node in 1:nnodes
        first = nodetodirsptr[node]
        last = nodetodirsptr[node + 1] - 1
        if first <= last
            linedirs[first:last] = dirs[node]
            offset = Int(ptr[node])
            for (i, c) in enumerate(dircount[node])
                diridxtofarptr[first + i - 1] = offset
                offset += c
            end
            @assert offset == ptr[node + 1] "Direction counts for node $node do not sum to its far node count."
        end

        pfirst = nodetochdirptr[node]
        plast = nodetochdirptr[node + 1] - 1
        pfirst <= plast && (linepartochddirmap[pfirst:plast] = partochddirmap[node])
    end
    diridxtofarptr[end] = Int(ptr[end])

    return DirectionalData(
        fardata,
        nodetodirsptr,
        linedirs,
        diridxtofarptr,
        nodetochdirptr,
        linepartochddirmap,
        isnear,
    )
end

function fardata(tree::BlockTree, isnear::IsNearWidebandFunctor)
    ttree = H2Trees.testtree(tree)
    stree = H2Trees.trialtree(tree)
    testfardata, trialfardata = farinteractions(tree; isnear=isnear)

    testdirdata = directionalfardata(testfardata, ttree, stree, isnear)
    trialdirdata = directionalfardata(trialfardata, stree, ttree, isnear)
    return testdirdata, trialdirdata
end

@inline fardata(data::DirectionalData) = data.fardata
@inline farptr(data::DirectionalData) = farptr(data.fardata)
@inline farnodes(data::DirectionalData) = farnodes(data.fardata)
@inline isnear(data::DirectionalData) = data.isnear

@inline function nfarnodes(data::DirectionalData, node::Int)
    ptr = farptr(data)
    return Int(ptr[node + 1] - ptr[node])
end

@inline function farrange(data::DirectionalData, node::Int)
    ptr = farptr(data)
    return Int(ptr[node]):(Int(ptr[node + 1]) - 1)
end

# View of the reference-side far nodes for `node`.
@inline function farnodes(data::DirectionalData, node::Int)
    ptr = farptr(data)
    nodes = farnodes(data)
    return @view nodes[Int(ptr[node]):(Int(ptr[node + 1]) - 1)]
end

@inline function ndirections(data::DirectionalData, node::Int)
    return Int(data.nodetodirsptr[node + 1] - data.nodetodirsptr[node])
end

@inline function dirrange(data::DirectionalData, node::Int)
    return Int(data.nodetodirsptr[node]):(Int(data.nodetodirsptr[node + 1]) - 1)
end

# View of the direction indices for `node`.
@inline function dirs(data::DirectionalData, node::Int)
    return @view data.dirs[Int(data.nodetodirsptr[node]):(Int(data.nodetodirsptr[node + 1]) - 1)]
end

# View of the far nodes belonging to direction value `dir` of `node`.
# `dir` must be one of dirs(data, node). Far nodes within a node are stored
# contiguously sorted by direction.
function dirfarnodes(data::DirectionalData, node::Int, dir::Int)
    node_dirs = dirs(data, node)
    localdir = findfirst(==(dir), node_dirs)
    isnothing(localdir) && error(
        "Direction $dir is not present in node $node. Use dirs(data, node) to query available directions.",
    )

    diridx = Int(data.nodetodirsptr[node]) + localdir - 1
    farstart = Int(data.diridxtofarptr[diridx])
    farend = Int(data.diridxtofarptr[diridx + 1]) - 1
    return @view farnodes(data)[farstart:farend]
end

function diridxfromlocalfaridx(data::DirectionalData, node::Int, faridx::Int)
    nf = nfarnodes(data, node)
    @assert nf >= faridx "Local far index $faridx exceeds total far nodes $nf for node $node."

    remaining = faridx
    @inbounds for diridx in dirrange(data, node)
        remaining -= Int(data.diridxtofarptr[diridx + 1] - data.diridxtofarptr[diridx])
        if remaining <= 0
            return diridx
        end
    end
    return error("Local far index $faridx exceeds total far nodes $nf for node $node.")
end

function dirmap(data::DirectionalData, node::Int)
    return data.partochddirmap[data.nodetochdirptr[node]:(data.nodetochdirptr[node + 1] - 1)]
end

# Return all far interactions associated with `dir` at `node`, including direct
# far nodes and inherited far nodes from paternal nodes.
# HF case (`dir != 0`): propagate matching directions upward using dirmap.
# LF case (`dir == 0`): climb while ancestors stay LF (i.e. still have dir 0).
function dirfarfield(tree, data::DirectionalData, node::Int, dir::Int)
    nnodes = length(farptr(data)) - 1
    (1 <= node <= nnodes) ||
        error("Invalid node index $node for DirectionalData with $nnodes nodes.")

    node_dirs = dirs(data, node)
    localdir = findfirst(==(dir), node_dirs)
    isnothing(localdir) && error(
        "Direction $dir is not present in node $node. Use dirs(data, node) to query available directions.",
    )

    ff = Int[]
    if dir == 0
        current = node
        while current != 0
            current_dirs = dirs(data, current)
            if isempty(current_dirs) || findfirst(==(0), current_dirs) === nothing
                break
            end
            append!(ff, dirfarnodes(data, current, 0))
            current = H2Trees.parent(tree, current)
        end
        return ff
    end

    append!(ff, dirfarnodes(data, node, dir))
    current = node
    diridcs = [localdir]
    while H2Trees.parent(tree, current) != 0
        curdirmap = dirmap(data, current)
        diridcs = findall(x -> x in diridcs, curdirmap)
        pardirs = dirs(data, H2Trees.parent(tree, current))
        for pardir in pardirs[diridcs]
            append!(ff, dirfarnodes(data, H2Trees.parent(tree, current), pardir))
        end
        current = H2Trees.parent(tree, current)
    end

    return ff
end

# Compute the interaction vector from node in tree to farnode in fartree.
function interaction(node::Int, farnode::Int, tree, fartree)
    return H2Trees.center(tree, node) - H2Trees.center(fartree, farnode)
end

# Per-direction analogue of `_farworkload(tree, fardata::FarData)`: counts the
# (node, dir) pairs that do work in the DirectionalData traversal. Reuses `dirfarfield`
# instead of re-deriving its LF/HF inheritance rules.
function _farworkload(tree, fardata::DirectionalData)
    workload = 0
    for level in levels(tree)
        for t in LevelIterator(tree, level)
            for diridx in dirrange(fardata, t)
                isempty(dirfarfield(tree, fardata, t, fardata.dirs[diridx])) ||
                    (workload += 1)
            end
        end
    end
    return workload
end

function isbasisnode(tree, data::DirectionalData, node::Int, dir::Int)
    H2Trees.isleaf(tree, node) && return true
    dir == 0 && return false
    firstchild = H2Trees.firstchild(tree, node)
    return (dirs(data, firstchild) == [0] || dirs(data, firstchild) == [])
end
