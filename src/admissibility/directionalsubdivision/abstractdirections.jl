using H2Trees: H2Trees

"""
    DirectionalData{I<:Integer}

Far interactions of the clusters of one tree, subdivided into directions.

A cluster in the high-frequency regime holds a set of directions, and each of its far nodes
belongs to one of them. A cluster in the low-frequency regime holds the single direction
`0`. The position of a (cluster, direction) pair in `dirs` is its direction index, by which
the pivots, bases and transfer matrices of a directional H²-matrix are addressed.

# Fields

  - `fardata::FarData{I}`: far nodes of each node, grouped by direction
  - `nodetodirsptr::Vector{I}`: position of the first direction of each node in `dirs`
  - `dirs::Vector{I}`: directions of all nodes
  - `diridxtofarptr::Vector{I}`: position of the first far node of each direction index
  - `nodetochdirptr::Vector{I}`: position of the first entry of each node in `partochddirmap`
  - `partochddirmap::Vector{I}`: for each direction of the parent the matching local
    direction of the node, `0` if there is none
  - `isnear`: admissibility condition
"""
struct DirectionalData{I<:Integer}
    fardata::FarData{I}
    nodetodirsptr::Vector{I}
    dirs::Vector{I}
    diridxtofarptr::Vector{I}
    nodetochdirptr::Vector{I}
    partochddirmap::Vector{I}
    isnear

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

# Flattens the per-node vectors of the `directionalfardata` methods into the CSR layout;
# `dircount[node]` holds the number of far nodes per direction of `node`.
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

@inline function dirs(data::DirectionalData, node::Int)
    return @view data.dirs[Int(data.nodetodirsptr[node]):(Int(data.nodetodirsptr[node + 1]) - 1)]
end

# Far nodes of `node` in direction `dir`; they are stored grouped by direction.
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

"""
    dirfarfield(tree, data::DirectionalData, node::Int, dir::Int)

Return the far field of `node` in direction `dir`: its own far nodes in this direction
followed by those inherited from its ancestors.

For a high-frequency direction the matching directions of the ancestors are followed
upwards. For the low-frequency direction `0` the far nodes of the ancestors are collected
as long as these are in the low-frequency regime.
"""
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

# Vector from the centre of `farnode` to the centre of `node`.
function interaction(node::Int, farnode::Int, tree, fartree)
    return H2Trees.center(tree, node) - H2Trees.center(fartree, farnode)
end

function isbasisnode(tree, data::DirectionalData, node::Int, dir::Int)
    H2Trees.isleaf(tree, node) && return true
    dir == 0 && return false
    firstchild = H2Trees.firstchild(tree, node)
    return (dirs(data, firstchild) == [0] || dirs(data, firstchild) == [])
end
