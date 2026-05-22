using H2Trees: H2Trees

struct DirectionalData{I<:Integer}
    fdata::FarData{I}      # farptr/fars bundled once via the generic FarData container
    dirptr::Vector{I}      # length nnodes+1;      node     → range in dirs/parentdir/dircounter
    dirs::Vector{I}        # direction-tree node indices, grouped by cluster node
    dircounter::Vector{I}  # number of fars per direction slot (same indexing as dirs)
    parentdirptr::Vector{I}  # length nnodes+1; parent node → range in parentdir
    parentdir::Vector{I}   # local parent-dir index per dir-slot (0 if none)
    isnear                 # near-interaction classifier functor

    function DirectionalData(
        fardata::FarData{I},
        dirptr::Vector{I},
        dirs::Vector{I},
        dircounter::Vector{I},
        parentdirptr::Vector{I},
        parentdir::Vector{I},
        isnear,
    ) where {I<:Integer}
        return new{I}(fardata, dirptr, dirs, dircounter, parentdirptr, parentdir, isnear)
    end
end

function fardata(tree::BlockTree, isnear::IsNearWidebandFunctor)
    ttree = H2Trees.testtree(tree)
    stree = H2Trees.trialtree(tree)
    testfardata, trialfardata = NestedCrossApproximation.farinteractions(
        tree; isnear=isnear
    )

    testdirdata = directionalfardata(testfardata, ttree, stree, isnear)
    trialdirdata = directionalfardata(trialfardata, stree, ttree, isnear)
    return testdirdata, trialdirdata
end

@inline fardata(data::DirectionalData) = data.fdata
@inline farptr(data::DirectionalData) = farptr(data.fdata)
@inline fars(data::DirectionalData) = fars(data.fdata)
@inline isnear(data::DirectionalData) = data.isnear

@inline function nfars(data::DirectionalData, node::Int)
    ptr = farptr(data)
    return Int(ptr[node + 1] - ptr[node])
end

@inline function farrange(data::DirectionalData, node::Int)
    ptr = farptr(data)
    return Int(ptr[node]):(Int(ptr[node + 1]) - 1)
end

# View of the reference-side far nodes for `node`.
@inline function fars(data::DirectionalData, node::Int)
    ptr = farptr(data)
    ff = fars(data)
    return @view ff[Int(ptr[node]):(Int(ptr[node + 1]) - 1)]
end

@inline function ndirections(data::DirectionalData, node::Int)
    return Int(data.dirptr[node + 1] - data.dirptr[node])
end

@inline function dirrange(data::DirectionalData, node::Int)
    return Int(data.dirptr[node]):(Int(data.dirptr[node + 1]) - 1)
end

# View of the direction indices for `node`.
@inline function dirs(data::DirectionalData, node::Int)
    return @view data.dirs[Int(data.dirptr[node]):(Int(data.dirptr[node + 1]) - 1)]
end

@inline function directionref(data::DirectionalData, node::Int, localdir::Int)
    return Int(data.dirs[Int(data.dirptr[node]) + localdir - 1])
end

# Local index of the parent's direction that contains localdir of node.
@inline function parentdirref(data::DirectionalData, node::Int, localdir::Int)
    return Int(data.parentdir[Int(data.parentdirptr[node]) + localdir - 1])
end

# View of the far nodes belonging to direction value `dir` of `node`.
# `dir` must be one of dirs(data, node). Fars within a node are stored
# contiguously sorted by direction.
function dirfars(data::DirectionalData, node::Int, dir::Int)
    node_dirs = dirs(data, node)
    localdir = findfirst(==(dir), node_dirs)
    isnothing(localdir) && error(
        "Direction $dir is not present in node $node. Use dirs(data, node) to query available directions.",
    )

    ptr = farptr(data)
    ff = fars(data)
    farstart = Int(ptr[node])
    base = Int(data.dirptr[node])
    for i in 1:(localdir - 1)
        farstart += Int(data.dircounter[base + i - 1])
    end
    count = Int(data.dircounter[base + localdir - 1])
    return @view ff[farstart:(farstart + count - 1)]
end

function diridxfromlocalfaridx(data::DirectionalData, node::Int, faridx::Int)
    drange = dirrange(data, node)
    dctr = data.dircounter[drange]
    @assert sum(dctr) >= faridx "Local far index $faridx exceeds total fars $(sum(dctr)) for node $node."

    for (diridx, count) in enumerate(dctr)
        faridx -= count
        if faridx <= 0
            return drange[diridx]
        end
    end
    return error("Local far index $faridx exceeds total fars $(sum(dctr)) for node $node.")
end

# Range of local dir indices of node whose parent direction is pdir.
function childdirrange(data::DirectionalData, node::Int, pdir::Int)
    block = data.parentdir[Int(data.parentdirptr[node]):(Int(data.parentdirptr[node + 1]) - 1)]
    return findall(==(pdir), block)
end

function dirmap(data::DirectionalData, node::Int)
    return data.parentdir[data.parentdirptr[node]:(data.parentdirptr[node + 1] - 1)]
end

# Return all far interactions associated with `dir` at `node`, including direct
# fars and inherited fars from paternal nodes.
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
            append!(ff, dirfars(data, current, 0))
            current = H2Trees.parent(tree, current)
        end
        return ff
    end

    append!(ff, dirfars(data, node, dir))
    #println("levelfars: ", length(ff))
    ##
    current = node
    diridcs = [localdir]
    while H2Trees.parent(tree, current) != 0
        dirmap = NestedCrossApproximation.dirmap(data, current)
        diridcs = findall(x -> x in diridcs, dirmap)
        dirs = NestedCrossApproximation.dirs(data, H2Trees.parent(tree, current))
        for dir in dirs[diridcs]
            append!(
                ff,
                NestedCrossApproximation.dirfars(data, H2Trees.parent(tree, current), dir),
            )
        end
        current = H2Trees.parent(tree, current)
    end
    ##
    #=current_node = node
    current_local = [Int(localdir)]
    while true
        pnode = H2Trees.parent(tree, current_node)
        pnode == 0 && break

        map = dirmap(data, current_node)
        parent_local = Int[]
        for (plocal, clocal) in enumerate(map)
            clocal == 0 && continue
            clocal in current_local || continue
            push!(parent_local, plocal)
        end
        isempty(parent_local) && break

        parent_dirs = dirs(data, pnode)
        for plocal in parent_local
            append!(ff, dirfars(data, pnode, Int(parent_dirs[plocal])))
        end

        current_node = pnode
        current_local = parent_local
    end=#

    return ff
end

# Compute the interaction vector from node in tree to farnode in fartree.
function interaction(node::Int, farnode::Int, tree, fartree)
    return H2Trees.center(tree, node) - H2Trees.center(fartree, farnode)
end

function isbasisnode(tree, data::DirectionalData, node::Int, dir::Int)
    H2Trees.isleaf(tree, node) && return true
    dir == 0 && return false
    firstchild = H2Trees.firstchild(tree, node)
    return (dirs(data, firstchild) == [0] || dirs(data, firstchild) == [])
end

function angle(a::SVector{D,F}, b::SVector{D,F}) where {D,F}
    return acos(clamp(dot(a, b) / (norm(a) * norm(b)), -1.0, 1.0))
end
