using LinearAlgebra: dot, norm
using StaticArrays: SVector
import H2Trees: TwoNTree

mutable struct DirectionNode{N,T}
    level::Int
    axis::Int
    dir::SVector{N,T}
    parent::Int
    firstchild::Int
    nchildren::Int
end

"""
    DirectionTree{N,T}

Hierarchy of directions for the directional subdivision on a `TwoNTree`.

The roots are the `2N` axis directions, the centres of the faces of a cube. Each level
subdivides a face patch into `2^(N-1)` patches, so that the directions of a level are
nested in those of the previous one. The roots belong to the finest high-frequency level
of the cluster tree, and every coarser level uses the next level of directions.

# Fields

  - `level::Int`: depth of the direction tree
  - `dirroot::Int`: level of the cluster tree that uses the root directions
  - `rootcount::Int`: number of root directions
  - `branching::Int`: number of children of a direction
  - `nodes::Vector{DirectionNode{N,T}}`: directions, level by level
"""
struct DirectionTree{N,T}
    level::Int       # max depth of direction tree
    dirroot::Int     # coarsest cluster tree level that is HF (direction tree root)
    rootcount::Int
    branching::Int
    nodes::Vector{DirectionNode{N,T}}
end

@inline _rootcount(::Val{N}) where {N} = 2 * N
@inline _branching(::Val{N}) where {N} = Int(1) << (N - 1)

function _total_nodes(rootcount::Int, branching::Int, maxlevel::Int)
    return rootcount * ((branching^maxlevel - 1) ÷ (branching - 1))
end

function _root_directions(::Val{N}, ::Type{T}) where {N,T}
    dirs = Vector{Tuple{Int,SVector{N,T}}}(undef, _rootcount(Val(N)))
    idx = 1
    for axis in 1:N
        pos = SVector{N,T}(ntuple(i -> i == axis ? one(T) : zero(T), N))
        neg = SVector{N,T}(ntuple(i -> i == axis ? -one(T) : zero(T), N))
        dirs[idx] = (axis, pos)
        idx += 1
        dirs[idx] = (axis, neg)
        idx += 1
    end
    return dirs
end

function _child_offsets(::Val{N}, axis::Int, ::Type{T}) where {N,T}
    nchildren = _branching(Val(N))
    offsets = Vector{SVector{N,T}}(undef, nchildren)
    for mask in 0:(nchildren - 1)
        offsets[mask + 1] = SVector{N,T}(
            ntuple(N) do d
                if d == axis
                    return zero(T)
                end
                bitpos = d < axis ? d : d - 1
                return ((mask >>> (bitpos - 1)) & 0x1) == 0x0 ? -one(T) : one(T)
            end,
        )
    end
    return offsets
end

function DirectionTree(
    ::Val{N}, maxlevel::Int, dirroot::Int=1, ::Type{T}=Float64
) where {N,T<:Real}
    N < 2 && error("TwoN directional subdivision requires dimension N >= 2.")
    maxlevel < 1 && error("Directional tree maxlevel must be >= 1.")

    rootcount = _rootcount(Val(N))
    branching = _branching(Val(N))
    nnodes = _total_nodes(rootcount, branching, maxlevel)
    nodes = Vector{DirectionNode{N,T}}(undef, nnodes)
    childoffsets = [_child_offsets(Val(N), axis, T) for axis in 1:N]

    rootdirs = _root_directions(Val(N), T)
    nextidx = 1
    for (axis, dirvec) in rootdirs
        nodes[nextidx] = DirectionNode{N,T}(1, axis, dirvec, 0, 0, 0)
        nextidx += 1
    end

    levelstart = 1
    levelcount = rootcount
    scale = one(T) / T(2)
    for level in 1:(maxlevel - 1)
        levelend = levelstart + levelcount - 1
        for nodeidx in levelstart:levelend
            node = nodes[nodeidx]
            node.firstchild = nextidx
            node.nchildren = branching
            for offset in childoffsets[node.axis]
                nodes[nextidx] = DirectionNode{N,T}(
                    level + 1, node.axis, node.dir + scale * offset, nodeidx, 0, 0
                )
                nextidx += 1
            end
        end
        levelstart = levelend + 1
        levelcount *= branching
        scale /= T(2)
    end

    @assert nextidx == nnodes + 1

    # `_descend_direction` compares dot products, which requires unit vectors; the patch
    # centres `parent.dir + scale * offset` are not.
    @inbounds for node in nodes
        node.dir /= norm(node.dir)
    end

    return DirectionTree{N,T}(maxlevel, dirroot, rootcount, branching, nodes)
end

# dirroot: finest HF level, whose clusters use the 2N root directions.
# maxlevel: depth of the direction tree, the number of levels from the coarsest HF level
# with a far interaction to dirroot.
function _dirroot_and_maxlevel(tree::TwoNTree, ptr::AbstractVector{<:Integer}, islf)
    nnodes = H2Trees.numberofnodes(tree)
    length(ptr) == nnodes + 1 ||
        error("Invalid ptr length: expected $(nnodes + 1), got $(length(ptr)).")

    lflevel = H2Trees.minimumlevel(tree)
    while !islf(tree, lflevel)
        lflevel += 1
    end
    dirroot = lflevel - 1

    firsthflevel = dirroot
    @inbounds for node in 1:nnodes
        if ptr[node + 1] > ptr[node]
            lev = H2Trees.level(tree, node)
            !islf(tree, lev) && lev < firsthflevel && (firsthflevel = lev)
        end
    end

    return dirroot, dirroot - firsthflevel + 1
end

function DirectionTree(
    tree::TwoNTree{N,D,T}, ptr::AbstractVector{<:Integer}, islf
) where {N,D,T}
    dirroot, maxlvl = _dirroot_and_maxlevel(tree, ptr, islf)
    return DirectionTree(Val(N), maxlvl, dirroot, T)
end

@inline numberofnodes(tree::DirectionTree) = length(tree.nodes)
@inline rootnodes(tree::DirectionTree) = 1:(tree.rootcount)
@inline parent(tree::DirectionTree, node::Int) = tree.nodes[node].parent
@inline directionvector(tree::DirectionTree, node::Int) = tree.nodes[node].dir
@inline isleaf(tree::DirectionTree, node::Int) = tree.nodes[node].nchildren == 0

function children(tree::DirectionTree, node::Int)
    node == 0 && return rootnodes(tree)
    nchildren = tree.nodes[node].nchildren
    nchildren == 0 && return 1:0
    firstchild = tree.nodes[node].firstchild
    return firstchild:(firstchild + nchildren - 1)
end

# Direction node closest to `interaction`. The depth follows the level of the cluster:
# one at `tree.dirroot` and one more per coarser level.
function direction(
    interaction::SVector{N,T}, tree::DirectionTree{N}, clusterlevel::Int
) where {N,T}
    maxdepth = min(tree.dirroot - clusterlevel + 1, tree.level)
    return _descend_direction(interaction, tree, maxdepth)
end

function direction(interaction::SVector{N,T}, tree::DirectionTree{N}) where {N,T}
    return _descend_direction(interaction, tree, tree.level)
end

# For unit vectors the smallest angle is the largest dot product.
function _descend_direction(
    interaction::SVector{N,T}, tree::DirectionTree{N}, maxdepth::Int
) where {N,T}
    current = 0
    @inbounds for _ in 1:maxdepth
        nextnodes = children(tree, current)
        isempty(nextnodes) && break

        bestnode = first(nextnodes)
        bestdot = typemin(T)
        for candidate in nextnodes
            newdot = dot(interaction, tree.nodes[candidate].dir)
            if newdot > bestdot
                bestdot = newdot
                bestnode = candidate
            end
        end
        current = bestnode
    end
    return current
end

"""
    directionalfardata(fardata::FarData, tree, fartree, isnear)

Subdivide the far interactions of `tree` into directions and return a
[`DirectionalData`](@ref).

Each far node of a high-frequency cluster is assigned to the direction closest to the
vector between the two cluster centres, and the far nodes in `fardata` are reordered by
direction. A cluster also takes over the directions of its parent, which makes the
directional bases nested.

For a `TwoNTree` the directions are the nodes of a [`DirectionTree`](@ref). For a
`BoundingBallTree` each cluster has its own set of directions, whose number follows from
the wavenumber and the diameter of the cluster.
"""
function directionalfardata(fardata::FarData, tree::TwoNTree, fartree::TwoNTree, isnear)
    ptr = farptr(fardata)
    nodes = farnodes(fardata)
    dirtree = DirectionTree(tree, ptr, isnear.islf)
    nnodes = length(ptr) - 1
    dirs = Vector{Vector{Int}}(undef, nnodes)
    dircount = [Int[] for _ in 1:nnodes]
    partochddirmap = [Int[] for _ in 1:nnodes]
    partochddircount = zeros(Int, nnodes)

    for level in H2Trees.levels(tree)
        for node in H2Trees.LevelIterator(tree, level)
            pnode = H2Trees.parent(tree, node)
            hasfars = ptr[node + 1] > ptr[node]
            hasparentdirs = pnode != 0 && !isempty(dirs[pnode])

            # LF nodes carry the direction 0, if they have far nodes or an LF parent with it.
            if isnear.islf(tree, level)
                inheritlf = hasparentdirs && level > 1 && isnear.islf(tree, level - 1)
                dirs[node] = (hasfars || inheritlf) ? Int[0] : Int[]
                partochddirmap[node] = inheritlf ? [1] : Int[]
                continue
            end

            (!hasfars && !hasparentdirs) && (dirs[node]=Int[]; continue)

            nfar = Int(ptr[node + 1] - ptr[node])
            nodedirs = Vector{Int}(undef, nfar)
            for (i, faridx) in enumerate(ptr[node]:(ptr[node + 1] - 1))
                vec = interaction(node, nodes[faridx], tree, fartree)
                nodedirs[i] = direction(vec, dirtree, level)
            end

            perm = sortperm(nodedirs)
            permute!(view(nodes, ptr[node]:(ptr[node + 1] - 1)), perm)
            uniquenodedirs = sort!(unique(nodedirs))
            noriginal = length(uniquenodedirs)

            # Inherit child-directions from parent.
            partochddir = iszero(pnode) ? Int[] : zeros(Int, length(dirs[pnode]))
            if hasparentdirs
                for (i, pardir) in enumerate(dirs[pnode])
                    partochddir[i] = parent(dirtree, pardir)
                end

                for (idx, cdir) in enumerate(partochddir)
                    !(cdir in uniquenodedirs) && push!(uniquenodedirs, cdir)
                    partochddir[idx] = findfirst(==(cdir), uniquenodedirs)
                end
            end
            # Number of far nodes per direction; the inherited directions have none.
            localdircount = zeros(Int, length(uniquenodedirs))
            sortednodedirs = view(nodedirs, perm)
            n = length(sortednodedirs)
            i = 1
            for uidx in 1:noriginal
                c = 0
                while i <= n && sortednodedirs[i] == uniquenodedirs[uidx]
                    c += 1
                    i += 1
                end
                localdircount[uidx] = c
            end

            partochddirmap[node] = partochddir
            partochddircount[node] = count(!iszero, partochddir)
            dirs[node] = uniquenodedirs
            dircount[node] = localdircount
        end
    end

    # All far nodes of an LF node belong to the direction 0.
    for node in eachindex(dirs)
        if !isempty(dirs[node]) && isempty(dircount[node])
            dircount[node] = [max(0, Int(ptr[node + 1] - ptr[node]))]
            partochddircount[node] = length(partochddirmap[node])
        end
    end

    return _finalize_directionaldata(
        fardata, dirs, dircount, partochddirmap, partochddircount, isnear
    )
end
