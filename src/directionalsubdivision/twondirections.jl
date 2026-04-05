using LinearAlgebra: dot
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
    return DirectionTree{N,T}(maxlevel, dirroot, rootcount, branching, nodes)
end

# Returns (dirroot, maxlevel) where:
#   dirroot  = lflevel - 1: finest HF level, where the direction tree is rooted
#              (clusters at this level use the 2N root directions)
#   maxlevel = dirroot - firsthflevel + 1: depth of the direction tree, determined
#              by the coarsest HF level that has a far interaction
function _dirroot_and_maxlevel(tree::TwoNTree, farptr::AbstractVector{<:Integer}, islf)
    nnodes = H2Trees.numberofnodes(tree)
    length(farptr) == nnodes + 1 ||
        error("Invalid farptr length: expected $(nnodes + 1), got $(length(farptr)).")

    # Find the first LF level (going coarse→fine, level numbers increase)
    lflevel = 1
    while !islf(tree, lflevel)
        lflevel += 1
    end
    # Direction tree root lives at the finest HF level, just above LF
    dirroot = lflevel - 1

    # Find the coarsest HF level that has a far interaction to bound the tree depth
    firsthflevel = dirroot  # sentinel; will be updated below
    @inbounds for node in 1:nnodes
        if farptr[node + 1] > farptr[node]
            lev = H2Trees.level(tree, node)
            !islf(tree, lev) && lev < firsthflevel && (firsthflevel = lev)
        end
    end

    return dirroot, dirroot - firsthflevel + 1
end

function DirectionTree(
    tree::TwoNTree{N,D,T}, farptr::AbstractVector{<:Integer}, islf
) where {N,D,T}
    dirroot, maxlvl = _dirroot_and_maxlevel(tree, farptr, islf)
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

# Find the direction node in `tree` that best matches `interaction`.
# `clusterlevel` is the level of the cluster in the cluster tree; the direction
# tree depth is derived as  min(tree.dirroot-clusterlevel + 1, tree.level),
# so finer cluster levels use finer directions, clamped to the direction tree depth.
function direction(
    interaction::SVector{N,T}, tree::DirectionTree{N}, clusterlevel::Int
) where {N,T}
    maxdepth = min(tree.dirroot - clusterlevel + 1, tree.level)
    return _descend_direction(interaction, tree, maxdepth)
end

# Variant without a cluster level: descend to full depth of the direction tree.
function direction(interaction::SVector{N,T}, tree::DirectionTree{N}) where {N,T}
    return _descend_direction(interaction, tree, tree.level)
end

function _descend_direction(
    interaction::SVector{N,T}, tree::DirectionTree{N}, maxdepth::Int
) where {N,T}
    current = 0
    @inbounds for _ in 1:maxdepth
        nextnodes = children(tree, current)
        isempty(nextnodes) && break

        bestnode = first(nextnodes)
        minangle = typemax(T)
        for candidate in nextnodes
            newangle = angle(interaction, tree.nodes[candidate].dir)
            if newangle < minangle
                minangle = newangle
                bestnode = candidate
            end
        end
        current = bestnode
    end
    return current
end

function directionalfardata(fardata::FarData, cltree::TwoNTree, reftree::TwoNTree, isnear)
    farptr = NestedCrossApproximation.farptr(fardata)
    fars = NestedCrossApproximation.fars(fardata)
    dirtree = DirectionTree(cltree, farptr, isnear.islf)
    nnodes = length(farptr) - 1
    dirs = Vector{Vector{Int}}(undef, nnodes)
    dircounter = [Int[] for _ in 1:nnodes]
    pdirmap = [Int[] for _ in 1:nnodes]
    pdircounter = zeros(Int, nnodes)
    for level in H2Trees.levels(cltree)
        for node in H2Trees.LevelIterator(cltree, level)
            pnode = H2Trees.parent(cltree, node)
            hasfars = farptr[node + 1] > farptr[node]
            hasparentdirs = pnode != 0 && !isempty(dirs[pnode])

            # LF nodes do not use directional subdivision.
            # Store [0] for direct LF fars, or when inheriting from an LF parent with dirs.
            if isnear.islf(cltree, level)
                inheritlf = hasparentdirs && level > 1 && isnear.islf(cltree, level - 1)
                dirs[node] = (hasfars || inheritlf) ? Int[0] : Int[]
                pdrimap[node] = inheritlf ? [1] : Int[]
                continue
            end

            (!hasfars && hasparentdirs) && (dirs[node] = Int[]; continue)

            nfars_node = Int(farptr[node + 1] - farptr[node])
            nodedirs = Vector{Int}(undef, nfars_node)
            for (i, faridx) in enumerate(farptr[node]:(farptr[node + 1] - 1))
                vec = interaction(node, fars[faridx], cltree, reftree)
                nodedirs[i] = direction(vec, dirtree, level)
            end

            perm = sortperm(nodedirs)
            permute!(view(fars, farptr[node]:(farptr[node + 1] - 1)), perm)
            uniquenodedirs = sort!(unique(nodedirs))

            # Inherit child-directions from parent.
            dirmap = iszero(pnode) ? Int[] : zeros(Int, length(dirs[pnode]))
            if hasparentdirs
                for (i, parentdir) in enumerate(dirs[pnode])
                    dirmap[i] = parent(dirtree, parentdir)
                end

                for (idx, dir) in enumerate(dirmap)
                    !(dir in uniquenodedirs) && push!(uniquenodedirs, dir)
                    dirmap[idx] = findfirst(==(dir), uniquenodedirs)
                end
            end
            localdircounter = zeros(Int, length(uniquenodedirs))
            for (uidx, udir) in enumerate(uniquenodedirs)
                for dir in nodedirs
                    udir == dir && (localdircounter[uidx] += 1)
                end
            end

            pdirmap[node] = dirmap
            pdircounter[node] = count(!iszero, dirmap)
            dirs[node] = uniquenodedirs
            dircounter[node] = localdircounter
        end
    end

    # Fill dircounter for LF sentinel nodes (dirs=[0], dircounter not set in second pass).
    for node in eachindex(dirs)
        if !isempty(dirs[node]) && isempty(dircounter[node])
            dircounter[node] = [max(0, Int(farptr[node + 1] - farptr[node]))]
        end
    end

    # Build linear direction storage.
    dirptr = Vector{Int}(undef, length(dirs) + 1)
    pdirptr = Vector{Int}(undef, length(pdirmap) + 1)
    dirptr[1] = 1
    pdirptr[1] = 1
    for node in eachindex(dirs)
        dirptr[node + 1] = dirptr[node] + length(dirs[node])
        pdirptr[node + 1] = pdirptr[node] + pdircounter[node]
    end
    linedirs = Vector{Int}(undef, dirptr[end] - 1)
    linedircounter = Vector{Int}(undef, dirptr[end] - 1)
    lineparentdir = Vector{Int}(undef, pdirptr[end] - 1)
    for node in eachindex(dirs)
        first = dirptr[node]
        last = dirptr[node + 1] - 1
        first > last && continue
        linedirs[first:last] = dirs[node]
        linedircounter[first:last] = dircounter[node]

        pfirst = pdirptr[node]
        plast = pdirptr[node + 1] - 1
        pfirst > plast && continue
        lineparentdir[pfirst:plast] = pdirmap[node]
    end

    linefardata = fardata

    return DirectionalData(
        linefardata, dirptr, linedirs, linedircounter, pdirptr, lineparentdir, isnear
    )
end
