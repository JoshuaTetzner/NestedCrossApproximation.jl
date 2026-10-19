import H2Trees: testtree, trialtree, children, numberofnodes

"""
    FarData{I<:Integer}

Far interactions of the clusters of one tree in compressed sparse row layout.

The far nodes of `node`, i.e. the clusters of the other tree that form an admissible pair
with it, are `farnodes[nodetofarnodesptr[node]:(nodetofarnodesptr[node + 1] - 1)]`.

# Fields

  - `nodetofarnodesptr::Vector{I}`: position of the first far node of each node
  - `farnodes::Vector{I}`: far nodes of all nodes
"""
struct FarData{I<:Integer}
    nodetofarnodesptr::Vector{I}
    farnodes::Vector{I}
end

@inline farptr(data::FarData) = data.nodetofarnodesptr
@inline farnodes(data::FarData) = data.farnodes

function farnodes(data::FarData, node::Int)
    ptr = farptr(data)
    nodes = farnodes(data)
    return @view nodes[Int(ptr[node]):(Int(ptr[node + 1]) - 1)]
end

"""
    farfield(tree, fardata::FarData, node::Int)

Return the far field of `node`: its own far nodes followed by those of all its ancestors.

A nested basis has to represent the cluster with respect to this set, since the
interactions of the ancestors are passed down through the transfer matrices.
"""
function farfield(tree, fardata::FarData, node::Int)
    ptr = farptr(fardata)
    nodes = farnodes(fardata)

    total = ptr[node + 1] - ptr[node]
    for parent in H2Trees.ParentUpwardsIterator(tree, node)
        total += ptr[parent + 1] - ptr[parent]
    end

    result = Vector{Int}(undef, total)
    k = 0
    for i in ptr[node]:(ptr[node + 1] - 1)
        k += 1
        result[k] = nodes[i]
    end
    for parent in H2Trees.ParentUpwardsIterator(tree, node)
        for i in ptr[parent]:(ptr[parent + 1] - 1)
            k += 1
            result[k] = nodes[i]
        end
    end
    return result
end

function farinteractions(tree::BlockTree; isnear=isnear())
    return farinteractions(tree, interactions(tree, isnear))
end

function farinteractions(tree::BlockTree, partition)
    testptr, testnodes = partition.farptr, partition.farnodes
    ntrial = numberofnodes(trialtree(tree))
    trialptr = zeros(eltype(testptr), ntrial + 1)
    @inbounds for node in testnodes
        trialptr[node + 1] += 1
    end
    trialptr[1] = 1
    @inbounds for node in 1:ntrial
        trialptr[node + 1] += trialptr[node]
    end

    # Use the row starts as write cursors, then restore them without a scratch array.
    trialnodes = similar(testnodes)
    @inbounds for node in 1:(length(testptr) - 1)
        for k in testptr[node]:(testptr[node + 1] - 1)
            trialnode = testnodes[k]
            trialnodes[trialptr[trialnode]] = node
            trialptr[trialnode] += 1
        end
    end
    @inbounds for node in ntrial:-1:1
        trialptr[node + 1] = trialptr[node]
    end
    trialptr[1] = 1
    return FarData(testptr, testnodes), FarData(trialptr, trialnodes)
end

"""
    fardata(tree::BlockTree, isnear, partition=interactions(tree, isnear))

Determine the admissible cluster pairs of `tree` and return the far interactions of the
test and of the trial tree.

An existing ACA interaction partition can be reused. The test-side far arrays are
shared with it; directional subdivision may reorder its far nodes in place.

# Returns

  - Two [`FarData`](@ref) for the standard admissibility condition, two
    [`DirectionalData`](@ref) for [`isnearwideband`](@ref).
"""
function fardata(tree::BlockTree, isnear, partition=interactions(tree, isnear))
    return farinteractions(tree, partition)
end
