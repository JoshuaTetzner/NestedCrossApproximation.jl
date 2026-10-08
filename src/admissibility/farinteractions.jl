import H2Trees: testtree, trialtree, root, children, numberofnodes

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

function farnodesfrompairs(
    farpairs::AbstractVector{<:Tuple{Int,Int}},
    ptr::AbstractVector{<:Integer},
    nodeidx::Int,
    farnodeidx::Int,
)
    nodes = Vector{Int}(undef, length(farpairs))
    writeptr = Int.(ptr)
    @inbounds for pair in farpairs
        n = pair[nodeidx]
        nodes[writeptr[n]] = pair[farnodeidx]
        writeptr[n] += 1
    end
    return nodes
end

function farpairs!(
    ttree,
    stree,
    tcounts,
    scounts,
    farpairs::Vector{Tuple{Int,Int}},
    tnode::Int,
    snode::Int;
    isnear=isnear(),
)
    if !isnear(ttree, stree, tnode, snode)
        push!(farpairs, (tnode, snode))
        tcounts[tnode] += 1
        scounts[snode] += 1
    else
        for tchild in children(ttree, tnode)
            for schild in children(stree, snode)
                farpairs!(
                    ttree, stree, tcounts, scounts, farpairs, tchild, schild; isnear=isnear
                )
            end
        end
    end
end

function farinteractions(tree::BlockTree; isnear=isnear())
    testcounts = zeros(Int, numberofnodes(testtree(tree)))
    trialcounts = zeros(Int, numberofnodes(trialtree(tree)))
    if !isnear(testtree(tree), trialtree(tree), root(testtree(tree)), root(trialtree(tree)))
        testcounts[root(testtree(tree))] += 1
        trialcounts[root(trialtree(tree))] += 1
        testptr = ptrfromcounts(testcounts)
        trialptr = ptrfromcounts(trialcounts)
        return FarData(testptr, [root(trialtree(tree))]),
        FarData(trialptr, [root(testtree(tree))])
    end
    farpairs = Tuple{Int,Int}[]
    farpairs!(
        testtree(tree),
        trialtree(tree),
        testcounts,
        trialcounts,
        farpairs,
        root(testtree(tree)),
        root(trialtree(tree));
        isnear=isnear,
    )
    testptr = ptrfromcounts(testcounts)
    trialptr = ptrfromcounts(trialcounts)
    testfarnodes = farnodesfrompairs(farpairs, testptr, 1, 2)
    trialfarnodes = farnodesfrompairs(farpairs, trialptr, 2, 1)
    return FarData(testptr, testfarnodes), FarData(trialptr, trialfarnodes)
end

"""
    fardata(tree::BlockTree, isnear)

Determine the admissible cluster pairs of `tree` and return the far interactions of the
test and of the trial tree.

# Returns

  - Two [`FarData`](@ref) for the standard admissibility condition, two
    [`DirectionalData`](@ref) for [`isnearwideband`](@ref).
"""
function fardata(tree::BlockTree, isnear::IsNearFunctor)
    return farinteractions(tree; isnear=isnear)
end
