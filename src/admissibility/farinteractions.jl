import H2Trees: testtree, trialtree, root, children, numberofnodes

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

function fardata(tree::BlockTree, isnear::IsNearFunctor)
    return farinteractions(tree; isnear=isnear)
end
