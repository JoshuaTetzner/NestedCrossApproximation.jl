import H2Trees: isleaf, testtree, trialtree, root, children, numberofnodes

struct FarData{I<:Integer}
    farptr::Vector{I}
    fars::Vector{I}
end

@inline farptr(data::FarData) = data.farptr
@inline fars(data::FarData) = data.fars
function fars(data::FarData, node::Int)
    ptr = farptr(data)
    ff = fars(data)
    return @view ff[Int(ptr[node]):(Int(ptr[node + 1]) - 1)]
end

function farfield(tree, fardata::FarData, node::Int)
    fp = farptr(fardata)
    fs = fars(fardata)
    far_nodes = Int[]
    for i in fp[node]:(fp[node + 1] - 1)
        push!(far_nodes, fs[i])
    end
    for parent in H2Trees.ParentUpwardsIterator(tree, node)
        for i in fp[parent]:(fp[parent + 1] - 1)
            push!(far_nodes, fs[i])
        end
    end
    return far_nodes
end

function farptr(counter::AbstractVector{<:Integer})
    ptr = Vector{Int}(undef, length(counter) + 1)
    ptr[1] = 1
    @inbounds for i in eachindex(counter)
        ptr[i + 1] = ptr[i] + Int(counter[i])
    end
    return ptr
end

function reorder_fars(
    fars::AbstractVector{<:Tuple{Int,Int}},
    farptr::AbstractVector{<:Integer},
    tupleindex::Int,
)
    sorted = Vector{Tuple{Int,Int}}(undef, length(fars))
    writeptr = Int.(farptr)
    @inbounds for far in fars
        node = far[tupleindex]
        sorted[writeptr[node]] = far
        writeptr[node] += 1
    end
    return sorted
end

function fars!(
    ttree,
    stree,
    tfctr,
    sfctr,
    fars::Vector{Tuple{Int,Int}},
    tnode::Int,
    snode::Int;
    isnear=isnear(),
)
    if !isnear(ttree, stree, tnode, snode)
        push!(fars, (tnode, snode))
        tfctr[tnode] += 1
        sfctr[snode] += 1
    else
        for tchild in children(ttree, tnode)
            for schild in children(stree, snode)
                fars!(ttree, stree, tfctr, sfctr, fars, tchild, schild; isnear=isnear)
            end
        end
    end
end

function farinteractions(tree::BlockTree; isnear=isnear())
    testfarcounter = zeros(Int, H2Trees.numberofnodes(testtree(tree)))
    trialfarcounter = zeros(Int, H2Trees.numberofnodes(trialtree(tree)))
    if !isnear(testtree(tree), trialtree(tree), root(testtree(tree)), root(trialtree(tree)))
        testfarcounter[root(testtree(tree))] += 1
        trialfarcounter[root(trialtree(tree))] += 1
        testfarptr = farptr(testfarcounter)
        trialfarptr = farptr(trialfarcounter)
        return FarData(testfarptr, [root(trialtree(tree))]),
        FarData(trialfarptr, [root(testtree(tree))])
    end
    fars = Tuple{Int,Int}[]
    fars!(
        testtree(tree),
        trialtree(tree),
        testfarcounter,
        trialfarcounter,
        fars,
        root(testtree(tree)),
        root(trialtree(tree));
        isnear=isnear,
    )
    testfarptr = farptr(testfarcounter)
    trialfarptr = farptr(trialfarcounter)
    testfars_sorted = reorder_fars(fars, testfarptr, 1)
    trialfars_sorted = reorder_fars(fars, trialfarptr, 2)
    testfars = [t[2] for t in testfars_sorted]
    trialfars = [t[1] for t in trialfars_sorted]
    return FarData(testfarptr, testfars), FarData(trialfarptr, trialfars)
end

function fardata(tree::BlockTree, isnear::IsNearFunctor)
    return farinteractions(tree; isnear=isnear)
end
