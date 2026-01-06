import H2Trees: isleaf, testtree, trialtree, root, children

function fars!(
    treea, treeb, farnodes::Vector{V}, tnode::Int, snodes::V; isnear=H2Trees.isnear
) where {V<:Vector{Int}}
    childnodes = Int[]
    localfarnodes = Int[]
    for snode in snodes
        if !isnear(treea, treeb, tnode, snode)
            push!(localfarnodes, snode)
        else
            append!(childnodes, collect(children(treeb, snode)))
        end
    end
    farnodes[tnode] = localfarnodes
    if childnodes != []
        for child in children(treea, tnode)
            fars!(treea, treeb, farnodes, child, childnodes; isnear=isnear)
        end
    end
end

function farinteractions(treea, treeb; isnear=H2Trees.isnear)
    farnodes = Vector{Vector{Int}}(undef, length(treea.nodes))
    !isnear(treea, treeb, root(treea), root(treeb)) &&
        (farnodes[root(treea)] = root(treeb); return farnodes)
    fars!(treea, treeb, farnodes, root(treea), [root(treeb)]; isnear=isnear)
    return farnodes
end
