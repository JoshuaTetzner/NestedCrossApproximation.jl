import H2Trees: isleaf, testtree, trialtree, root, children

function nears!(
    tree,
    values::Vector{V},
    nearvalues::Vector{V},
    tnode::Int,
    snodes::V;
    isnear=H2Trees.isnear,
) where {V<:Vector{Int}}
    nearnodes = Int[]
    childnearnodes = Int[]
    for snode in snodes
        if isnear(testtree(tree), trialtree(tree), tnode, snode)
            if isleaf(testtree(tree), tnode) || isleaf(trialtree(tree), snode)
                push!(nearnodes, snode)
            else
                append!(childnearnodes, collect(children(trialtree(tree), snode)))
            end
        end
    end
    if nearnodes != []
        push!(nearvalues, H2Trees.values(trialtree(tree), nearnodes))
        push!(values, H2Trees.values(testtree(tree), tnode))
    end
    if childnearnodes != []
        for child in children(testtree(tree), tnode)
            nears!(tree, values, nearvalues, child, childnearnodes; isnear=isnear)
        end
    end
end

function nearinteractions(tree::H2Trees.BlockTree; isnear=H2Trees.isnear)
    !isnear(testtree(tree), trialtree(tree), root(testtree(tree)), root(trialtree(tree))) &&
        return Vector{Int}(), Vector{Int}[]
    values = Vector{Int}[]
    nearvalues = Vector{Int}[]
    nears!(
        tree,
        values,
        nearvalues,
        root(testtree(tree)),
        [root(trialtree(tree))];
        isnear=isnear,
    )
    return values, nearvalues
end
