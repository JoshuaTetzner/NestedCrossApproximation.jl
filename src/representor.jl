struct TreeMimicryRepresentor
    pivoting::TreeMimicryPivoting
end

function (tpr::TreeMimicryRepresentor)(node::Int, fars::Vector{Int}, nidcs::Int)
    newidcs = zeros(Int, nidcs)

    #center of normal pivoting
    ref = tpr.pivoting.tree.nodes[node].node.data.ct
    cts = [tpr.pivoting.tree.nodes[node].node.data.ct for node in fars]

    newidcs[1], data = tpr.pivoting(ref, fars, cts)
    for i in 2:nidcs
        newidcs[i], data = tpr.pivoting(ref, fars, cts, newidcs[1:(i - 1)], data)
    end

    return newidcs
end
