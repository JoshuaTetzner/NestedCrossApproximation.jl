using FastBEAST
using ClusterTrees
import FastBEAST.NminClusterTrees.NminTree
import ClusterTrees.PointerBasedTrees
import ClusterTrees.PointerBasedTrees.Node

struct Data{T}
    face::Int
    ℯ::T
end

childfaces = [
    [
        SVector(1.0, 1.0, 0.0),
        SVector(-1.0, 1.0, 0.0),
        SVector(-1.0, -1.0, 0.0),
        SVector(1.0, -1.0, 0.0),
    ],
    [
        SVector(1.0, 0.0, 1.0),
        SVector(-1.0, 0.0, 1.0),
        SVector(-1.0, 0.0, -1.0),
        SVector(1.0, 0.0, -1.0),
    ],
    [
        SVector(0.0, 1.0, 1.0),
        SVector(0.0, -1.0, 1.0),
        SVector(0.0, -1.0, -1.0),
        SVector(0.0, 1.0, -1.0),
    ],
]

function split!(
    nodes::Vector{T}, node::Int, hs::F, ηₕ::F, k::F
) where {F,T<:(PointerBasedTrees.Node{Data{SVector{D,F}}} where {D})}
    diameter = 2√2hs
    nodes[node].first_child = length(nodes) + 1
    directions = [
        nodes[node].data.ℯ + childfaces[nodes[node].data.face][1] .* hs,
        nodes[node].data.ℯ + childfaces[nodes[node].data.face][2] .* hs,
        nodes[node].data.ℯ + childfaces[nodes[node].data.face][3] .* hs,
        nodes[node].data.ℯ + childfaces[nodes[node].data.face][4] .* hs,
    ]

    append!(
        nodes,
        [
            Node(Data(nodes[node].data.face, directions[1]), 4, length(nodes) + 2, node, 0)
            Node(Data(nodes[node].data.face, directions[2]), 4, length(nodes) + 3, node, 0)
            Node(Data(nodes[node].data.face, directions[3]), 4, length(nodes) + 4, node, 0)
            Node(Data(nodes[node].data.face, directions[4]), 4, 0, node, 0)
        ],
    )
    k * √3hs <= 1 && return nothing#diameter <= 2ηₕ / (k * 2 * √3hs) && return nothing
    nodeidcs = Vector((length(nodes) - 3):length(nodes))
    for i in nodeidcs
        split!(nodes, i, hs / 2, ηₕ, k)
    end
end

function directionaltree(hs::F, k::F, ηₕ::F) where {F}
    diameter = 2√2hs

    nodes = [
        Node(Data(0, SVector(0, 0, 0.0)), 6, 2, 0, 0)
        Node(Data(1, SVector(0, 0, 1.0)), 4, 3, 1, 0)
        Node(Data(1, SVector(0, 0, -1.0)), 4, 4, 1, 0)
        Node(Data(2, SVector(0, 1.0, 0)), 4, 5, 1, 0)
        Node(Data(2, SVector(0, -1.0, 0)), 4, 6, 1, 0)
        Node(Data(3, SVector(1.0, 0, 0)), 4, 7, 1, 0)
        Node(Data(3, SVector(-1.0, 0, 0)), 4, 0, 1, 0)
    ]
    if k * √3hs > 1#diameter > 2ηₕ / (k * 2 * √3hs)
        for node in 2:7
            split!(nodes, node, hs / 2, ηₕ, k)
        end
    end

    tree = PointerBasedTrees.PointerBasedTree(nodes, 1)
    return tree
end

function cluster_link(tree::PointerBasedTrees.PointerBasedTree{D}) where {D}
    return cluster_link
end

function newstate(
    tree::PointerBasedTrees.PointerBasedTree{D}, node::Int, ct::SVector{3,F}
) where {D,F}
    return acos(
        min(
            dot(tree.nodes[node].data.ℯ, ct) / (norm(tree.nodes[node].data.ℯ) * norm(ct)),
            1.0,
        ),
    ),
    node
end

function find_direction(
    tree::PointerBasedTrees.PointerBasedTree{D},
    ct::SVector{3,F},
    destination::Int;
    state=newstate(tree, 2, ct),
    node=3,
    depth=1,
) where {D,F}
    depth > destination && return ClusterTrees.parent(tree, node)

    γ_min, statenode = state
    γ = acos(
        min(
            dot(tree.nodes[node].data.ℯ, ct) / (norm(tree.nodes[node].data.ℯ) * norm(ct)),
            1.0,
        ),
    )
    γ < γ_min && (γ_min = γ; statenode = node)

    if PointerBasedTrees.nextsibling(tree, node) == 0
        PointerBasedTrees.firstchild(tree, statenode) == 0 && (return statenode)
        state = newstate(tree, tree.nodes[statenode].first_child, ct)
        return find_direction(
            tree,
            ct,
            destination;
            state=state,
            node=PointerBasedTrees.nextsibling(tree, state[2]),
            depth=depth + 1,
        )
    else
        return find_direction(
            tree,
            ct,
            destination;
            state=(γ_min, statenode),
            node=PointerBasedTrees.nextsibling(tree, node),
            depth=depth,
        )
    end
end

function direction(directions::Dict{Int,Vector{Int}}, node::Int)
    for (key, nodes) in directions
        node in nodes && return key
    end
end
