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
mutable struct 𝒟node{F}
    level::Int
    face::Int
    ℯ::SVector{3,F}
    parent::Int
    children::Vector{Int}
end
mutable struct 𝒟tree{F}
    level::Int
    nodes::Vector{𝒟node{F}}
end

function 𝒟tree(hs::F, k::F) where {F}
    maxlevel = max(0, floor(log2(k * hs * 2)) + 1)
    maxlevel == 0 && return 𝒟tree(0, 𝒟node{F}[])
    nodes = [
        𝒟node(1, 1, SVector(0, 0, 1.0), 0, Int[]),
        𝒟node(1, 1, SVector(0, 0, -1.0), 0, Int[]),
        𝒟node(1, 2, SVector(0, 1.0, 0), 0, Int[]),
        𝒟node(1, 2, SVector(0, -1.0, 0), 0, Int[]),
        𝒟node(1, 3, SVector(1.0, 0, 0), 0, Int[]),
        𝒟node(1, 3, SVector(-1.0, 0, 0), 0, Int[]),
    ]
    tree = 𝒟tree{F}(maxlevel, nodes)
    for node in 1:6
        root!(tree, node, 1, maxlevel, hs / 2)
    end
    return tree
end

function root!(tree::𝒟tree{F}, node, level, maxlevel, hs) where {F}
    level == maxlevel && return nothing
    append!(
        tree.nodes[node].children, Vector((length(tree.nodes) + 1):(length(tree.nodes) + 4))
    )
    for child in 1:4
        push!(
            tree.nodes,
            𝒟node(
                tree.nodes[node].level + 1,
                tree.nodes[node].face,
                tree.nodes[node].ℯ + childfaces[tree.nodes[node].face][child] .* hs,
                node,
                Int[],
            ),
        )
    end
    for node in tree.nodes[node].children
        root!(tree, node, level + 1, maxlevel, hs / 2)
    end
end

function angle(a::SVector{3,F}, b::SVector{3,F}) where {F}
    return acos(min(dot(a, b) / (norm(a) * norm(b)), 1.0))
end

function children(tree::𝒟tree{F}, node::Int) where {F}
    iszero(node) && return Vector(1:6)
    return tree.nodes[node].children
end

function parent(tree::𝒟tree{F}, node::Int) where {F}
    return tree.nodes[node].parent
end

function direction(interaction, tree, maxdepth; depth=0, dir=0)
    depth == maxdepth && return dir
    newdir = argmin([
        angle(interaction, tree.nodes[child].ℯ) for child in children(tree, dir)
    ])
    return direction(
        interaction, tree, maxdepth; depth=depth + 1, dir=children(tree, dir)[newdir]
    )
end

function isleaf(tree::𝒟tree{F}, node::Int) where {F}
    tree.nodes[node].parent == 0 ? (return true) : (return false)
end
