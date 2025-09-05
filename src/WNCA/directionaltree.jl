function directionalitneractions(tree, islf, isnear)
    lk = Threads.SpinLock()
    values, nearvalues = H2Trees.nearinteractions(
        tree; isnear=isnear, extractselfvalues=false
    )
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)
    fars = Vector{Tuple{Int,Int}}[]
    lfvalues = Vector{Int}[]
    lffarvalues = Vector{Vector{{Int}}}[]
    dirs = Vector{Int}[]

    for level in H2Trees.levels(H2Trees.testtree(tree))
        levelfars = Tuple{Int,Int}[]
        leveldirs = Int[]
        testclusters = collect(H2Trees.LevelIterator(H2Trees.testtree(tree), level))
        @tasks for t in testclusters
            lfclusterfars = Vector{Int}[]
            for s in iterator(H2Trees.testtree(tree), H2Trees.trialtree(tree), t)
                dir = 0
                islf(tree, level) && dir = direction(
                    H2Trees.center(tree.trialcluster, s) -
                    H2Trees.center(tree.testcluster, t),
                    dtree,
                    maxlevel,
                )
                if dir == 0
                    push!(lfclusterfars, H2Trees.values(tree.trialcluster, s))
                else
                    lock(lk) do
                        push!(levelfars, (t, s))
                        push!(leveldirs, dir)
                    end
                end
            end
            lock(lk) do
                push!(lfvalues, H2Trees.values(tree.trialcluster, t))
                push!(lffarvalues, levelnodelffarvalues)
            end
        end
        push!(fars, levelfars)
        push!(dirs, leveldirs)
    end
    return values, nearvalues, fars, dirs, lfvalues, lffarvalues
end

using StaticArrays
using LinearAlgebra
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
    maxlevel = floor(log2(k * hs * 2))
    println(maxlevel)
    maxlevel == 0 && error()
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

function direction(interaction, tree, maxdepth; depth=0, node=0)
    depth == maxdepth && return node
    node = argmin([
        angle(interaction, tree.nodes[child].ℯ) for child in children(tree, node)
    ])
    return direction(interaction, tree, maxdepth; depth=depth + 1, node=node)
end
##
lambda = 1
k = 2 * pi / lambda
hs = 1.0
tester = SVector(-1.0, 0.0, 0.0)
@time tree = 𝒟tree(hs, k)

direction(tester, tree, 1)

tree.nodes[6]
##
for node in eachindex(tree.nodes)
    if tree.nodes[node].level == 3
        println(tree.nodes[node].parent, ": ", node)
    end
end
##
##
tree.children[1].children[1].children

##
lambda = 1
k = 2 * pi / lambda
hs = 1

l = floor(log2(k * hs))

hs / 8 * k

##
a = [1, 3, 4, 2, 2, 3, 2, 1, 5, 1]
b = Vector(1:10)

function dirsort(a, b)
    return [b[findall(x -> x == dir, a)] for dir in unique(a)]
end

@time dirsort(a, b)
@time findall(x -> x == 1, a)
##
comb = [(rand(1:100), rand(1:10000)) for i in 1:10000]

function getvalues(comb)
    return [rand(1:100, 100) for i in findall(x -> x[1] == 10, comb)]
end

function getvalues2(comb)
    vals = Int[]
    for i in comb
        i[1] == 10 && append!(vals, rand(1:100, 100))
    end
    return vals
end
##
@time getvalues(comb);
@time getvalues2(comb);
