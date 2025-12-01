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
mutable struct 𝓔Node{F}
    level::Int
    face::Int
    e::SVector{3,F}
    parent::Int
    children::Vector{Int}
end

mutable struct 𝓔Tree{F}
    level::Int
    nodes::Vector{𝓔Node{F}}
end

function (tree::𝓔Tree{F})(hs::F, maxlevel::Int) where {F}
    tree.level != 0 && warning("Rerouting an existing 𝓔Tree will overwrite previous data.")
    tree.nodes = [
        𝓔Node(1, 1, SVector(0, 0, 1.0), 0, Int[]),
        𝓔Node(1, 1, SVector(0, 0, -1.0), 0, Int[]),
        𝓔Node(1, 2, SVector(0, 1.0, 0), 0, Int[]),
        𝓔Node(1, 2, SVector(0, -1.0, 0), 0, Int[]),
        𝓔Node(1, 3, SVector(1.0, 0, 0), 0, Int[]),
        𝓔Node(1, 3, SVector(-1.0, 0, 0), 0, Int[]),
    ]
    tree.level = maxlevel
    for node in 1:6
        root!(tree, node, 1, maxlevel, hs / 2)
    end
end

function root!(tree::𝓔Tree{F}, node, level, maxlevel, hs) where {F}
    level == maxlevel && return nothing
    append!(
        tree.nodes[node].children, Vector((length(tree.nodes) + 1):(length(tree.nodes) + 4))
    )
    for child in 1:4
        push!(
            tree.nodes,
            𝓔Node(
                tree.nodes[node].level + 1,
                tree.nodes[node].face,
                tree.nodes[node].e + childfaces[tree.nodes[node].face][child] .* hs,
                node,
                Int[],
            ),
        )
    end
    for node in tree.nodes[node].children
        root!(tree, node, level + 1, maxlevel, hs / 2)
    end
end

function children(tree::𝓔Tree{F}, node::Int) where {F}
    iszero(node) && return Vector(1:6)
    return tree.nodes[node].children
end

function parent(tree::𝓔Tree{F}, node::Int) where {F}
    iszero(node) && return 0

    return tree.nodes[node].parent
end

function direction(interaction, tree, maxdepth; depth=0, dir=0)
    depth == maxdepth && return dir
    newdir = argmin([
        angle(interaction, tree.nodes[child].e) for child in children(tree, dir)
    ])
    return direction(
        interaction, tree, maxdepth; depth=depth + 1, dir=children(tree, dir)[newdir]
    )
end

function isleaf(tree::𝓔Tree{F}, node::Int) where {F}
    tree.nodes[node].parent == 0 ? (return true) : (return false)
end

function isroot(tree::𝓔Tree{F}, level::Int) where {F}
    tree.level == level ? (return true) : (return false)
end

function maxlevel(tree::TwoNTree, islf::IsLowFrequencyFunctor{F}) where {F}
    level = 0
    while !islf(tree, level)
        level += 1
    end
    return level
end

struct TwoNDirectionalData{T<:Vector{Vector{Int}}} <: DirectionalData
    F::T
    𝓣ₑ::𝓔Tree
    𝓔::T
    inherited𝓔::T
end

function inheritedtrialpivots(
    data::TwoNDirectionalData, pivots::Vector{T}, tree, t::Int, e::Int; islf=islf(1.0)
) where {T}
    (islf(tree, level(tree, t)) && !islf(tree, level(tree, parent(tree, t)))) &&
        return Int[]
    islf(tree, level(tree, parent(tree, t))) && return pivots[parent(tree, t)][0][2]
    return map(x -> x[t][2], pivots[parent(tree, t)][children(data.𝓣ₑ, e)])
end

function testfarfield(data::TwoNDirectionalData, tree, t::Int, e::Int; islf=islf(1.0))
    if islf(tree, level(tree, t))
        Ft = data.F[t]
        for parent in ParentUpwardsIterator(tree, t)
            !islf(tree, level(tree, parent)) && return Ft
            append!(Ft, data.F[parent])
        end
        return Ft
    else
        Ft = data.F[t][findall(x -> x == e, data.𝓔[t])]
        𝓔t = children(data.𝓣ₑ, e)
        for parent in ParentUpwardsIterator(tree, t)
            data.F[parent] == Int[] && continue
            append!(Ft, data.F[parent][findall(x -> x in 𝓔t, data.𝓔[parent])])
            𝓔t = map(e -> children(data.𝓣ₑ, e), 𝓔t)
        end
        return Ft
    end
end

function inheritedtestpivots(
    data::TwoNDirectionalData, pivots::Vector{T}, tree, s::Int, e::Int; islf=islf(1.0)
) where {T}
    (islf(tree, level(tree, s)) && !islf(tree, level(tree, parent(tree, s)))) &&
        return Int[]
    islf(tree, level(tree, parent(tree, s))) && return pivots[parent(tree, s)][0][1]
    return map(x -> x[s][1], pivots[parent(tree, s)][children(data.𝓣ₑ, e)])
end

function trialfarfield(data::TwoNDirectionalData, tree, s::Int, e::Int; islf=islf(1.0))
    if islf(tree, level(tree, s))
        Fs = data.F[s]
        for parent in ParentUpwardsIterator(tree, s)
            !islf(tree, level(tree, parent)) && return Fs
            append!(Fs, data.F[parent])
        end
        return Fs
    else
        Fs = data.F[s][findall(x -> x == e, data.𝓔[s])]
        𝓔s = children(data.𝓣ₑ, e)
        for parent in ParentUpwardsIterator(tree, s)
            data.F[parent] == Int[] && continue
            append!(Fs, data.F[parent][findall(x -> x in 𝓔s, data.𝓔[parent])])
            𝓔s = map(e -> children(data.𝓣ₑ, e), 𝓔s)
        end
        return Fs
    end
end

function directionaltestfars(
    tree::BlockTree{T}; islf=islf(k), isnear=isnear(k), ntasks=Threads.nthreads()
) where {N,D,K,T<:TwoNTree{N,D,K}}
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)
    F = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    𝓔 = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    inherited𝓔 = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    𝓣ₑ = 𝓔Tree(0, 𝓔Node{K}[])
    dirmaxlevel = maxlevel(testtree(tree), islf)

    for level in levels(testtree(tree))
        @tasks for t in collect(H2Trees.LevelIterator(testtree(tree), level))
            @set ntasks = ntasks
            F[t] = collect(iterator(trialtree(tree), testtree(tree), t))
            if F[t] != Int[] && !islf(testtree(tree), level)
                𝓣ₑ.level == 0 && 𝓣ₑ(halfsize(testtree(tree), t), dirmaxlevel - level + 1)
                𝓔[t] = Vector{Int}(
                    map(F[t]) do s
                        r = center(trialtree(tree), s) - center(testtree(tree), t)
                        direction(r, 𝓣ₑ, dirmaxlevel - level + 1)
                    end,
                )

                pt = parent(testtree(tree), t)
                if pt != 0 && F[pt] != Int[]
                    inherited𝓔[t] = unique(
                        Vector{Int}(map(e -> parent(𝓣ₑ, e), unique(inherited𝓔[pt], 𝓔[pt])))
                    )
                else
                    inherited𝓔[t] = Int[]
                end
            else
                𝓔[t] = Int[]
            end
        end
    end

    return TwoNDirectionalData(F, 𝓣ₑ, 𝓔, inherited𝓔)
end

function directionaltrialfars(
    tree::BlockTree{T}; islf=islf(k), isnear=isnear(k)
) where {N,D,K,T<:TwoNTree{N,D,K}}
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)
    F = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    𝓔 = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    inherited𝓔 = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    𝓣ₑ = 𝓔Tree(0, 𝓔Node{K}[])
    dirmaxlevel = maxlevel(trialtree(tree), islf)

    for level in levels(trialtree(tree))
        @tasks for s in collect(H2Trees.LevelIterator(trialtree(tree), level))
            F[s] = collect(iterator(trialtree(tree), testtree(tree), s))
            if F[s] != Int[] && !islf(trialtree(tree), level)
                𝓣ₑ.level == 0 && 𝓣ₑ(halfsize(testtree(tree), s), dirmaxlevel - level + 1)
                𝓔[s] = Vector{Int}(
                    map(F[s]) do t
                        r = center(testtree(tree), t) - center(trialtree(tree), s)
                        direction(r, 𝓣ₑ, dirmaxlevel - level + 1)
                    end,
                )

                ps = parent(trialtree(tree), s)
                if ps != 0 && 𝓕[ps] != Int[]
                    inherited𝓔[s] = unique(
                        Vector{Int}(map(e -> parent(𝓣ₑ, e), unique(inherited𝓔[ps], 𝓔[ps])))
                    )
                else
                    inherited𝓔[s] = Int[]
                end
            else
                𝓔[s] = Int[]
            end
        end
    end

    return TwoNDirectionalData(F, 𝓣ₑ, 𝓔, inherited𝓔)
end
