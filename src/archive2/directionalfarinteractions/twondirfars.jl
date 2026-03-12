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
mutable struct ENode{F}
    level::Int
    face::Int
    e::SVector{3,F}
    parent::Int
    children::Vector{Int}
end

mutable struct ETree{F}
    level::Int
    nodes::Vector{ENode{F}}
end

function (tree::ETree{F})(hs::F, maxlevel::Int) where {F}
    tree.level != 0 && warning("Rerouting an existing ETree will overwrite previous data.")
    tree.nodes = [
        ENode(1, 1, SVector(0, 0, 1.0), 0, Int[]),
        ENode(1, 1, SVector(0, 0, -1.0), 0, Int[]),
        ENode(1, 2, SVector(0, 1.0, 0), 0, Int[]),
        ENode(1, 2, SVector(0, -1.0, 0), 0, Int[]),
        ENode(1, 3, SVector(1.0, 0, 0), 0, Int[]),
        ENode(1, 3, SVector(-1.0, 0, 0), 0, Int[]),
    ]
    tree.level = maxlevel
    for node in 1:6
        root!(tree, node, 1, maxlevel, hs / 2)
    end
end

function root!(tree::ETree{F}, node, level, maxlevel, hs) where {F}
    level == maxlevel && return nothing
    append!(
        tree.nodes[node].children, Vector((length(tree.nodes) + 1):(length(tree.nodes) + 4))
    )
    for child in 1:4
        push!(
            tree.nodes,
            ENode(
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

function children(tree::ETree{F}, node::Int) where {F}
    iszero(node) && return Vector(1:6)
    return tree.nodes[node].children
end

function parent(tree::ETree{F}, node::Int) where {F}
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

function isleaf(tree::ETree{F}, node::Int) where {F}
    tree.nodes[node].parent == 0 ? (return true) : (return false)
end

function isleaflevel(tree::ETree{F}, level::Int) where {F}
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
    𝓣ₑ::ETree
    E::T
    inheritedE::T
end

function directions(data::TwoNDirectionalData, node::Int)
    return union(data.E[node], data.inheritedE[node])
end

function paternaldirection(data::TwoNDirectionalData, _::Int, dir::Int)
    return parent(data.𝓣ₑ, dir)
end

function inheritedtrialpivots(
    data::TwoNDirectionalData, pivots::Vector{T}, tree, t::Int, e::Int; islf=islf(1.0)
) where {T}
    (islf(tree, level(tree, t)) && !islf(tree, level(tree, parent(tree, t)))) &&
        return Int[]
    !isassigned(pivots, parent(tree, t)) && return Int[]
    islf(tree, level(tree, parent(tree, t))) && return pivots[parent(tree, t)][0][2]
    return mapreduce(vcat, children(data.𝓣ₑ, e)) do eprime
        haskey(pivots[parent(tree, t)], eprime) && return pivots[parent(tree, t)][eprime][2]
        return Int[]
    end
end

function testfarfield(data::TwoNDirectionalData, tree, t::Int, e::Int; islf=islf(1.0))
    #println(t, ", in ther its ", islf(tree, level(tree, t)))
    if islf(tree, level(tree, t))
        Ft = copy(data.F[t])
        for parent in ParentUpwardsIterator(tree, t)
            !islf(tree, level(tree, parent)) && return Ft
            append!(Ft, data.F[parent])
        end
        return Ft
    else
        Ft = data.F[t][findall(x -> x == e, data.E[t])]
        Et = children(data.𝓣ₑ, e)
        for parent in ParentUpwardsIterator(tree, t)
            data.F[parent] == Int[] && continue
            append!(Ft, data.F[parent][findall(x -> x in Et, data.E[parent])])
            Et = reduce(vcat, map(x -> children(data.𝓣ₑ, x), Et))
        end
        return Ft
    end
end

function inheritedtestpivots(
    data::TwoNDirectionalData, pivots::Vector{T}, tree, s::Int, e::Int; islf=islf(1.0)
) where {T}
    (islf(tree, level(tree, s)) && !islf(tree, level(tree, parent(tree, s)))) &&
        return Int[]
    !isassigned(pivots, parent(tree, s)) && return Int[]
    islf(tree, level(tree, parent(tree, s))) && return pivots[parent(tree, s)][0][1]
    return mapreduce(vcat, children(data.𝓣ₑ, e)) do eprime
        haskey(pivots[parent(tree, s)], eprime) && return pivots[parent(tree, s)][eprime][1]
        return Int[]
    end
end

function trialfarfield(data::TwoNDirectionalData, tree, s::Int, e::Int; islf=islf(1.0))
    if islf(tree, level(tree, s))
        Fs = copy(data.F[s])
        for parent in ParentUpwardsIterator(tree, s)
            !islf(tree, level(tree, parent)) && return Fs
            append!(Fs, data.F[parent])
        end
        return Fs
    else
        Fs = data.F[s][findall(x -> x == e, data.E[s])]
        Es = children(data.𝓣ₑ, e)
        for parent in ParentUpwardsIterator(tree, s)
            data.F[parent] == Int[] && continue
            append!(Fs, data.F[parent][findall(x -> x in Es, data.E[parent])])
            Es = reduce(vcat, map(e -> children(data.𝓣ₑ, e), Es))
        end
        return Fs
    end
end

function directionaltestfars(
    tree::BlockTree{T}; isnear=isnear(1.0), ntasks=Threads.nthreads()
) where {N,D,K,T<:TwoNTree{N,D,K}}
    F = farinteractions(testtree(tree), trialtree(tree); isnear=isnear)
    E = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    inheritedE = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    𝓣ₑ = ETree(0, ENode{K}[])
    dirmaxlevel = maxlevel(testtree(tree), isnear.islf)

    for level in levels(testtree(tree))
        for t in collect(H2Trees.LevelIterator(testtree(tree), level))
            if F[t] != Int[] && !isnear.islf(testtree(tree), level)
                𝓣ₑ.level == 0 && 𝓣ₑ(halfsize(testtree(tree), t), dirmaxlevel - level + 1)
                E[t] = Vector{Int}(
                    map(F[t]) do s
                        r = center(trialtree(tree), s) - center(testtree(tree), t)
                        direction(r, 𝓣ₑ, dirmaxlevel - level + 1)
                    end,
                )

                pt = parent(testtree(tree), t)
                if pt != 0 && F[pt] != Int[]
                    inheritedE[t] = unique(
                        Vector{Int}(map(e -> parent(𝓣ₑ, e), union(inheritedE[pt], E[pt])))
                    )
                else
                    inheritedE[t] = Int[]
                end
            else
                isnear.islf(testtree(tree), level) ? (E[t] = Int[0]) : (E[t] = Int[])
                inheritedE[t] = Int[]
            end
        end
    end

    return TwoNDirectionalData(F, 𝓣ₑ, E, inheritedE)
end

function directionaltrialfars(
    tree::BlockTree{T}; isnear=isnear(1.0), ntasks=Threads.nthreads()
) where {N,D,K,T<:TwoNTree{N,D,K}}
    F = farinteractions(trialtree(tree), testtree(tree); isnear=isnear)
    E = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    inheritedE = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    𝓣ₑ = ETree(0, ENode{K}[])
    dirmaxlevel = maxlevel(trialtree(tree), isnear.islf)

    for level in levels(trialtree(tree))
        for s in collect(H2Trees.LevelIterator(trialtree(tree), level))
            if F[s] != Int[] && !isnear.islf(trialtree(tree), level)
                𝓣ₑ.level == 0 && 𝓣ₑ(halfsize(trialtree(tree), s), dirmaxlevel - level + 1)
                E[s] = Vector{Int}(
                    map(F[s]) do t
                        r = center(testtree(tree), t) - center(trialtree(tree), s)
                        direction(r, 𝓣ₑ, dirmaxlevel - level + 1)
                    end,
                )
                ps = parent(trialtree(tree), s)
                if ps != 0 && F[ps] != Int[]
                    inheritedE[s] = unique(
                        Vector{Int}(map(e -> parent(𝓣ₑ, e), union(inheritedE[ps], E[ps])))
                    )
                else
                    inheritedE[s] = Int[]
                end
            else
                isnear.islf(trialtree(tree), level) ? (E[s] = Int[0]) : (E[s] = Int[])
                inheritedE[s] = Int[]
            end
        end
    end

    return TwoNDirectionalData(F, 𝓣ₑ, E, inheritedE)
end
