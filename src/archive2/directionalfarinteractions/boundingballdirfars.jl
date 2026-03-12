using H2Trees
import H2Trees: testtree, trialtree, levels, LevelIterator, numberofnodes, center
import H2Trees: radius, parent, firstchild, BoundingBallTree, halfsize, level

struct BoundingBallDirectionalData{T<:Vector{Vector{Int}}} <: DirectionalData
    F::T
    E::T
    Emap::T
end

function directions(data::BoundingBallDirectionalData, node::Int)
    return union(data.E[node], data.Emap[node])
end

function genradius(tree::BoundingBallTree, t::Int)
    pt = parent(tree, t)

    pt == 0 && return H2Trees.radius(tree, t)
    radius = mapreduce(+, H2Trees.children(tree, pt)) do c
        H2Trees.radius(tree, c)
    end
    return min(genradius(tree, pt), radius / length(collect(H2Trees.children(tree, pt))))
end

function paternaldirection(data::BoundingBallDirectionalData, cnode::Int, dir::Int)
    dir == 0 && return 0
    return data.Emap[cnode][dir]
end

function inheritedtrialpivots(
    dirdata::BoundingBallDirectionalData,
    pivots::Vector{T},
    tree::BoundingBallTree,
    t::Int,
    e::Int;
    islf=islf(1.0),
) where {T}
    (
        isdirectionalroot(dirdata, tree, parent(tree, t)) ||
        !isassigned(pivots, parent(tree, t))
    ) && return Int[]
    islf(2genradius(tree, t)) && return pivots[parent(tree, t)][0][2]
    e in dirdata.Emap[t] && return Vector{Int}(
        mapreduce(vcat, findall(x -> x == e, dirdata.Emap[t])) do dir
            pivots[parent(tree, t)][dir][2]
        end,
    )
    return Int[]
end

function testfarfield(
    data::BoundingBallDirectionalData, tree, t::Int, e::Int; islf=islf(1.0)
)
    if islf(2genradius(tree, t))
        Ft = copy(data.F[t])
        for parent in ParentUpwardsIterator(tree, t)
            !islf(2genradius(tree, parent)) && return Ft
            append!(Ft, data.F[parent])
        end
        return Ft
    else
        Ft = data.F[t][findall(x -> x == e, data.E[t])]
        Et = findall(x -> x == e, data.Emap[t])
        for parent in ParentUpwardsIterator(tree, t)
            data.F[parent] == Int[] && (Et = findall(x -> x in Et, data.Emap[parent]);
            continue)
            append!(Ft, data.F[parent][findall(x -> x in Et, data.E[parent])])
            Et = findall(x -> x in Et, data.Emap[parent])
        end
        return Ft
    end
end

function inheritedtestpivots(
    dirdata::BoundingBallDirectionalData,
    pivots::Vector{T},
    tree::BoundingBallTree,
    s::Int,
    e::Int;
    islf=islf(1.0),
) where {T}
    (
        isdirectionalroot(dirdata, tree, parent(tree, s)) ||
        !isassigned(pivots, parent(tree, s))
    ) && return Int[]
    islf(2genradius(tree, s)) && return pivots[parent(tree, s)][0][1]
    e in dirdata.Emap[s] && return Vector{Int}(
        mapreduce(vcat, findall(x -> x == e, dirdata.Emap[s])) do dir
            pivots[parent(tree, s)][dir][1]
        end,
    )
    return []
end

function trialfarfield(
    data::BoundingBallDirectionalData, tree, s::Int, e::Int; islf=islf(1.0)
)
    if islf(2genradius(tree, s))
        Fs = data.F[s]
        for parent in ParentUpwardsIterator(tree, s)
            !islf(2genradius(tree, parent)) && return Fs
            append!(Fs, data.F[parent])
        end
        return Fs
    else
        Fs = data.F[s][findall(x -> x == e, data.E[s])]
        Es = findall(x -> x == e, data.Emap[s])
        for parent in ParentUpwardsIterator(tree, s)
            data.F[parent] == Int[] && (Es = findall(x -> x in Es, data.Emap[parent]);
            continue)
            append!(Fs, data.F[parent][findall(x -> x in Es, data.E[parent])])
            Es = findall(x -> x in Es, data.Emap[parent])
        end
        return Fs
    end
end

function directions(::Val{2}, diamX::F, k::F) where {F}
    N = pi / asin(1 / (k * diamX))
    return [SVector(cos(2π * k / N), sin(2π * k / N)) for k in 0:(N - 1)]
end

function directions(::Val{3}, diamX::F, k::F) where {F}
    N = ceil(Int, 6 * 4^(log(2, acos(1 / sqrt(3)) / asin(min(1, 1 / (k * diamX))))))
    return sphericalfibonaccipoints(N)
end

# fibonaccti_sphere generates N points on the unit sphere using the Fibonacci lattice method.
function sphericalfibonaccipoints(N::Int)
    ga = pi * (3 - sqrt(5))     # golden angle
    pts = Vector{SVector{3,Float64}}(undef, N)
    for i in 0:(N - 1)
        z = 1 - 2 * (i + 0.5) / N
        r = sqrt(1 - z * z)
        pts[i + 1] = SVector(r * cos(i * ga), r * sin(i * ga), z)
    end

    return pts
end

function hasinteractions(
    F::T, E::T, Emap::T, tree::BoundingBallTree, t::Int
) where {T<:Vector{Vector{Int}}}
    F[t] != [] && return true
    parent(tree, t) == 0 && return false
    return (E[parent(tree, t)] != [] || Emap[parent(tree, t)] != [])
end

function directionaltestfars(
    tree::BlockTree{T}; isnear=isnear(1.0), ntasks=Threads.nthreads()
) where {T<:BoundingBallTree}
    F = farinteractions(testtree(tree), trialtree(tree); isnear=isnear)
    E = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    Emap = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    Evec = Vector{Vector{SVector{3,Float64}}}(undef, numberofnodes(testtree(tree)))

    for level in levels(testtree(tree))
        for t in collect(H2Trees.LevelIterator(testtree(tree), level))
            if !isnear.islf(testtree(tree), t) &&
                hasinteractions(F, E, Emap, testtree(tree), t)
                totalEvec = directions(
                    Val(size(eltype(tree), 1)), 2 * radius(testtree(tree), t), isnear.k
                )

                E[t] = Vector{Int}(
                    map(F[t]) do node
                        r = center(trialtree(tree), node) - center(testtree(tree), t)
                        findmin(x -> NestedCrossApproximation.angle(x, r), totalEvec)[2]
                    end,
                )

                if parent(testtree(tree), t) != 0 &&
                    isassigned(Evec, parent(testtree(tree), t))
                    Emap[t] = Vector{Int}(
                        map(Evec[parent(testtree(tree), t)]) do e
                            findmin(x -> NestedCrossApproximation.angle(x, e), totalEvec)[2]
                        end,
                    )
                else
                    Emap[t] = Int[]
                end
                #createlocaldirs
                Evec[t] = totalEvec[union(E[t], Emap[t])]
                uniqueE = union(E[t], Emap[t])
                for (idx, e) in enumerate(E[t])
                    E[t][idx] = findfirst(x -> x == e, uniqueE)
                end
                for (idx, e) in enumerate(Emap[t])
                    Emap[t][idx] = findfirst(x -> x == e, uniqueE)
                end

            else
                Emap[t] = Int[]
                isnear.islf(testtree(tree), t) ? (E[t] = Int[0]) : (E[t] = Int[])
            end
        end
    end

    return BoundingBallDirectionalData(F, E, Emap)
end

function directionaltrialfars(
    tree::BlockTree{T}; isnear=isnear(1.0), ntasks=Threads.nthreads()
) where {T<:BoundingBallTree}
    F = farinteractions(trialtree(tree), testtree(tree); isnear=isnear)
    E = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    Emap = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    Evec = Vector{Vector{SVector{3,Float64}}}(undef, numberofnodes(trialtree(tree)))

    for level in levels(trialtree(tree))
        for s in collect(H2Trees.LevelIterator(trialtree(tree), level))
            if !isnear.islf(trialtree(tree), s) &&
                hasinteractions(F, E, Emap, trialtree(tree), s)
                totalEvec = directions(
                    Val(size(eltype(tree), 1)), 2 * radius(trialtree(tree), s), isnear.k
                )
                E[s] = Vector{Int}(
                    map(F[s]) do node
                        r = center(testtree(tree), node) - center(trialtree(tree), s)
                        findmin(x -> NestedCrossApproximation.angle(x, r), totalEvec)[2]
                    end,
                )

                if parent(trialtree(tree), s) != 0 &&
                    isassigned(Evec, parent(trialtree(tree), s))
                    Emap[s] = Vector{Int}(
                        map(Evec[parent(trialtree(tree), s)]) do e
                            findmin(x -> NestedCrossApproximation.angle(x, e), totalEvec)[2]
                        end,
                    )
                else
                    Emap[s] = Int[]
                end
                #createlocaldirs
                Evec[s] = totalEvec[union(E[s], Emap[s])]
                uniqueE = union(E[s], Emap[s])
                for (idx, e) in enumerate(E[s])
                    E[s][idx] = findfirst(x -> x == e, uniqueE)
                end
                for (idx, e) in enumerate(Emap[s])
                    Emap[s][idx] = findfirst(x -> x == e, uniqueE)
                end

            else
                Emap[s] = Int[]
                isnear.islf(trialtree(tree), s) ? (E[s] = Int[0]) : (E[s] = Int[])
            end
        end
    end

    return BoundingBallDirectionalData(F, E, Emap)
end
