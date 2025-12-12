using H2Trees
import H2Trees: testtree, trialtree, levels, LevelIterator, numberofnodes, center
import H2Trees: radius, parent, firstchild, BoundingBallTree, halfsize, level

struct BoundingBallDirectionalData{T<:Vector{Vector{Int}}} <: DirectionalData
    F::T
    𝓔::T
    𝓔map::T
end

function directions(data::BoundingBallDirectionalData, node::Int)
    return union(data.𝓔[node], data.𝓔map[node])
end

function paternaldirection(data::BoundingBallDirectionalData, cnode::Int, dir::Int)
    dir == 0 && return 0
    return data.𝓔map[cnode][dir]
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
    islf(radius(tree, parent(tree, t))) && return pivots[parent(tree, t)][0][2]
    e in dirdata.𝓔map[t] && return Vector{Int}(
        mapreduce(vcat, findall(x -> x == e, dirdata.𝓔map[t])) do dir
            pivots[parent(tree, t)][dir][2]
        end,
    )
    return Int[]
end

function testfarfield(
    data::BoundingBallDirectionalData, tree, t::Int, e::Int; islf=islf(1.0)
)
    if islf(radius(tree, parent(tree, t)))
        Ft = data.F[t]
        for parent in ParentUpwardsIterator(tree, t)
            !islf(radius(tree, parent(tree, parent))) && return Ft
            append!(Ft, data.F[parent])
        end
        return Ft
    else
        Ft = data.F[t][findall(x -> x == e, data.𝓔[t])]
        𝓔t = findall(x -> x == e, data.𝓔map[t])
        for parent in ParentUpwardsIterator(tree, t)
            data.F[parent] == Int[] && continue
            append!(Ft, data.F[parent][findall(x -> x in 𝓔t, data.𝓔[parent])])
            𝓔t = findall(
                x -> x in data.𝓔[parent][findall(x -> x in 𝓔t, data.𝓔[parent])],
                data.𝓔map[parent],
            )
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
    islf(radius(tree, parent(tree, s))) && return pivots[parent(tree, s)][0][1]
    e in dirdata.𝓔map[s] && return Vector{Int}(
        mapreduce(vcat, findall(x -> x == e, dirdata.𝓔map[s])) do dir
            pivots[parent(tree, s)][dir][1]
        end,
    )
    return []
end

function trialfarfield(
    data::BoundingBallDirectionalData, tree, s::Int, e::Int; islf=islf(1.0)
)
    if islf(radius(tree, parent(tree, s)))
        Fs = data.F[s]
        for parent in ParentUpwardsIterator(tree, s)
            !islf(radius(tree, parent(tree, parent))) && return Fs
            append!(Fs, data.F[parent])
        end
        return Fs
    else
        Fs = data.F[s][findall(x -> x == e, data.𝓔[s])]
        𝓔s = findall(x -> x == e, data.𝓔map[s])
        for parent in ParentUpwardsIterator(tree, s)
            data.F[parent] == Int[] && continue
            append!(Fs, data.F[parent][findall(x -> x in 𝓔s, data.𝓔[parent])])
            𝓔s = findall(
                x -> x in data.𝓔[parent][findall(x -> x in 𝓔s, data.𝓔[parent])],
                data.𝓔map[parent],
            )
        end
        return Fs
    end
end

# fibonaccti_sphere generates N points on the unit sphere using the Fibonacci lattice method.
function fibonacci_sphere(diamX::F, k::F) where {F}
    #radial distance to number of nodes
    N = ceil(Int, 6 * 4^(log(2, acos(1 / sqrt(3)) / asin(min(1, 1 / (k * diamX))))))
    ga = pi * (3 - sqrt(5))     # golden angle
    pts = Vector{SVector{3,F}}(undef, N)
    for i in 0:(N - 1)
        z = 1 - 2 * (i + 0.5) / N
        r = sqrt(1 - z * z)
        pts[i + 1] = SVector(r * cos(i * ga), r * sin(i * ga), z)
    end

    return pts
end

function hasinteractions(
    F::T, 𝓔::T, 𝓔map::T, tree::BoundingBallTree, t::Int
) where {T<:Vector{Vector{Int}}}
    F[t] != [] && return true
    parent(tree, t) == 0 && return false
    return (𝓔[parent(tree, t)] != [] || 𝓔map[parent(tree, t)] != [])
end

function directionaltestfars(
    tree::BlockTree{T}; islf=islf(1.0), isnear=isnear(1.0), ntasks=Threads.nthreads()
) where {T<:BoundingBallTree}
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)
    F = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    𝓔 = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    𝓔map = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    𝓔vec = Vector{Vector{SVector{3,Float64}}}(undef, numberofnodes(testtree(tree)))

    for level in levels(testtree(tree))
        for t in collect(H2Trees.LevelIterator(testtree(tree), level))
            F[t] = collect(iterator(trialtree(tree), testtree(tree), t))

            if !islf(testtree(tree), t) && hasinteractions(F, 𝓔, 𝓔map, testtree(tree), t)
                total𝓔vec = fibonacci_sphere(2 * radius(testtree(tree), t), isnear.k)

                𝓔[t] = Vector{Int}(
                    map(F[t]) do node
                        r = center(trialtree(tree), node) - center(testtree(tree), t)
                        findmin(x -> angle(x, r), total𝓔vec)[2]
                    end,
                )

                if parent(testtree(tree), t) != 0 &&
                    isassigned(𝓔vec, parent(testtree(tree), t))
                    𝓔map[t] = Vector{Int}(
                        map(𝓔vec[parent(testtree(tree), t)]) do e
                            findmin(x -> angle(x, e), total𝓔vec)[2]
                        end,
                    )
                else
                    𝓔map[t] = Int[]
                end
                #createlocaldirs
                𝓔vec[t] = total𝓔vec[union(𝓔[t], 𝓔map[t])]
                unique𝓔 = union(𝓔[t], 𝓔map[t])
                for (idx, e) in enumerate(𝓔[t])
                    𝓔[t][idx] = findfirst(x -> x == e, unique𝓔)
                end
                for (idx, e) in enumerate(𝓔map[t])
                    𝓔map[t][idx] = findfirst(x -> x == e, unique𝓔)
                end

            else
                F[t] == [] ? (𝓔[t] = Int[]) : (𝓔[t] = Int[0])
                𝓔map[t] = Int[]
            end
        end
    end

    return BoundingBallDirectionalData(F, 𝓔, 𝓔map)
end

function directionaltrialfars(
    tree::BlockTree{T}; islf=islf(1.0), isnear=isnear(1.0), ntasks=Threads.nthreads()
) where {T<:BoundingBallTree}
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)
    F = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    𝓔 = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    𝓔map = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    𝓔vec = Vector{Vector{SVector{3,Float64}}}(undef, numberofnodes(trialtree(tree)))

    for level in levels(trialtree(tree))
        for s in collect(H2Trees.LevelIterator(trialtree(tree), level))
            F[s] = collect(iterator(testtree(tree), trialtree(tree), s))

            if !islf(trialtree(tree), s) && hasinteractions(F, 𝓔, 𝓔map, trialtree(tree), s)
                total𝓔vec = fibonacci_sphere(2 * radius(trialtree(tree), s), isnear.k)

                𝓔[s] = Vector{Int}(
                    map(F[s]) do node
                        r = center(testtree(tree), node) - center(trialtree(tree), s)
                        findmin(x -> angle(x, r), total𝓔vec)[2]
                    end,
                )

                if parent(trialtree(tree), s) != 0 &&
                    isassigned(𝓔vec, parent(trialtree(tree), s))
                    𝓔map[s] = Vector{Int}(
                        map(𝓔vec[parent(trialtree(tree), s)]) do e
                            findmin(x -> angle(x, e), total𝓔vec)[2]
                        end,
                    )
                else
                    𝓔map[s] = Int[]
                end
                #createlocaldirs
                𝓔vec[s] = total𝓔vec[union(𝓔[s], 𝓔map[s])]
                unique𝓔 = union(𝓔[s], 𝓔map[s])
                for (idx, e) in enumerate(𝓔[s])
                    𝓔[s][idx] = findfirst(x -> x == e, unique𝓔)
                end
                for (idx, e) in enumerate(𝓔map[s])
                    𝓔map[s][idx] = findfirst(x -> x == e, unique𝓔)
                end

            else
                F[s] == [] ? (𝓔[s] = Int[]) : (𝓔[s] = Int[0])
                𝓔map[s] = Int[]
            end
        end
    end

    return BoundingBallDirectionalData(F, 𝓔, 𝓔map)
end
