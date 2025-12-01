using H2Trees
import H2Trees: testtree, trialtree, levels, LevelIterator, numberofnodes, center
import H2Trees: radius, parent, firstchild, BoundingBallTree, halfsize, level

struct BoundingBallDirectionalData{T<:Vector{Vector{Int}}} <: DirectionalData
    F::T
    𝓔::T
    𝓔map::T
end

function admissiblelevel(tree, data::BoundingBallDirectionalData)
    lflevel = 0
    hflevel = 0
    for level in levels(tree)
        lflevelinteractions = 0
        hflevelinteractions = 0
        for node in H2Trees.LevelIterator(tree, level)
            if data.F[node] != Int[] && data.𝓔[node] != Int[]
                hflevelinteractions += 1
            elseif data.F[node] != Int[]
                lflevelinteractions += 1
            end
            (hflevelinteractions != 0 && lflevelinteractions != 0) && break
        end
        hflevelinteractions != 0 && hflevel += 1
        lflevelinteractions != 0 && lflevel += 1
    end

    return max(lflevel, hflevel)
end

function inheritedtrialpivots(
    data::BoundingBallDirectionalData,
    pivots::Vector{T},
    tree::BoundingBallTree,
    t::Int,
    e::Int;
    islf=islf(1.0),
) where {T}
    (islf(2radius(tree, t)) && !islf(2radius(tree, parent(tree, t)))) && return Int[]
    islf(2radius(tree, parent(tree, t))) && return pivots[parent(tree, t)][0][2]
    return map(
        x -> x[t][2],
        pivots[parent(tree, t)][unique(
            data.𝓔[parent(tree, t)][findall(x -> x == e, data.𝓔map[t])]
        )],
    )
end

function testfarfield(
    data::BoundingBallDirectionalData, tree, t::Int, e::Int; islf=islf(1.0)
)
    if islf(2radius(tree, t))
        Ft = data.F[t]
        for parent in ParentUpwardsIterator(tree, t)
            !islf(2radius(tree, parent)) && return Ft
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
    data::BoundingBallDirectionalData,
    pivots::Vector{T},
    tree::BoundingBallTree,
    s::Int,
    e::Int;
    islf=islf(1.0),
) where {T}
    (islf(2radius(tree, s)) && !islf(2radius(tree, parent(tree, s)))) && return Int[]
    islf(2radius(tree, parent(tree, s))) && return pivots[parent(tree, s)][0][1]
    return map(
        x -> x[s][1],
        pivots[parent(tree, s)][unique(
            data.𝓔[parent(tree, s)][findall(x -> x == e, data.𝓔map[s])]
        )],
    )
end

function trialfarfield(
    data::BoundingBallDirectionalData, tree, s::Int, e::Int; islf=islf(1.0)
)
    if islf(2radius(tree, s))
        Fs = data.F[s]
        for parent in ParentUpwardsIterator(tree, s)
            !islf(2radius(tree, parent)) && return Fs
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
    N = ceil(Int, 6 * 4^(log(2, acos(1 / sqrt(3)) / asin(1 / (k * diamX)))))
    ga = pi * (3 - sqrt(5))     # golden angle
    pts = Vector{SVector{3,F}}(undef, N)
    for i in 0:(N - 1)
        z = 1 - 2 * (i + 0.5) / N
        r = sqrt(1 - z * z)
        pts[i + 1] = SVector(r * cos(i * ga), r * sin(i * ga), z)
    end

    return pts
end

function directionaltestfars(
    tree::BlockTree{T}; islf=islf(1.0), isnear=isnear(1.0)
) where {T<:BoundingBallTree}
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)
    𝓕 = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    𝓔 = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    𝓔map = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    𝓔vec = Vector{Vector{SVector{3,Float64}}}(undef, numberofnodes(testtree(tree)))

    for level in levels(testtree(tree))
        for t in collect(H2Trees.LevelIterator(testtree(tree), level))
            𝓕[t] = collect(iterator(trialtree(tree), testtree(tree), t))
            if !islf(2 * radius(testtree(tree), t)) && (
                𝓕[t] != Int[] ||
                𝓔[parent(testtree(tree), t)] != Int[] ||
                𝓔map[parent(testtree(tree), t)] != Int[]
            )
                total𝓔vec = fibonacci_sphere(2 * radius(testtree(tree), t), isnear.k)
                𝓔[t] = Vector{Int}(
                    map(𝓕[t]) do node
                        r = center(trialtree(tree), node) - center(testtree(tree), t)
                        findmin(x -> angle(x, r), total𝓔vec)[2]
                    end,
                )

                if parent(testtree(tree), t) != 0 && 𝓕[parent(testtree(tree), t)] != Int[]
                    𝓔map[t] = Vector{Int}(
                        map(𝓔vec[parent(testtree(tree), t)]) do e
                            findmin(x -> angle(x, e), total𝓔vec)[2]
                        end,
                    )

                    #createlocaldirs
                    𝓔vec[t] = total𝓔vec[union(𝓔[t], 𝓔map[t])]
                    unique𝓔 = union(𝓔[t], 𝓔map[t])
                    for (idx, e) in enumerate(𝓔[t])
                        𝓔[t][idx] = findfirst(x -> x == e, unique𝓔)
                    end
                    for (idx, e) in enumerate(𝓔map[t])
                        𝓔map[t][idx] = findfirst(x -> x == e, unique𝓔)
                    end

                    𝓔unique[t] = union(𝓔[t])
                else
                    𝓔map[t] = Int[]
                    #createlocaldirs
                    𝓔vec[t] = total𝓔vec[unique(𝓔[t])]
                    unique𝓔 = unique(𝓔[t])
                    for (idx, e) in enumerate(𝓔[t])
                        𝓔[t][idx] = findfirst(x -> x == e, unique𝓔)
                    end
                end
            else
                𝓔[t] = Int[]
                𝓔map[t] = Int[]
            end
        end
    end

    return BoundingBallDirectionalData(𝓕, 𝓔, 𝓔map)
end

function directionaltrialfars(
    tree::BlockTree{T}; islf=islf(1.0), isnear=isnear(1.0)
) where {T<:BoundingBallTree}
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)
    𝓕 = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    𝓔 = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    𝓔map = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    𝓔vec = Vector{Vector{SVector{3,Float64}}}(undef, numberofnodes(trialtree(tree)))

    for level in levels(trialtree(tree))
        @tasks for s in collect(H2Trees.LevelIterator(trialtree(tree), level))
            𝓕[s] = collect(iterator(testtree(tree), trialtree(tree), s))
            if !islf(2 * radius(trialtree(tree), s)) && (
                𝓕[s] != Int[] ||
                𝓔[parent(trialtree(tree), s)] != Int[] ||
                𝓔map[parent(trialtree(tree), s)] != Int[]
            )
                total𝓔vec = fibonacci_sphere(2 * radius(trialtree(tree), s), isnear.k)
                𝓔[s] = Vector{Int}(
                    map(𝓕[s]) do node
                        r = center(testtree(tree), node) - center(trialtree(tree), s)
                        findmin(x -> angle(x, r), total𝓔vec)[2]
                    end,
                )

                if parent(trialtree(tree), s) != 0 && 𝓕[parent(trialtree(tree), s)] != Int[]
                    𝓔map[s] = Vector{Int}(
                        map(𝓔vec[parent(trialtree(tree), s)]) do e
                            findmin(x -> angle(x, e), total𝓔vec)[2]
                        end,
                    )

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
                    #createlocaldirs
                    𝓔map[s] = Int[]
                    𝓔vec[s] = total𝓔vec[unique(𝓔[s])]
                    unique𝓔 = unique(𝓔[s])
                    for (idx, e) in enumerate(𝓔[s])
                        𝓔[s][idx] = findfirst(x -> x == e, unique𝓔)
                    end
                end
            else
                𝓔[s] = Int[]
                𝓔map[s] = Int[]
            end
        end
    end

    return BoundingBallDirectionalData(𝓕, 𝓔, 𝓔map)
end
