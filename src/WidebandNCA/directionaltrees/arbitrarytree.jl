# fibonaccti_sphere generates N points on the unit sphere using the Fibonacci lattice method.
function fibonacci_sphere(d_rad::Int)
    #radial distance to number of nodes
    N = 6 * 4^log(2, (sqrt(2 / 3) / d_rad))

    ga = pi * (3 - sqrt(5))     # golden angle
    pts = Vector{SVector{3,Float64}}(undef, N)
    for i in 0:(N - 1)
        z = 1 - 2 * (i + 0.5) / N
        r = sqrt(1 - z * z)
        pts[i + 1] = SVector(r * cos(i * ga), r * sin(i * ga), z)
    end

    return pts
end

struct DirectionalFarInteractions{T<:Vector{Vector{Int}}} <: DirectionalData
    nodes::T
    𝓔::T
    𝓔map::T
end

function angle(a::SVector{3,F}, b::SVector{3,F}) where {F}
    return acos(min(dot(a, b) / (norm(a) * norm(b)), 1.0))
end

function directionaltestfars(
    tree::BlockTree{T}; islf=islf(k), isnear=isnear(k)
) where {T<:BoundingBallTree}
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)
    nodes = Vector{Vector{Int}}(undef, numberofvalues(testtree(tree)))
    𝓔 = Vector{Vector{Int}}(undef, numberofvalues(testtree(tree)))
    𝓔map = Vector{Vector{Int}}(undef, numberofvalues(testtree(tree)))
    𝓔vec = Vector{Vector{SVector{3,Float64}}}(undef, numberofvalues(testtree(tree)))

    for level in levels(testtree(tree))
        @tasks for t in collect(H2Trees.LevelIterator(ttree, level))
            nodes[t] = collect(iterator(trialtree(tree), testtree(tree), t))
            islf(2 * radius(testtree(tree), t)) && (𝓔[t] = Int[];
            𝓔map[t] = Int[];
            continue)

            total𝓔vec = fibonacci_sphere(
                ceil(Int, 2 * asin(1 / (k * radius(testtree(tree), t))))
            )
            𝓔[t] = Vector{Int}(
                map(nodes[t]) do node
                    r = center(trialtree(tree), node) - center(testtree(tree), t)
                    findemin(x -> angle(x, r), total𝓔vec)[2]
                end,
            )
            firstchild(testree(tree), t) != 0 && 𝓔vec[t] = total𝓔vec[unique(𝓔[t])]
            parent(testtree(tree), t) == 0 && continue
            nodes[parent(testtree(tree), t)] && continue
            𝓔map = Vector{Int}(
                map(𝓔vec[parent(testtree(tree), total𝓔vec)]) do e
                    findemin(x -> angle(x, e), total𝓔vec)
                end,
            )

            #createlocaldirs
            𝓔vec[t] = total𝓔vec[union()]
        end
    end
end

##
f(x, y) = x + y

pts = [3, 2, 3, 4, 5, 65, 67, 7]
e = [1, 2, 3, 4]
e = Int[]
x = Vector{Int}(
    map(e) do es
        a = 2
        findmin(n -> f(a, n), pts)[2]
    end,
)
