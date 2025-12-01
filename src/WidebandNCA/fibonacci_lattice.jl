using LinearAlgebra, StaticArrays

function fibonacci_sphere(N::Int)
    ga = pi * (3 - sqrt(5))     # golden angle
    pts = Vector{SVector{3,Float64}}(undef, N)
    for i in 0:(N - 1)
        z = 1 - 2 * (i + 0.5) / N
        r = sqrt(1 - z * z)
        pts[i + 1] = SVector(r * cos(i * ga), r * sin(i * ga), z)
    end

    return pts
end

function mindist(points::Vector{SVector{3,Float64}})
    mindists = zeros(size(points, 1))
    for p in 1:size(points, 1)
        dist = [norm(points[p, :] - points[q, :]) for q in 1:size(points, 1) if q != p]
        mindists[p] = minimum(dist)
    end
    return mindists
end

##

##

d6 = sqrt(2 / 3)
kb_d6 = 2 * asin(d6 / 2)

N(p) = 6 * 4^p
p(N) = log(4, N / 6)

d(N) = sqrt(2 / 3) / 2^p(N)

Nnn(d::Float64) = 6 * 4^log(2, (sqrt(2 / 3) / d))

Nnn(5)
##
using CompScienceMeshes
using ParallelKMeans
using H2Trees
using NestedCrossApproximation

Γ = meshsphere(1.0, 0.01)
ttree = KMeansTree(Γ.vertices, 2; minvalues=100)
tree = H2Trees.BlockTree(ttree, ttree)
λ = 0.5
k = 2 * pi / λ

function angle(a::SVector{3,F}, b::SVector{3,F}) where {F}
    return acos(min(dot(a, b) / (norm(a) * norm(b)), 1.0))
end

isnear = NestedCrossApproximation.isnear(k)

function (isnear::NestedCrossApproximation.IsNearFunctor{F})(
    treea::H2Trees.BoundingBallTree, treeb::H2Trees.BoundingBallTree, nodea::Int, nodeb::Int
) where {F}
    ths = H2Trees.radius(treea, nodea)
    shs = H2Trees.radius(treeb, nodeb)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)
    if isnear.islf(min(ths, shs))
        (2 * max(ths, shs) <= isnear.ηₗ * max(dist, 0.0)) ? (return false) : (return true)
    else
        if (4 * isnear.k * max(ths^2, shs^2) <= isnear.ηₕ * max(dist, 0.0))
            (return false)
        else
            (return true)
        end
    end
end

##
function dira(ttree; isnear=H2Trees.isnear)
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)
    fars = Vector{Vector{Int}}(undef, length(ttree.nodes))
    dirs = Vector{Vector{Int}}(undef, length(ttree.nodes))
    pdirs = Vector{Vector{Int}}(undef, length(ttree.nodes))
    vdirs = Vector{Vector{SVector{3,Float64}}}(undef, length(ttree.nodes))
    for level in H2Trees.levels(ttree)
        @tasks for t in collect(H2Trees.LevelIterator(ttree, level))
            if k * H2Trees.radius(ttree, t) > 0.5
                N = ceil(Int, Nnn(2 * asin(1 / (k * H2Trees.radius(ttree, t)))))
                pts = fib_sphere2(N)
                Ft = collect(iterator(ttree, ttree, t))
                fars[t] = Ft
                tdirs = zeros(Int, length(Ft))
                for (is, s) in enumerate(Ft)
                    center_diff = H2Trees.center(ttree, s) - H2Trees.center(ttree, t)
                    dir = 0
                    minangle = Float64(π)
                    for (i, pt) in enumerate(pts)
                        ang = angle(pt, center_diff)
                        if ang < minangle
                            minangle = ang
                            dir = i
                        end
                    end
                    tdirs[is] = dir
                end
                dirs[t] = tdirs

                tp = H2Trees.parent(ttree, t)
                if tp != 0
                    tpdirs = zeros(Int, length(vdirs[tp]))
                    for (id, vdir) in enumerate(vdirs[tp])
                        dir = 0
                        minangle = Float64(π)
                        @inbounds for (i, pt) in enumerate(pts)
                            ang = angle(pt, vdir)
                            if ang < minangle
                                minangle = ang
                                dir = i
                            end
                        end
                        tpdirs[id] = dir
                    end
                    pdirs[t] = tpdirs
                else
                    pdirs[t] = Int[]
                end

                vdirs[t] = pts[unique(vcat(dirs[t], pdirs[t]))]
                localdirs = unique(vcat(dirs[t], pdirs[t]))
                for id in eachindex(dirs[t])
                    for ld in eachindex(localdirs)
                        if dirs[t][id] == localdirs[ld]
                            dirs[t][id] = ld
                            break
                        end
                    end
                end
                if tp != 0
                    for id in eachindex(pdirs[tp])
                        for ld in eachindex(localdirs)
                            if pdirs[tp][id] == localdirs[ld]
                                pdirs[tp][id] = ld
                                break
                            end
                        end
                    end
                end
            end
        end
    end

    return fars, dirs, pdirs, vdirs
end
##

function dirfars(directional_fars, dirs, pdirs, fars, t, dir, tree)
    fars[t] == Int[] && return nothing
    append!(
        directional_fars, fars[H2Trees.parent(tree, t)][findall(x -> x == dir, pdirs[t])]
    )
    for cdir in unique(dirs[H2Trees.parent(tree, t)][findall(x -> x == dir, pdirs[t])])
        dirfars(directional_fars, dirs, pdirs, fars, H2Trees.parent(tree, t), cdir, tree)
    end
end

function directionalfarfield(
    dirs::Vector{Vector{Int}},
    pdirs::Vector{Vector{Int}},
    fars::Vector{Vector{Int}},
    t::Int,
    dir::Int,
    tree,
)
    Fte = fars[t][findall(x -> x == dir, dirs[t])]
    dirfars(Fte, dirs, pdirs, fars, t, dir, tree)
    return Fte
end
##

##
directionalfarfield(dirs, pdirs, fars, 10, 1, ttree)

fars[8]
H2Trees.parent(ttree, 10)
collect(H2Trees.leaves(ttree))

##
leveledfars = Int[]
for level in H2Trees.levels(ttree)
    farsnodes = 0
    for t in collect(H2Trees.LevelIterator(ttree, level))
        if fars[t] != Int[]
            farsnodes += 1
        end
    end
    push!(leveledfars, farsnodes)
end
