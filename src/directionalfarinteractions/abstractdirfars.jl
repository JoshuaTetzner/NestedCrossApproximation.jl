abstract type DirectionalData end

#=
struct IsLowFrequencyFunctor{F}
    k::F
end

islf(k) = IsLowFrequencyFunctor(k)
(islf::IsLowFrequencyFunctor)(diam) = (diam * islf.k <= 1)
wavenumber(islf::IsLowFrequencyFunctor) = islf.k

function (islf::IsLowFrequencyFunctor)(tree::TwoNTree, level::Int)
    return 2 * sqrt(3) * H2Trees.halfsize(tree) * islf.k / (2.0^(level - 1)) <= 1
end

function (islf::IsLowFrequencyFunctor)(tree::BoundingBallTree, node::Int)
    parent(tree, node) == 0 && return islf(2radius(tree, node))
    return islf(2genradius(tree, node))
end=#
#=
struct IsNearFrequencyFunctor{F}
    k::F
    ηₗ::F
    ηₕ::F
    islf::IsLowFrequencyFunctor{F}
end

function (isnear::IsNearFrequencyFunctor)(
    treea::TwoNTree, treeb::TwoNTree, nodea::Int, nodeb::Int
)
    ths = halfsize(treea, nodea) * sqrt(3)
    shs = halfsize(treeb, nodeb) * sqrt(3)
    dist = norm(center(treea, nodea) - center(treeb, nodeb)) - (ths + shs)
    if isnear.islf(min(ths, shs))
        return 2 * max(ths, shs) > isnear.ηₗ * max(dist, 0.0)
    else
        return 4 * isnear.k * max(ths^2, shs^2) > isnear.ηₕ * max(dist, 0.0)
    end
end

function (isnear::IsNearFrequencyFunctor)(
    treea::H2Trees.BoundingBallTree, treeb::H2Trees.BoundingBallTree, nodea::Int, nodeb::Int
)
    ths = radius(treea, nodea)
    shs = radius(treeb, nodeb)
    dist = norm(center(treea, nodea) - center(treeb, nodeb)) - (ths + shs)
    if isnear.islf(min(2genradius(treea, nodea), 2genradius(treeb, nodeb)))
        return 2 * max(ths, shs) > isnear.ηₗ * max(dist, 0.0)
    else
        return 4 * isnear.k * max(ths^2, shs^2) > isnear.ηₕ * max(dist, 0.0)
    end
end
=#
function angle(a::SVector{D,F}, b::SVector{D,F}) where {D,F}
    return acos(min(dot(a, b) / (norm(a) * norm(b)), 1.0))
end

function admissiblelevel(tree, data::DirectionalData)
    lflevel = 0
    hflevel = 0
    for level in levels(tree)
        lflevelinteractions = 0
        hflevelinteractions = 0
        for node in H2Trees.LevelIterator(tree, level)
            if data.F[node] != Int[] && data.E[node] != Int[]
                hflevelinteractions += 1
            elseif data.F[node] != Int[]
                lflevelinteractions += 1
            end
            (hflevelinteractions != 0 && lflevelinteractions != 0) && break
        end
        hflevelinteractions != 0 && (hflevel += 1)
        lflevelinteractions != 0 && (lflevel += 1)
    end

    return max(lflevel, hflevel)
end

function isdirectionalleaf(dirdata::DirectionalData, tree::H2Trees.H2ClusterTree, node::Int)
    parent(tree, node) == 0 && return true
    isdirectionalroot(dirdata, tree, parent(tree, node)) && return true
    return false
end

function isdirectionalroot(dirdata::DirectionalData, tree::H2Trees.TwoNTree, node::Int)
    firstchild(tree, node) == 0 && return true
    (dirdata.E[node] != [0] && dirdata.E[firstchild(tree, node)] == [0]) && return true
    return false
end

function isdirectionalroot(
    dirdata::DirectionalData, tree::H2Trees.BoundingBallTree, node::Int
)
    firstchild(tree, node) == 0 && return true
    (dirdata.E[node] != [0] && dirdata.Emap[firstchild(tree, node)] == []) && return true
    return false
end

#function farinteractions(tree::BlockTree, isnear::IsNearFunctor)
#    testdirfars = directionaltestfars(testtree(tree), trialtree(tree); isnear=isnear)
#    testtree(tree) == trialtree(tree) && (return testdirfars, testdirfars)
#    return testdirfars, directionaltrialfars(trialtree(tree), testtree(tree); isnear=isnear)
#end
