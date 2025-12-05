struct IsLowFrequencyFunctor{F}
    k::F
end

function islf(k::F) where {F}
    return IsLowFrequencyFunctor{F}(k)
end

wavenumber(islf::IsLowFrequencyFunctor{F}) where {F} = islf.k

function (islf::IsLowFrequencyFunctor{F})(tree::TwoNTree, level::Int) where {F}
    return 2 * sqrt(3) * H2Trees.halfsize(tree) * islf.k / (2.0^(level - 1)) <= 1
end

function (islf::IsLowFrequencyFunctor{F})(tree::BoundingBallTree, node::Int) where {F}
    parent(tree, node) == 0 && return islf(2radius(tree, node))
    return islf(radius(tree, parent(tree, node)))
end

function (islf::IsLowFrequencyFunctor{F})(diam::F) where {F}
    return diam * islf.k <= 1
end

struct IsNearFunctor{F}
    k::F
    ηₗ::F
    ηₕ::F
    islf::IsLowFrequencyFunctor{F}
end

function isnear(k::F; ηₗ::F=1.0, ηₕ::F=5.0, islf=islf(k)) where {F}
    return IsNearFunctor{F}(k, ηₗ, ηₕ, islf)
end

function (isnear::IsNearFunctor{F})(
    treea::TwoNTree, treeb::TwoNTree, nodea::Int, nodeb::Int
) where {F}
    ths = H2Trees.halfsize(treea, nodea) * sqrt(3)
    shs = H2Trees.halfsize(treeb, nodeb) * sqrt(3)
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
