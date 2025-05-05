using ChebyshevApprox

abstract type Representor end

struct PetrovGelarkinChebyshevRep{I,D,F} <: Representor
    testpos::Vector{SVector{D,F}}
    trialpos::Vector{SVector{D,F}}
    nodes::Vector{SVector{D,F}}
    N::I
end

struct ChebyshevRep{I,D,F} <: Representor
    pos::Vector{SVector{D,F}}
    nodes::Vector{SVector{D,F}}
    N::I
end

function ChebyshevRep(
    ε::F,
    η::F,
    pos::Vector{SVector{D,F}};
    γ=0.35,
    N=Int(abs(round(log((sqrt(3) * γ * η), ε)))),
) where {D,F}
    cn = ChebyshevApprox.nodes(N, :chebyshev_nodes).points
    #To-Do: This should use D to determine the dimension. I think 2D is better in general
    #Bebendorf proposes 3D.
    nodes3d = [SVector(cn[i], cn[j], cn[k]) for i in 1:N for j in 1:N for k in 1:N]
    return ChebyshevRep(pos, nodes3d, N^3)
end

#Covariance matrix (Bebendorf)
function covmat(M::AbstractVector{SVector{D,F}}) where {D,F}
    Cₜ = zeros(F, D, D)
    cₘ = sum(M) / length(M)

    for mᵢ in M
        Cₜ += (mᵢ - cₘ) * transpose((mᵢ - cₘ))
    end
    s, _, _ = svd(Cₜ)

    hs = zeros(F, 3)
    for mᵢ in M
        h1 = dot((mᵢ - cₘ), s[1:3, 1])
        h2 = dot((mᵢ - cₘ), s[1:3, 2])
        h3 = dot((mᵢ - cₘ), s[1:3, 3])
        if (hs[1] < h1)
            hs[1] = h1
        end
        if (hs[2] < h2)
            hs[2] = h2
        end
        if (hs[3] < h3)
            hs[3] = h3
        end
    end

    return s, hs, cₘ
end

function (cr::ChebyshevRep{I})(rc::Vector{Int}) where {I}
    length(rc) < cr.N && return rc
    @views S, hs, cₘ = covmat(cr.pos[rc])
    refnodes = copy(cr.nodes)

    @views for n in eachindex(refnodes)
        refnodes[n] = S * (refnodes[n] .* hs) + cₘ
    end

    idcs = zeros(Int, cr.N)
    used = zeros(Bool, length(rc))
    @views for (idx, refnode) in enumerate(refnodes)
        minval = norm(cr.pos[argmin(used)] - refnode)
        idcs[idx] = argmin(used)
        for (nidx, node) in enumerate(cr.pos[rc])
            if !used[nidx] && (minval > norm(node - refnode) || minval == 0.0)
                minval = norm(node - refnode)
                idcs[idx] = nidx
            end
        end
        used[idcs[idx]] = true
    end

    return rc[idcs]
end

struct RandomRep{I} <: Representor
    N::I
end

function RandomRep(ε::F, η::F) where {F}
    return RandomRep(Int(abs(log(η, ε))))
end

function (cr::RandomRep{I})(M::Vector{SVector{D,F}}) where {D,I,F}
    return rand(M, cr.N)
end
