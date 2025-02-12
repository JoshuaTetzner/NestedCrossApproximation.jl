abstract type Representor end

struct ChebyshevRep{I, D, F} <: Representor
    nodes::SVector{D, F}
    N::I
end

function ChebyshevRep(ε::F, η::F) where F
    N = Int(abs(log(η, ε)))
    cn = ChebyshevApprox.nodes(N,:chebyshev_nodes).points
    nodes3d = [SVector(cn[i], cn[j], cn[k]) for i=1:N for j=1:N for k=1:N]
    return ChebyshevRep(nodes3d, N^3)
end

#Covariance matrix (Bebendorf)
function covmat(M::Vector{SVector{D, F}}) where {D, F}
    Cₜ = zeros(F, D, D)
    cₘ = sum(M)/length(M)
    
    for mᵢ in M
        Cₜ += (mᵢ - cₘ)*transpose((mᵢ - cₘ))
    end
    s, _,_ = svd(Cₜ)

    hs = zeros(F, 3)
    for mᵢ in M
        h1 = dot((mᵢ - cₘ), s[1, 1:3])
        h2 = dot((mᵢ - cₘ), s[2, 1:3])
        h3 = dot((mᵢ - cₘ), s[3, 1:3])
        if (hs[1] < h1) hs[1] = h1 end
        if (hs[2] < h2) hs[2] = h2 end
        if (hs[3] < h3) hs[3] = h3 end
    end

    return s, hs, cₘ
end

function (cr::ChebyshevRep{I})(M::Vector{SVector{D, F}}) where {D, I, F}
    S, hs, cₘ = covmat(M)
    refnodes = copy(cr.nodes)
    
    for n in eachindex(refnodes)
        refnodes[n] = S*(refnodes[n] .* hs) + cₘ    
    end 

    idcs = zeros(Int, cr.N)
    used = zeros(Bool, cr.N)
    for (idx, refnode) in enumerate(refnodes)
        @views idcs[idx] = argmin(norm.(M .- Scalar(refnode)) .* !used)
        used[idcs[idx]] = true
    end

    return idcs
end


struct RandomRep{I} <: Representor
    N::I
end

function RandomRep(ε::F, η::F) where F
    return RandomRep(Int(abs(log(η, ε))))
end

function (cr::RandomRep{I})(M::Vector{SVector{D, F}}) where {D, I, F}
    return rand(1:length(M), cr.N)
end