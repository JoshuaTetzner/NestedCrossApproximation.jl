using LinearMaps

#storage for pivoting
struct ClusterMatrix{I, K} <: LinearMaps.LinearMap{K}
    U::Matrix{K}
    V::Matrix{K}
    τ::Vector{I}
    σ::Vector{I}
end

Base.size(lrm::ClusterMatrix) = (size(lrm.U, 1), size(lrm.V, 2))

struct PivotBlocks{I, K}
    M::FastBEAST.MatrixBlock{I, K, ClusterMatrix{I, K}}
    interactionmap::Tuple{Vector{I}, Vector{I}}
    children::Vector{Tuple{I, UnitRange{I}}}
end

struct H2MatrixBlock{I,K}
    Z::FastBEAST.MatrixBlock{I,K,Matrix{K}}
    τ::Vector{I}
    σ::Vector{I}
    row_basis::I
    col_basis::I
end

struct H2BasisBlock{I,K}
    T::Union{Vector{Matrix{K}}, Matrix{K}}
    τ::Vector{I}
    σ::Vector{I}
    children::Vector{I}
end

