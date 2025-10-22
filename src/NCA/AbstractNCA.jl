struct H2MatrixBlock{I,K}
    Z::Matrix{K}
    row_basis::I
    col_basis::I
end

struct H2BasisBlock{I,K}
    T::Union{Vector{Matrix{K}},Matrix{K}}
    τ::Vector{I}
    σ::Vector{I}
    children::Vector{I}
end
