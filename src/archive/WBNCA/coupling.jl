struct DH2MatrixBlock{I,K}
    Z::Matrix{K}
    row_basis::I
    row_dir::I
    col_basis::I
    col_dir::I
end

function computecoupling(
    farassembler::Function,
    tpivots::Vector{Dict{Int,Tuple{Vector{Int},Vector{Int}}}},
    tdirfars::Vector{Dict{Int,Vector{Int}}},
    spivots::Vector{Dict{Int,Tuple{Vector{Int},Vector{Int}}}},
    sdirfars::Vector{Dict{Int,Vector{Int}}},
    fars::Vector{Tuple{Int,Int}};
    multithreading=true,
)
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    coupling = Vector{DH2MatrixBlock{Int,ComplexF64}}(undef, length(fars))
    _foreach(enumerate(fars)) do (i, far)
        rows = tpivots[far[1]][direction(tdirfars[far[1]], far[2])][1]
        cols = spivots[far[2]][direction(sdirfars[far[2]], far[1])][1]
        blk = zeros(ComplexF64, length(rows), length(cols))
        farassembler(blk, rows, cols)
        coupling[i] = DH2MatrixBlock(
            blk,
            far[1],
            direction(tdirfars[far[1]], far[2]),
            far[2],
            direction(sdirfars[far[2]], far[1]),
        )
    end
    return coupling
end
