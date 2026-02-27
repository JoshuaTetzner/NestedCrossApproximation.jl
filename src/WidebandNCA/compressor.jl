
function testcompress(
    compressor,
    farmatrix::AbstractKernelMatrix{T},
    rowidcs::Vector{Int},
    colidcs::Vector{Int},
    colbuffer::AbstractMatrix{T},
    rowbuffer::AbstractMatrix{T};
    maxrank=40,
) where {T}
    fill!(view(colbuffer, rowidcs, 1:maxrank), zero(T))
    npivots, rows, cols = compressor.lrf(
        farmatrix,
        view(colbuffer, rowidcs, 1:maxrank),
        rowbuffer,
        maxrank;
        rows=zeros(Int, maxrank),
        cols=zeros(Int, maxrank),
        rowidcs=rowidcs,
        colidcs=colidcs,
    )
    @views colbuffer[rowidcs, 1:npivots] =
        colbuffer[rowidcs, 1:npivots] * rowbuffer[1:npivots, 1:npivots]
    fill!(view(rowbuffer, 1:npivots, 1:npivots), zero(T))
    return rows, cols
end

function trialcompress(
    compressor,
    farmatrix::AbstractKernelMatrix{T},
    rowidcs::Vector{Int},
    colidcs::Vector{Int},
    colbuffer::AbstractMatrix{T},
    rowbuffer::AbstractMatrix{T};
    maxrank=40,
) where {T}
    rowbuffer[1:maxrank, colidcs] .= 0.0
    npivots, rows, cols = compressor.lrf(
        farmatrix,
        colbuffer,
        view(rowbuffer, 1:maxrank, colidcs),
        maxrank;
        rows=zeros(Int, maxrank),
        cols=zeros(Int, maxrank),
        rowidcs=rowidcs,
        colidcs=colidcs,
    )
    rowbuffer[1:npivots, colidcs] =
        colbuffer[1:npivots, 1:npivots] * rowbuffer[1:npivots, colidcs]
    colbuffer[1:npivots, 1:npivots] .= 0.0
    return rows, cols
end
