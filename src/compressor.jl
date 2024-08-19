function getcompressedmatrixview(
    matrixassembler,
    testidcs::Vector{I},
    trialidcs::Vector{I},
    ::Type{K},
    am,
    compressor::FastBEAST.ACAOptions{B, I, F}
) where {B, I, F, K}

    lm = FastBEAST.LazyMatrix(matrixassembler, testidcs, trialidcs, K)

    compressor.maxrank == 0 ? compressor.maxrank = Int(
        round(length(lm.τ)*length(lm.σ)/(length(lm.τ)+length(lm.σ)))
    ) : maxrank=compressor.maxrank
    
    U, V, rows, cols = aca(
        lm,
        am;
        rowpivstrat=compressor.rowpivstrat,
        columnpivstrat=compressor.columnpivstrat,
        convcrit=compressor.convcrit,
        tol=compressor.tol,
        svdrecompress=compressor.svdrecompress,
        maxrank=maxrank
    )

    @views MU = U * V[:, cols]
    @views MV = U[rows, :] * V

    return MatrixBlock{I, K, ClusterMatrix{I, K}}(
        ClusterMatrix(MU, MV, rows, cols),
        testidcs,
        trialidcs
    )
end