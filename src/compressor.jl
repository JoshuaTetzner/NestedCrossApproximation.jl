function getcompressedmatrixview(
    matrixassembler,
    testidcs::Vector{I},
    trialidcs::Vector{I},
    ::Type{K},
    am::FastBEAST.ACAGlobalMemory{I, F, K},
    compressor::FastBEAST.ACAOptions{B, I, F};
    noadm=false,
    tolmult=tolmult
) where {B, I, F, K}
    
    maxrank = min(Int(round(
        length(testidcs) * length(trialidcs)/(length(testidcs) + length(trialidcs)))),
        compressor.maxrank
    )
    if noadm
        maxrank = min(length(test_idcs), length(trial_idcs))
    end
    
    #am = allocate_aca_memory(K, length(testidcs), length(trialidcs); maxrank=maxrank)
    lm = FastBEAST.LazyMatrix(matrixassembler, testidcs, trialidcs, K)

    U, V, rows, cols = aca(
        lm,
        am;
        rowpivstrat=compressor.rowpivstrat,
        columnpivstrat=compressor.columnpivstrat,
        convcrit=compressor.convcrit,
        tol=compressor.tol,#*tolmult,
        svdrecompress=compressor.svdrecompress,
        maxrank=maxrank
    )
    println(trialidcs[cols])
    @views MU = U * V[:, cols]
    @views MV = U[rows, :] * V
    #MV = V

    return MatrixBlock{I, K, ClusterMatrix{I, K}}(
        ClusterMatrix(MU, MV, rows, cols),
        testidcs,
        trialidcs
    )
end