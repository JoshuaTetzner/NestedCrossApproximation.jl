function getcompressedmatrix_rm2(
    matrixassembler,
    test_idcs::Vector{I},
    trial_idcs::Vector{I},
    ::Type{K},
    compressor::PCAOptions{B, I, F};
    refcenter = SVector(0.0, 0.0, 0.0),
    noadm=false
) where {B, I, F, K}
    
    maxrank = min(Int(round(
        length(test_idcs) * length(trial_idcs)/(length(test_idcs) + length(trial_idcs)))),
        compressor.maxrank
    )
    if noadm
        maxrank = min(length(test_idcs), length(trial_idcs))
    end

    #am = allocate_pca_memory_rm(K, length(test_idcs), length(trial_idcs); maxrank=maxrank)
    lm = FastBEAST.LazyMatrix(
        matrixassembler,
        test_idcs,
        trial_idcs,
        K
    )

    pivstrat = compressor.columnpivstrat
    if compressor.columnpivstrat isa PCAPivoting
        pivstrat = PCAPivoting(
            compressor.columnpivstrat.fct, refcenter, compressor.columnpivstrat.pos[trial_idcs]
        )
    end

    U, V, rows, cols = pca_rm(
        lm,
        am,
        pivstrat;
        maxrank=maxrank,
        tol=compressor.tol
    )

    return MatrixBlock{I, K, ClusterMatrix{I, K}}(
        ClusterMatrix(U, V, rows, cols),
        test_idcs,
        trial_idcs
    )
end



function getcompressedmatrix_rm(
    matrixassembler,
    test_idcs::Vector{I},
    trial_idcs::Vector{I},
    ::Type{K},
    am::PCAGlobalMemory{K},
    compressor::PCAOptions{B, I, F};
    refcenter = SVector(0.0, 0.0, 0.0),
    noadm=false
) where {B, I, F, K}
    
    maxrank = min(Int(round(
        length(test_idcs) * length(trial_idcs)/(length(test_idcs) + length(trial_idcs)))),
        compressor.maxrank
    )
    if noadm
        maxrank = min(length(test_idcs), length(trial_idcs))
    end

    lm = FastBEAST.LazyMatrix(
        matrixassembler,
        test_idcs,
        trial_idcs,
        K
    )

    pivstrat = compressor.columnpivstrat
    if compressor.columnpivstrat isa PCAPivoting
        pivstrat = PCAPivoting(
            compressor.columnpivstrat.fct, refcenter, compressor.columnpivstrat.pos[trial_idcs]
        )
    end

    U, V, rows, cols = pca_rm(
        lm,
        am,
        pivstrat;
        maxrank=maxrank,
        tol=compressor.tol
    )

    return MatrixBlock{I, K, ClusterMatrix{I, K}}(
        ClusterMatrix(U, V, rows, cols),
        test_idcs,
        trial_idcs
    )
end

function getcompressedmatrix_cm(
    matrixassembler,
    test_idcs::Vector{I},
    trial_idcs::Vector{I},
    ::Type{K},
    am,
    compressor::PCAOptions{B, I, F};
    refcenter=SVector(F(0.0),F(0.0),F(0.0)),
    noadm=false
) where {B, I, F, K}

    maxrank = min(Int(round(
        length(test_idcs) * length(trial_idcs)/(length(test_idcs) + length(trial_idcs)))),
        compressor.maxrank
    )
    if noadm
        maxrank = min(length(test_idcs), length(trial_idcs))
    end
    #am = allocate_pca_memory_cm(K, length(test_idcs), length(trial_idcs); maxrank=maxrank)
    lm = FastBEAST.LazyMatrix(
        matrixassembler,
        test_idcs,
        trial_idcs,
        K
    )

    pivstrat = compressor.rowpivstrat
    if compressor.rowpivstrat isa PCAPivoting
        pivstrat = PCAPivoting(
            compressor.rowpivstrat.fct, refcenter, compressor.rowpivstrat.pos[test_idcs]
        )
    end

    U, V, rows, cols = pca_cm(
        lm,
        am,
        pivstrat;
        maxrank=maxrank,
        tol=compressor.tol
    )

    return MatrixBlock{I, K, ClusterMatrix{I, K}}(
        ClusterMatrix(U, V, rows, cols),
        test_idcs,
        trial_idcs
    )
end