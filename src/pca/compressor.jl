function getcompressedmatrix_rm(
    matrixassembler,
    test_idcs::Vector{I},
    trial_idcs::Vector{I},
    ::Type{K},
    am,
    compressor::PCAOptions{B, I, F};
    refcenter = SVector(0.0, 0.0, 0.0),
    noadm=false
) where {B, I, F, K}

    lm = FastBEAST.LazyMatrix(
        matrixassembler,
        test_idcs,
        trial_idcs,
        K
    )

    maxrank = max(Int(round(
        length(test_idcs) * length(trial_idcs)/(length(test_idcs) + length(trial_idcs)))),
        1
    )
    if noadm
        maxrank = min(length(test_idcs), length(trial_idcs))
    end

    pivstrat = compressor.columnpivstrat
    if compressor.columnpivstrat isa PCAPivoting
        pivstrat = PCAPivoting(
            refcenter, compressor.columnpivstrat.pos[trial_idcs]
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

    lm = FastBEAST.LazyMatrix(
        matrixassembler,
        test_idcs,
        trial_idcs,
        K
    )

    maxrank = max(Int(round(
        length(test_idcs) * length(trial_idcs)/(length(test_idcs) + length(trial_idcs)))),
        1
    )
    if noadm
        maxrank = min(length(test_idcs), length(trial_idcs))
    end

    pivstrat = compressor.rowpivstrat
    if compressor.rowpivstrat isa PCAPivoting
        pivstrat = PCAPivoting(
            refcenter, compressor.rowpivstrat.pos[test_idcs]
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