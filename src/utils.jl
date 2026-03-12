#=using BlockSparseMatrices

function storage(h2mat::GalerkinNCA)
    ref = size(h2mat, 1) * size(h2mat, 2)
    h2stor = 0.0
    for frb in h2mat.nearinteractions.diagonals
        h2stor += length(frb.rowindices) * length(frb.colindices)
    end
    for frb in h2mat.nearinteractions.offdiagonals
        h2stor += length(frb.rowindices) * length(frb.colindices)
    end

    for lrb in h2mat.couplingmatrices
        h2stor += length(lrb.Z)
    end

    for (ind, ntb) in h2mat.nestedbases
        h2stor += size(ntb.T, 1) * size(ntb.T, 2)
    end
    for level in h2mat.transfermatrices
        for (ind, tran) in level
            for t in tran.T
                h2stor += size(t, 1) * size(t, 2)
            end
        end
    end

    return h2stor * 8 * 10^-9, h2stor / ref
end

function storage(h2mat::PetrovGalerkinNCA)
    ref = size(h2mat, 1) * size(h2mat, 2)
    h2stor = 0.0
    for frb in h2mat.nearinteractions.blocks
        h2stor += length(frb.rowindices) * length(frb.colindices)
    end

    for lrb in h2mat.couplingmatrices
        h2stor += length(lrb.Z)
    end

    for (ind, ntb) in h2mat.nestedtestbases
        h2stor += size(ntb.T, 1) * size(ntb.T, 2)
    end

    for (ind, ntb) in h2mat.nestedtrialbases
        h2stor += size(ntb.T, 1) * size(ntb.T, 2)
    end

    for level in h2mat.testtransfermatrices
        for (ind, tran) in level
            for t in tran.T
                h2stor += size(t, 1) * size(t, 2)
            end
        end
    end

    for level in h2mat.trialtransfermatrices
        for (ind, tran) in level
            for t in tran.T
                h2stor += size(t, 1) * size(t, 2)
            end
        end
    end

    return h2stor * 8 * 10^-9, h2stor / ref
end

function lbases(h2mat::GalerkinNCA{K}) where {K}
    trialbases = Vector{Matrix{K}}(undef, length(h2mat.tree.test_cluster.nodes))
    testbases = Vector{Matrix{K}}(undef, length(h2mat.tree.test_cluster.nodes))
    for (ind, b) in h2mat.nestedbases
        trialbases[ind] = transpose(b.T)
        testbases[ind] = b.T
    end

    for (i, level) in enumerate(reverse(h2mat.transfermatrices))
        for (i, t) in level
            T = vcat(testbases[t.children[1]] * t.T[1], testbases[t.children[2]] * t.T[2])
            trialbases[i] = transpose(T)
            testbases[i] = T
        end
    end
    return testbases, trialbases
end

function lbases(h2mat::PetrovGalerkinNCA{K}) where {K}
    trialbases = Vector{Matrix{K}}(undef, length(h2mat.tree.trial_cluster.nodes))
    testbases = Vector{Matrix{K}}(undef, length(h2mat.tree.test_cluster.nodes))
    for (ind, b) in h2mat.nestedtestbases
        testbases[ind] = b.T
    end
    for (ind, b) in h2mat.nestedtrialbases
        trialbases[ind] = b.T
    end

    for (i, level) in enumerate(reverse(h2mat.testtransfermatrices))
        for (i, t) in level
            T = reduce(
                vcat,
                [testbases[t.children[child]] * t.T[child] for child in eachindex(t.T)],
            )
            testbases[i] = T
        end
    end

    for (i, level) in enumerate(reverse(h2mat.trialtransfermatrices))
        for (i, t) in level
            T = reduce(
                hcat,
                [t.T[child] * trialbases[t.children[child]] for child in eachindex(t.T)],
            )
            trialbases[i] = T
        end
    end

    return testbases, trialbases
end

function fullmat(h2mat::GalerkinNCA)
    A_h2 = zeros(eltype(h2mat), size(h2mat, 1), size(h2mat, 2))
    for M in h2mat.nearinteractions.diagonals
        A_h2[M.rowindices, M.colindices] = M.matrix
    end
    for M in h2mat.nearinteractions.offdiagonals
        A_h2[M.rowindices, M.colindices] = M.matrix
        A_h2[M.colindices, M.rowindices] = transpose(M.matrix)
    end
    testbases, trialbases = lbases(h2mat)
    for i2o in h2mat.couplingmatrices
        A_h2[
            value(h2mat.tree.test_cluster, i2o.row_basis),
            value(h2mat.tree.trial_cluster, i2o.col_basis),
        ] = testbases[i2o.row_basis] * i2o.Z * trialbases[i2o.col_basis]
        A_h2[
            value(h2mat.tree.trial_cluster, i2o.col_basis),
            value(h2mat.tree.test_cluster, i2o.row_basis),
        ] = testbases[i2o.col_basis] * transpose(i2o.Z) * trialbases[i2o.row_basis]
    end
    return A_h2
end

function fullmat(h2mat::PetrovGalerkinNCA)
    A_h2 = zeros(eltype(h2mat), size(h2mat, 1), size(h2mat, 2))
    for M in h2mat.nearinteractions.blocks
        A_h2[M.rowindices, M.colindices] = M.matrix
    end

    testbases, trialbases = lbases(h2mat)
    for i2o in h2mat.couplingmatrices
        A_h2[
            value(h2mat.tree.test_cluster, i2o.row_basis),
            value(h2mat.tree.trial_cluster, i2o.col_basis),
        ] = testbases[i2o.row_basis] * i2o.Z * trialbases[i2o.col_basis]
    end
    return A_h2
end

function lrbmat(h2mat)
    A_h2 = zeros(eltype(h2mat), size(h2mat, 1), size(h2mat, 2))
    testbases, trialbases = lbases(h2mat)
    for i2o in h2mat.couplingmatrices
        A_h2[
            value(h2mat.tree.test_cluster, i2o.row_basis),
            value(h2mat.tree.trial_cluster, i2o.col_basis),
        ] = testbases[i2o.row_basis] * i2o.Z * trialbases[i2o.col_basis]
    end

    return A_h2
end

function lrbmat(A, h2mat)
    lrbA = zeros(eltype(h2mat), size(h2mat, 1), size(h2mat, 2))
    for i2o in h2mat.couplingmatrices
        lrbA[value(h2mat.tree.test_cluster, i2o.row_basis), value(h2mat.tree.trial_cluster, i2o.col_basis)] = A[
            value(h2mat.tree.test_cluster, i2o.row_basis),
            value(h2mat.tree.trial_cluster, i2o.col_basis),
        ]
    end

    return lrbA
end

function lrbh2mat(h2mat::PetrovGalerkinNCA)
    blocks = BlockSparseMatrices.DenseMatrixBlock{
        ComplexF64,Matrix{ComplexF64},Vector{Int}
    }[]
    nears = BlockSparseMatrix(blocks, h2mat.dim)

    return PetrovGalerkinNCA{ComplexF64}(
        h2mat.tree,
        nears,
        h2mat.nestedtestbases,
        h2mat.nestedtrialbases,
        h2mat.testtransfermatrices,
        h2mat.trialtransfermatrices,
        h2mat.couplingmatrices,
        h2mat.fars,
        h2mat.dim,
        h2mat.ismultithreaded,
    )
end

function lrbh2mat(h2mat::NestedCrossApproximation.PetrovGalerkinWNCA)
    blocks = BlockSparseMatrices.DenseMatrixBlock{
        ComplexF64,Matrix{ComplexF64},Vector{Int}
    }[]
    nears = BlockSparseMatrix(blocks, h2mat.dim)

    return PetrovGalerkinWNCA{ComplexF64}(
        h2mat.tree,
        nears,
        h2mat.nestedtestbases,
        h2mat.nestedtrialbases,
        h2mat.testtransfermatrices,
        h2mat.trialtransfermatrices,
        h2mat.couplingmatrices,
        h2mat.dim,
        h2mat.ntasks,
    )
end

function storage(h2::NestedCrossApproximation.PetrovGalerkinWNCA)
    ref = size(h2, 1) * size(h2, 2)
    h2stor = 0.0

    for frb in eachindex(h2.nearinteractions.blocks)
        h2stor +=
            length(h2.nearinteractions.rowindices[frb]) *
            length(h2.nearinteractions.colindices[frb])
    end

    for t in h2.couplingmatrices
        for blk in values(t)
            h2stor += length(blk[2])
        end
    end

    for ntbs in values(h2.nestedtestbases)
        for ntb in values(ntbs)
            h2stor += length(ntb)
        end
    end

    for ntbs in values(h2.nestedtrialbases)
        for ntb in values(ntbs)
            h2stor += length(ntb)
        end
    end

    for trans in values(h2.testtransfermatrices)
        for ttrans in values(trans)
            for tran in ttrans
                h2stor += length(tran)
            end
        end
    end

    for trans in values(h2.trialtransfermatrices)
        for ttrans in values(trans)
            for tran in ttrans
                h2stor += length(tran)
            end
        end
    end

    return h2stor * 8 * 10^-9, h2stor / ref
end
=#

function collect_assigned(v::Vector{B}) where {B}
    nassigned = 0
    @inbounds for i in eachindex(v)
        nassigned += isassigned(v, i)
    end

    compact = Vector{B}(undef, nassigned)
    nodes = Vector{Int}(undef, nassigned)

    k = 0
    @inbounds for i in eachindex(v)
        if isassigned(v, i)
            k += 1
            compact[k] = v[i]
            nodes[k] = i
        end
    end

    return compact, nodes
end
