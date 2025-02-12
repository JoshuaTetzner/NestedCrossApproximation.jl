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
    trialbases = Vector{Matrix{K}}(undef, length(h2mat.tree.test_cluster.nodes))
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
