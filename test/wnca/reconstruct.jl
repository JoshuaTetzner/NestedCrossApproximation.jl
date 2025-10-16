function lbases(h2mat::NestedCrossApproximation.PetrovGalerkinWNCA{K}) where {K}
    trialbases = Vector{Dict{Int,Matrix{K}}}(undef, length(h2mat.tree.testcluster.nodes))
    testbases = Vector{Dict{Int,Matrix{K}}}(undef, length(h2mat.tree.testcluster.nodes))

    for (i, d) in h2mat.nestedtestbases
        dirmats = Matrix{ComplexF64}[]
        for (_, amat) in d
            push!(dirmats, amat.T)
        end
        testbases[i] = Dict(keys(d) .=> dirmats)
    end

    for (i, d) in h2mat.nestedtrialbases
        dirmats = Matrix{ComplexF64}[]
        for (_, amat) in d
            push!(dirmats, amat.T)
        end
        trialbases[i] = Dict(keys(d) .=> dirmats)
    end

    for level in reverse(h2mat.testtransfermatrices)
        for (j, t) in level
            dirmats = Matrix{ComplexF64}[]
            for (dir, transfer) in t
                base =
                    testbases[transfer.children[1]][NestedCrossApproximation.parent(
                        h2mat.dtree, dir
                    )] * transfer.T[1]
                for c in 2:length(transfer.children)
                    base = vcat(
                        base,
                        testbases[transfer.children[c]][NestedCrossApproximation.parent(
                            h2mat.dtree, dir
                        )] * transfer.T[c],
                    )
                end
                push!(dirmats, base)
            end
            testbases[j] = Dict(keys(t) .=> dirmats)
        end
    end

    for level in reverse(h2mat.trialtransfermatrices)
        for (j, t) in level
            dirmats = Matrix{ComplexF64}[]
            for (dir, transfer) in t
                if !allunique(transfer.children)
                    println("fail")
                end
                base =
                    transfer.T[1] *
                    trialbases[transfer.children[1]][NestedCrossApproximation.parent(
                        h2mat.dtree, dir
                    )]
                for c in 2:length(transfer.children)
                    base = hcat(
                        base,
                        transfer.T[c] *
                        trialbases[transfer.children[c]][NestedCrossApproximation.parent(
                            h2mat.dtree, dir
                        )],
                    )
                end
                push!(dirmats, base)
            end

            trialbases[j] = Dict(keys(t) .=> dirmats)
        end
    end

    return testbases, trialbases
end

function reconstruct(h2mat::NestedCrossApproximation.PetrovGalerkinWNCA)
    A_h2 = zeros(eltype(h2mat), size(h2mat, 1), size(h2mat, 2))
    testbases, trialbases = lbases(h2mat)
    for coupling in h2mat.couplingmatrices
        A_h2[
            H2Trees.values(h2mat.tree.testcluster, coupling.row_basis),
            H2Trees.values(h2mat.tree.trialcluster, coupling.col_basis),
        ] +=
            testbases[coupling.row_basis][coupling.dir] *
            coupling.Z *
            trialbases[coupling.col_basis][coupling.dir]
    end

    for lf in h2mat.lowfrequencyinteractions
        A_h2[lf.τ, lf.σ] += lf.M.U * lf.M.V
    end

    for block in h2mat.nearinteractions.blocks
        A_h2[block.rowindices, block.colindices] += block.matrix
    end

    return A_h2
end
