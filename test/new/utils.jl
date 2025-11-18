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
