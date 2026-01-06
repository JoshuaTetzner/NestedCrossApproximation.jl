function lbases(h2mat::NestedCrossApproximation.PetrovGalerkinWNCA{K}) where {K}
    trialbases = Vector{Dict{Int,Matrix{K}}}(undef, length(h2mat.tree.testcluster.nodes))
    testbases = Vector{Dict{Int,Matrix{K}}}(undef, length(h2mat.tree.testcluster.nodes))

    for (i, nb) in h2mat.nestedtestbases
        testbases[i] = nb
    end

    for (i, d) in h2mat.nestedtrialbases
        trialbases[i] = d
    end

    for level in reverse(H2Trees.levels(h2mat.tree.testcluster))
        for t in H2Trees.LevelIterator(h2mat.tree.testcluster, level)
            !haskey(h2mat.testtransfermatrices, t) && continue
            dirmats = Matrix{ComplexF64}[]
            for transfer in values(h2mat.testtransfermatrices[t])
                push!(
                    dirmats,
                    mapreduce(
                        vcat,
                        enumerate(collect(H2Trees.children(h2mat.tree.testcluster, t))),
                    ) do (cidx, c)
                        testbases[c][transfer[cidx][1]] * transfer[cidx][2]
                    end,
                )
            end
            testbases[t] = Dict(keys(h2mat.testtransfermatrices[t]) .=> dirmats)
        end
    end

    for level in reverse(H2Trees.levels(h2mat.tree.trialcluster))
        for s in H2Trees.LevelIterator(h2mat.tree.trialcluster, level)
            !haskey(h2mat.trialtransfermatrices, s) && continue
            dirmats = Matrix{ComplexF64}[]
            for transfer in values(h2mat.trialtransfermatrices[s])
                push!(
                    dirmats,
                    mapreduce(
                        hcat,
                        enumerate(collect(H2Trees.children(h2mat.tree.trialcluster, s))),
                    ) do (cidx, c)
                        transfer[cidx][2] * trialbases[c][transfer[cidx][1]]
                    end,
                )
            end
            trialbases[s] = Dict(keys(h2mat.trialtransfermatrices[s]) .=> dirmats)
        end
    end

    return testbases, trialbases
end

function reconstruct(h2mat::NestedCrossApproximation.PetrovGalerkinWNCA)
    A_h2 = zeros(eltype(h2mat), size(h2mat, 1), size(h2mat, 2))
    testbases, trialbases = lbases(h2mat)
    for couplings in h2mat.couplingmatrices
        for (key, coupling) in couplings
            if norm(
                A_h2[
                    H2Trees.values(h2mat.tree.testcluster, key[1]),
                    H2Trees.values(h2mat.tree.trialcluster, key[2]),
                ],
            ) != 0.0
                @warn "Overwriting with fars!"
            end
            A_h2[
                H2Trees.values(h2mat.tree.testcluster, key[1]),
                H2Trees.values(h2mat.tree.trialcluster, key[2]),
            ] +=
                testbases[key[1]][coupling[1][1]] *
                coupling[2] *
                trialbases[key[2]][coupling[1][2]]
        end
    end

    for (i, block) in enumerate(h2mat.nearinteractions.blocks)
        if norm(
            A_h2[h2mat.nearinteractions.rowindices[i], h2mat.nearinteractions.colindices[i]]
        ) != 0.0
            @warn "Overwriting with near interactions!"
        end
        A_h2[h2mat.nearinteractions.rowindices[i], h2mat.nearinteractions.colindices[i]] +=
            block
    end

    return A_h2
end
