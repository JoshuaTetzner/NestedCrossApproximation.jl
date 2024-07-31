function build_test_bases(
    tree::ClusterTrees.NminTrees.NminTree{T},
    test_fars::Vector{PivotBlocks{I, K}},
    ::Type{K};
    multithreading=true,
    verbose=false
) where {I,T,K}

    test_basis = Vector{H2BasisBlock{I,K}}(undef, length(tree.nodes))
    test_transfer = Vector{H2BasisBlock{I,K}}(undef, length(tree.nodes))
    if verbose
        p = Progress(length(test_fars), desc="Assemble row bases: ")
    end

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(eachindex(test_fars)) do idx
        verbose && next!(p)
        if isassigned(test_fars, idx)
            testfar = test_fars[idx]
            if testfar.children == []
                test_basis[idx] = H2BasisBlock(
                    test_fars[idx].M.M.U * test_fars[idx].M.M.U[test_fars[idx].M.M.τ, :]^-1,
                    test_fars[idx].M.τ,
                    test_fars[idx].M.σ,
                    Int[],
                )
            else
                transfer = Vector{Matrix{K}}(undef, length(testfar.children))
                children = zeros(Int, length(testfar.children))
                for (ind, (child, range)) in enumerate(testfar.children)
                    transfer[ind] =
                        testfar.M.M.U[range, :][test_fars[child].M.M.τ, :] *
                        testfar.M.M.U[testfar.M.M.τ, :]^-1
                    children[ind] = child
                end
                test_transfer[idx] = H2BasisBlock(
                    transfer, test_fars[idx].M.τ, test_fars[idx].M.σ, children
                )
            end
        end
    end

    return test_basis, test_transfer
end


function build_trial_bases(
    tree::ClusterTrees.NminTrees.NminTree{T},
    trial_fars::Vector{PivotBlocks{I, K}},
    ::Type{K};
    multithreading=true,
    verbose=false
) where {I,T,K}

    trial_basis = Vector{H2BasisBlock{I,K}}(undef, length(tree.nodes))
    trial_transfer = Vector{H2BasisBlock{I,K}}(undef, length(tree.nodes))
    if verbose
        p = Progress(length(trial_fars), desc="Assemble column bases: ")
    end

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(eachindex(trial_fars)) do idx
        verbose && next!(p)
        if isassigned(trial_fars, idx)
            trialfar = trial_fars[idx]
            if trialfar.children == []
                trial_basis[idx] = H2BasisBlock(
                    trial_fars[idx].M.M.V[:, trial_fars[idx].M.M.σ]^-1 * trial_fars[idx].M.M.V,
                    trial_fars[idx].M.τ,
                    trial_fars[idx].M.σ,
                    Int[],
                )
            else
                transfer = Vector{Matrix{K}}(undef, length(trialfar.children))
                children = zeros(Int, length(trialfar.children))
                for (ind, (child, range)) in enumerate(trialfar.children)
                    transfer[ind] =
                        trialfar.M.M.V[:, trialfar.M.M.σ]^-1 *
                        trialfar.M.M.V[:, range][:, trial_fars[child].M.M.σ]
                    children[ind] = child
                end
                trial_transfer[idx] = H2BasisBlock(
                    transfer, trial_fars[idx].M.τ, trial_fars[idx].M.σ, children
                )
            end
        end
    end

    return trial_basis, trial_transfer
end
