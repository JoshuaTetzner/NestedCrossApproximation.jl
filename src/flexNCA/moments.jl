function build_trialmoments(
    rbuffer::Matrix{K},
    pivots::Vector{Tuple{Vector{I},Vector{I}}},
    clusterlink,
    fars;
    multithreading=true,
) where {I,K}
    testmoments = Vector{H2BasisBlock{I,K}}(undef, length(tree.nodes))
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(clusterlink[end]) do node
        if fars[node] != []
            U =
                inv(rbuffer[1:length(pivots[node][1]), pivots[node][2]]) *
                rbuffer[1:length(pivots[node][1]), value(tree, node)]

            testmoments[node] = H2BasisBlock(U, pivots[node][1], pivots[node][2], Int[])
        end
    end
end
