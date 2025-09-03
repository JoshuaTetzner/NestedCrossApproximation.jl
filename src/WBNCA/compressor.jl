function compress(
    K::AbstractKernel{T},
    tree::BlockTree,
    level::Int,
    lfcompressor::AdaptiveCrossApproximation.ACA;
    isnear=H2Trees.isnear(),
    maxrank=40,
) where {T}
    blocks = MatrixBlock{Int,T,LowRankMatrix{T}}[]

    functor = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)
    iterator = functor(tree)

    for tnode in H2Trees.LevelIterator(H2Trees.testtree(tree), level)
        for snode in iterator(H2Trees.testtree(tree), H2Trees.trialtree(tree), tnode)
            #GlobalBuffer need to be implemented
            maxrows = length(H2Trees.values(H2Trees.testtree(tree), tnode))
            maxcols = length(H2Trees.values(H2Trees.trialtree(tree), snode))
            rowbuffer = zeros(T, maxrank, maxcols)
            colbuffer = zeros(T, maxrows, maxrank)
            comp = lfcompressor(
                K,
                H2Trees.values(H2Trees.testtree(tree), tnode),
                H2Trees.values(H2Trees.trialtree(tree), snode),
            )
            U, V = comp(
                K,
                rowbuffer,
                colbuffer,
                maxrank;
                rowidcs=H2Trees.values(H2Trees.testtree(tree), tnode),
                colidcs=H2Trees.values(H2Trees.trialtree(tree), snode),
            )
        end
    end
end
