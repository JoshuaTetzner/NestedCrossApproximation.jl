function NCA(
    operator,
    space;
    tree=create_tree(space.pos, KMeansTreeOptions(; nmin=50)),
    nearinteractionquadstrat=BEAST.defaultquadstrat(operator, space, space),
    momentquadstrat=BEAST.DoubleNumQStrat(2, 3),
    compressor=TopDownCompressor(),
    multithreading=true,
    verbose=false,
    maxrank=40,
    η=1.0,
    tol=1e-4,
)
    blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
    nears, fars = FastBEAST.computeinteractions(blktree; η=η)

    nearinteractions = FastBEAST.assemble(
        operator,
        space,
        blktree,
        nears,
        scalartype(operator);
        quadstrat=nearinteractionquadstrat,
        multithreading=multithreading,
    )

    @views farblkassembler = BEAST.blockassembler(
        operator, space, space; quadstrat=momentquadstrat
    )
    @views function farassembler(Z, tdata, sdata)
        @views store(v, m, n) = (Z[m, n] += v)
        return farblkassembler(tdata, sdata, store)
    end

    moments, translations, pivots = compress_testfunctions(
        tree,
        farassembler,
        fars,
        compressor,
        scalartype(operator);
        multithreading=multithreading,
        maxrank=maxrank,
        tol=tol,
    )

    i2otranslator = I2Otranslator(
        farassembler, scalartype(operator), fars, pivots; multithreading=multithreading
    )

    return GalerkinNCA{scalartype(operator)}(
        blktree,
        nearinteractions,
        moments,
        i2otranslator,
        translations,
        fars,
        (tree.num_elements, tree.num_elements),
        verbose,
        multithreading,
    )
end
