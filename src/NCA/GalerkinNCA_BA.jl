function GalerkinNCA2(
    operator,
    space; 
    tree=create_tree(space.pos, KMeansTreeOptions()),
    nearinteractionquadstrat=BEAST.defaultquadstrat(operator, space, space),
    momentquadstrat=BEAST.DoubleNumQStrat(2, 3),
    compressor=FastBEAST.ACAOptions(; tol=1e-3),
    multithreading=true,
    verbose=false,
    η=1.0,
    tol =1e-3
)
    blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
    nears, fars = FastBEAST.computeinteractions(blktree,  η=η)

    nearinteractions = FastBEAST.assemble(
        operator,
        space,
        blktree,
        nears, 
        scalartype(operator);
        quadstrat=nearinteractionquadstrat,
        multithreading=multithreading
    )

    @views farblkassembler = BEAST.blockassembler(
        operator, space, space, quadstrat=momentquadstrat
    )
    @views function farassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        farblkassembler(tdata,sdata,store)
    end

    @views cpivots = pivotselection(
        tree, fars, compressor.cpivstrat;
        maxrank=compressor.maxrank, multithreading=multithreading
    )

    momentcollection, translator, rpivots = assembly(
        cpivots, tree, farassembler, scalartype(operator), tol=tol
    )

    i2otranslator = i2o(
        farassembler,
        scalartype(operator), 
        fars, 
        rpivots;
        multithreading=multithreading, 
    )

    return GalerkinNCA{scalartype(operator)}(
        blktree,
        nearinteractions,
        momentcollection,
        i2otranslator,
        translator,
        fars,
        (tree.num_elements, tree.num_elements),
        false,
        multithreading
    )
end