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
    #println("Nears")
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

    #println("Fars")
    #println("pivotselection")
    cpivots = pivotselection(tree, fars, compressor.cpivstrat;
     maxrank=25, multithreading=multithreading)
    #return cpivots
    #println("assembly")
    momentcollection, translator, rpivots = assembly(
        cpivots, tree, farassembler, scalartype(operator), tol=tol
    )

    i2otranslator = assemble_couplingmatrices(
        farassembler,
        scalartype(operator), 
        fars, 
        rpivots,
        cpivots;
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