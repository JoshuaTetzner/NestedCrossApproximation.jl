
function PetrovGalerkinMCA(
    operator,
    testspace,
    trialspace;
    testtree=create_tree(testspace.pos, KMeansTreeOptions(; nmin=50, maxlevel=50)),
    trialtree=create_tree(trialspace.pos, KMeansTreeOptions(; nmin=50, maxlevel=50)),
    nearinteractionquadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
    momentquadstrat=BEAST.DoubleNumQStrat(2, 3),
    multithreading=true,
    maxrank=40, #Should be moved to the compressor
    tol=1e-4, #global
    η=1.0, #global
)
    blktree = ClusterTrees.BlockTrees.BlockTree(testtree, trialtree)
    nears, fars = computeinteractions(blktree; η=η)

    nearinteractions = FastBEAST.assemble(
        operator,
        testspace,
        trialspace,
        blktree,
        nears,
        scalartype(operator);
        quadstrat=nearinteractionquadstrat,
        multithreading=multithreading,
    )

    #rand =

    setup(
        testtree,
        trialtree,
        farassembler,
        fars,
        testcompressor,
        scalartype(operator);
        multithreading=multithreading,
        maxrank=maxrank,
        tol=tol,
    )

    @views farblkassembler = BEAST.blockassembler(
        operator, testspace, trialspace; quadstrat=momentquadstrat
    )
    @views function farassembler(Z, tdata, sdata)
        @views store(v, m, n) = (Z[m, n] += v)
        return farblkassembler(tdata, sdata, store)
    end

    nestedtestbases, testtransfermatrices, testpivots = compress_testtree(
        testtree,
        trialtree,
        farassembler,
        fars,
        testcompressor,
        scalartype(operator);
        multithreading=multithreading,
        maxrank=maxrank,
        tol=tol,
    )
    nestedtrialbases, trialtransfermatrices, trialpivots = compress_trialtree(
        testtree,
        trialtree,
        farassembler,
        fars,
        trialcompressor,
        scalartype(operator);
        multithreading=multithreading,
        maxrank=maxrank,
        tol=tol,
    )

    couplingmatrices = assemble_couplingmatrices(
        farassembler,
        scalartype(operator),
        fars,
        testpivots,
        trialpivots;
        multithreading=multithreading,
    )

    return PetrovGalerkinNCA{scalartype(operator)}(
        blktree,
        nearinteractions,
        nestedtestbases,
        nestedtrialbases,
        testtransfermatrices,
        trialtransfermatrices,
        couplingmatrices,
        fars,
        (testtree.num_elements, trialtree.num_elements),
        multithreading,
    )
end

function assemble(operator, testspace, trialspace; kwargs...)
    return PetrovGalerkinNCA(operator, testspace, trialspace; kwargs...)
end

##
