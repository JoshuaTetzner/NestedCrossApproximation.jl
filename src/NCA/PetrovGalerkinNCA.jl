struct PetrovGalerkinNCA{
    T,
    TreeType,
    NearInteractionType,
    MomentCollectionDict,
    I2ITranslatorType,
    I2OTranslatorType,
    O2OTranslatorType,
    FarInteractionType
} <: LinearMaps.LinearMap{T}
    tree::TreeType
    nearinteractions::NearInteractionType
    testmomentcollection::MomentCollectionDict
    trialmomentcollection::MomentCollectionDict
    i2itranslator::I2ITranslatorType
    i2otranslator::I2OTranslatorType
    o2otranslator::O2OTranslatorType
    fars::FarInteractionType
    dim::Tuple{Int, Int}
    verbose::Bool
    ismultithreaded::Bool

    function PetrovGalerkinNCA{T}(
        tree,
        nearinteractions,
        testmomentcollection,
        trialmomentcollection,
        i2itranslator,
        i2otranslator,
        o2otranslator,
        fars,
        dim,
        verbose,
        ismultithreaded,
    ) where T
        return new{
            T,
            typeof(tree),
            typeof(nearinteractions),
            typeof(testmomentcollection),
            typeof(i2itranslator),
            typeof(i2otranslator),
            typeof(o2otranslator),
            typeof(fars)
        }(
            tree,
            nearinteractions,
            testmomentcollection,
            trialmomentcollection,
            i2itranslator,
            i2otranslator,
            o2otranslator,
            fars,
            dim,
            verbose,
            ismultithreaded,
        )

    end

end

function PetrovGalerkinNCA(
    operator,
    testspace, 
    trialspace;
    testtree=create_tree(testspace.pos, KMeansTreeOptions()),
    trialtree=create_tree(trialspace.pos, KMeansTreeOptions()),
    nearinteractionquadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
    momentquadstrat=BEAST.DoubleNumQStrat(2, 2),
    compressor=FastBEAST.ACAOptions(; tol=1e-4),
    multithreading=true,
    verbose=true,
    η
)
    blktree = ClusterTrees.BlockTrees.BlockTree(testtree, trialtree)
    nears, fars = computeinteractions(blktree, η=η)
    
    nearinteractions = FastBEAST.assemble(
        operator,
        testspace,
        trialspace,
        blktree,
        nears,
        scalartype(operator);
        quadstrat=nearinteractionquadstrat,
        verbose=verbose,
        multithreading=multithreading
    ) 

    @views farblkassembler = BEAST.blockassembler(
        operator, testspace, trialspace, quadstrat=momentquadstrat
        )
    @views function farassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        farblkassembler(tdata,sdata,store)
    end
    println("Nearinteractions")
      

    test_fars = row_pivot_selection(
        testtree,
        trialtree,
        fars,
        farassembler,
        scalartype(operator);
        compressor=compressor,
        verbose=verbose,
        multithreading=multithreading
    )

    trial_fars = column_pivot_selection(
        testtree,
        trialtree,
        fars,
        farassembler,
        scalartype(operator);
        compressor=compressor,
        verbose=verbose,
        multithreading=multithreading
    )

    testmomentcollection, o2otranslator = build_test_bases(
        testtree, test_fars, scalartype(operator), verbose=verbose, multithreading=multithreading
    )
    trialmomentcollection, i2itranslator = build_trial_bases(
        trialtree, trial_fars, scalartype(operator), verbose=verbose, multithreading=multithreading
    )

    i2otranslator = assemble_couplingmatrices(
        farassembler,
        scalartype(operator), 
        fars, 
        test_fars, 
        trial_fars,
        compressor; 
        multithreading=multithreading, 
        verbose=verbose
    )

    return PetrovGalerkinNCA{scalartype(operator)}(
        blktree,
        nearinteractions,
        testmomentcollection,
        trialmomentcollection,
        i2itranslator,
        i2otranslator,
        o2otranslator,
        fars,
        (testtree.num_elements, trialtree.num_elements),
        verbose,
        multithreading
    )

end

function assemble(operator, testspace, trialspace; kwargs...)
    return PetrovGalerkinNCA(operator, testspace, trialspace; kwargs...)
end
