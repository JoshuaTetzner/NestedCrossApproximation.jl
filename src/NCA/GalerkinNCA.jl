struct GalerkinNCA{
    T,
    TreeType,
    NearInteractionType,
    MomentCollectionDict,
    I2OTranslatorType,
    TranslatorType,
    FarInteractionType
} <: LinearMaps.LinearMap{T}
    tree::TreeType
    nearinteractions::NearInteractionType
    momentcollection::MomentCollectionDict
    i2otranslator::I2OTranslatorType
    translator::TranslatorType
    fars::FarInteractionType
    dim::Tuple{Int, Int}
    verbose::Bool
    ismultithreaded::Bool

    function GalerkinNCA{T}(
        tree,
        nearinteractions,
        momentcollection,
        i2otranslator,
        translator,
        fars,
        dim,
        verbose,
        ismultithreaded
    ) where T
        return new{
            T,
            typeof(tree),
            typeof(nearinteractions),
            typeof(momentcollection),
            typeof(i2otranslator),
            typeof(translator),
            typeof(fars)
        }(
            tree,
            nearinteractions,
            momentcollection,
            i2otranslator,
            translator,
            fars,
            dim,
            verbose,
            ismultithreaded
        )
    end
end

function GalerkinNCA(
    operator,
    space; 
    tree=create_tree(space.pos, KMeansTreeOptions()),
    nearinteractionquadstrat=BEAST.defaultquadstrat(operator, space, space),
    momentquadstrat=BEAST.DoubleNumQStrat(2, 3),
    compressor=FastBEAST.ACAOptions(; tol=1e-3),
    multithreading=true,
    verbose=false,
    η=1.0
)
    
    blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
    nears, fars = FastBEAST.computeinteractions(blktree,  η=η)
   # println("Nears")
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

    test_fars = row_pivot_selection(
        tree,
        tree,
        fars,
        farassembler,
        scalartype(operator);
        compressor=compressor,
        verbose=verbose,
        multithreading=multithreading
    )
    
    momentcollection, translator = build_test_bases(
        tree, test_fars, scalartype(operator), verbose=verbose, multithreading=multithreading
    )
    
   

    i2otranslator = assemble_couplingmatrices(
        farassembler,
        scalartype(operator), 
        fars, 
        test_fars,
        compressor; 
        multithreading=multithreading, 
        verbose=verbose
    )

    return GalerkinNCA{scalartype(operator)}(
        blktree,
        nearinteractions,
        momentcollection,
        i2otranslator,
        translator,
        fars,
        (tree.num_elements, tree.num_elements),
        verbose,
        multithreading
    )
end

function assemble(operator, space; kwargs...)
    return GalerkinNCA(operator, space; kwargs...)
end