struct PetrovGalerkinNCA{
    T,
    TreeType,
    NearInteractionType,
    NestedBasesDict,
    TransferMatrixType,
    CouplingMatrixType,
    FarInteractionType,
} <: LinearMaps.LinearMap{T}
    tree::TreeType
    nearinteractions::NearInteractionType
    nestedtestbases::NestedBasesDict
    nestedtrialbases::NestedBasesDict
    testtransfermatrices::TransferMatrixType
    trialtransfermatrices::TransferMatrixType
    couplingmatrices::CouplingMatrixType
    fars::FarInteractionType
    dim::Tuple{Int,Int}
    ismultithreaded::Bool

    function PetrovGalerkinNCA{T}(
        tree,
        nearinteractions,
        nestedtestbases,
        nestedtrialbases,
        testtransfermatrices,
        trialtransfermatrices,
        couplingmatrices,
        fars,
        dim,
        ismultithreaded,
    ) where {T}
        return new{
            T,
            typeof(tree),
            typeof(nearinteractions),
            typeof(nestedtestbases),
            typeof(testtransfermatrices),
            typeof(couplingmatrices),
            typeof(fars),
        }(
            tree,
            nearinteractions,
            nestedtestbases,
            nestedtrialbases,
            testtransfermatrices,
            trialtransfermatrices,
            couplingmatrices,
            fars,
            dim,
            ismultithreaded,
        )
    end
end

function PetrovGalerkinNCA(
    operator,
    testspace,
    trialspace;
    testtree=create_tree(testspace.pos, KMeansTreeOptions(; nmin=50, maxlevel=50)),
    trialtree=create_tree(trialspace.pos, KMeansTreeOptions(; nmin=50, maxlevel=50)),
    nearinteractionquadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
    momentquadstrat=BEAST.DoubleNumQStrat(2, 3),
    testcompressor=TopDownCompressor(),
    trialcompressor=TopDownCompressor(),
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

    @views farblkassembler = BEAST.blockassembler(
        operator, testspace, trialspace; quadstrat=momentquadstrat
    )
    @views function farassembler(Z, tdata, sdata)
        @views store(v, m, n) = (Z[m, n] += v)
        return farblkassembler(tdata, sdata, store)
    end
    fartime = @elapsed begin
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
    end

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
