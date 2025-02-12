struct GalerkinNCA{
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
    nestedbases::NestedBasesDict
    transfermatrices::TransferMatrixType
    couplingmatrices::CouplingMatrixType
    fars::FarInteractionType
    dim::Tuple{Int,Int}
    ismultithreaded::Bool

    function GalerkinNCA{T}(
        tree,
        nearinteractions,
        nestedbases,
        transfermatrices,
        couplingmatrices,
        fars,
        dim,
        ismultithreaded,
    ) where {T}
        return new{
            T,
            typeof(tree),
            typeof(nearinteractions),
            typeof(nestedbases),
            typeof(transfermatrices),
            typeof(couplingmatrices),
            typeof(fars),
        }(
            tree,
            nearinteractions,
            nestedbases,
            transfermatrices,
            couplingmatrices,
            fars,
            dim,
            ismultithreaded,
        )
    end
end

function GalerkinNCA(
    operator,
    space;
    tree=create_tree(space.pos, KMeansTreeOptions(; nmin=50)),
    nearinteractionquadstrat=BEAST.defaultquadstrat(operator, space, space),
    momentquadstrat=BEAST.DoubleNumQStrat(2, 3),
    compressor=TopDownCompressor(),
    multithreading=true,
    maxrank=40, #Should be moved to the compressor
    tol=1e-4, #global
    η=1.0, #global
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
    println("incompression")
    nestedbases, transfermatrices, pivots = compress_testtree(
        tree,
        tree,
        farassembler,
        fars,
        compressor,
        scalartype(operator);
        multithreading=multithreading,
        maxrank=maxrank,
        tol=tol,
    )
    println("outcompression")
    couplingmatrices = assemble_couplingmatrices(
        farassembler, scalartype(operator), fars, pivots; multithreading=multithreading
    )

    return GalerkinNCA{scalartype(operator)}(
        blktree,
        nearinteractions,
        nestedbases,
        transfermatrices,
        couplingmatrices,
        fars,
        (tree.num_elements, tree.num_elements),
        multithreading,
    )
end

function assemble(operator, space; kwargs...)
    return GalerkinNCA(operator, space; kwargs...)
end
