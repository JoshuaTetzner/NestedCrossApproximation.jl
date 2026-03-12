struct PetrovGalerkinNCA{
    T,TreeType,NearInteractionType,NestedBasesDict,TransferMatrixType,CouplingMatrixType
} <: LinearMaps.LinearMap{T}
    tree::TreeType
    nearinteractions::NearInteractionType
    nestedtestbases::NestedBasesDict
    nestedtrialbases::NestedBasesDict
    testtransfermatrices::TransferMatrixType
    trialtransfermatrices::TransferMatrixType
    couplingmatrices::CouplingMatrixType
    dim::Tuple{Int,Int}
    ntasks::Int

    function PetrovGalerkinNCA{T}(
        tree,
        nearinteractions,
        nestedtestbases,
        nestedtrialbases,
        testtransfermatrices,
        trialtransfermatrices,
        couplingmatrices,
        dim,
        ntasks,
    ) where {T}
        return new{
            T,
            typeof(tree),
            typeof(nearinteractions),
            typeof(nestedtestbases),
            typeof(testtransfermatrices),
            typeof(couplingmatrices),
        }(
            tree,
            nearinteractions,
            nestedtestbases,
            nestedtrialbases,
            testtransfermatrices,
            trialtransfermatrices,
            couplingmatrices,
            dim,
            ntasks,
        )
    end
end

function PetrovGalerkinNCA(
    operator,
    testspace,
    trialspace,
    tree;
    farquadstrat=defaultfarquadstrat(operator, testspace, trialspace),
    nearquadstrat=defaultnearquadstrat(operator, testspace, trialspace),
    testcompressor=TopDownCompressor(),
    trialcompressor=TopDownCompressor(),
    ntasks=Threads.nthreads(),
    isnear=isnear(),
    maxrank=40,
)

    #near interactions
    nearmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; quadstrat=nearquadstrat
    )
    values, nearvalues = nearinteractions(tree; isnear=isnear)
    println("nearinteractions")
    blocks = zeros.(eltype(nearmatrix), length.(values), length.(nearvalues))
    @time @tasks for i in eachindex(blocks)
        @set ntasks = ntasks
        nearmatrix(blocks[i], values[i], nearvalues[i])
    end
    nears = BlockSparseMatrix(blocks, values, nearvalues, size(nearmatrix))

    # far interactions
    #testdfars, trialfars = farinteractions(tree; isnear=isnear)

    farmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; quadstrat=farquadstrat
    )
    testfars, trialfars = farinteractions(tree; isnear=isnear)
    println("compress_testtree")
    nestedtestbases, testtransfermatrices, testpivots = testcompressor(
        farmatrix,
        #testdfars,
        tree,
        reverse(testbuffer(testcompressor, farmatrix; maxrank=maxrank, ntasks=ntasks));
        isnear=isnear,
        ntasks=ntasks,
    )

    println("compress_trialtree")
    nestedtrialbases, trialtransfermatrices, trialpivots = trialcompressor(
        farmatrix,
        #trialfars,
        tree,
        trialbuffer(trialcompressor, farmatrix; maxrank=maxrank, ntasks=ntasks);
        isnear=isnear,
        ntasks=ntasks,
    )

    println("coupling")
    couplingmatrices = assemble_couplingmatrices(
        farmatrix, tree, testpivots, trialpivots; isnear=isnear, ntasks=ntasks
    )

    return PetrovGalerkinNCA{eltype(nearmatrix)}(
        tree,
        nearinteractions,
        Dict(
            i => nestedtestbases[i] for
            i in eachindex(nestedtestbases) if isassigned(nestedtestbases, i)
        ),
        Dict(
            i => nestedtrialbases[i] for
            i in eachindex(nestedtrialbases) if isassigned(nestedtrialbases, i)
        ),
        Dict(
            i => testtransfermatrices[i] for
            i in eachindex(testtransfermatrices) if isassigned(testtransfermatrices, i)
        ),
        Dict(
            i => trialtransfermatrices[i] for
            i in eachindex(trialtransfermatrices) if isassigned(trialtransfermatrices, i)
        ),
        Dict(
            i => couplingmatrices[i] for
            i in eachindex(couplingmatrices) if isassigned(couplingmatrices, i)
        ),
        size(farmatrix),
        ntasks,
    )
end
