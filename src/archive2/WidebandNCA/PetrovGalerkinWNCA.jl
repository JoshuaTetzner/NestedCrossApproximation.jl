struct PetrovGalerkinWNCA{
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

    function PetrovGalerkinWNCA{T}(
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

wavenumber(operator) = operator.wavenumber

function defaultfarquadstrat(operator, testspace, trialspace) end
function defaultnearquadstrat(operator, testspace, trialspace) end

function PetrovGalerkinWNCA(
    operator,
    testspace,
    trialspace,
    tree;
    farquadstrat=defaultfarquadstrat(operator, testspace, trialspace),
    nearquadstrat=defaultnearquadstrat(operator, testspace, trialspace),
    testcompressor=TopDownCompressor(),
    trialcompressor=TopDownCompressor(),
    ntasks=Threads.nthreads(),
    isnear=isnear(wavenumber(operator)),
    maxrank=40,
)

    # near interactions
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
    nears = BlockSparseMatrix(
        blocks, values, nearvalues, size(nearmatrix); scheduler=DynamicScheduler()
    )

    farmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; quadstrat=farquadstrat
    )
    testfardata = NestedCrossApproximation.directionaltestfars(
        tree; isnear=isnear, ntasks=ntasks
    )
    trialfardata = NestedCrossApproximation.directionaltrialfars(
        tree; isnear=isnear, ntasks=ntasks
    )

    tolerance!(testcompressor.lrf, admissiblelevel(testtree(tree), testfardata))
    tolerance!(trialcompressor.lrf, admissiblelevel(trialtree(tree), trialfardata))
    println(testcompressor.lrf.convergence.estimator.tol)
    println(trialcompressor.lrf.convergence.estimator.tol)

    println("compress_testtree")
    @time nestedtestbases, testtransfermatrices, testpivots = testcompressor(
        farmatrix,
        testfardata,
        tree,
        reverse(testbuffer(testcompressor, farmatrix; maxrank=maxrank, ntasks=ntasks));
        islf=isnear.islf,
        ntasks=ntasks,
        maxrank=maxrank,
    )
    println("compress_trialtree")
    @time nestedtrialbases, trialtransfermatrices, trialpivots = trialcompressor(
        farmatrix,
        trialfardata,
        tree,
        trialbuffer(trialcompressor, farmatrix; maxrank=maxrank, ntasks=ntasks);
        islf=isnear.islf,
        ntasks=ntasks,
        maxrank=maxrank,
    )
    println("couplingmatrices")
    @time coupling = assemble_couplingmatrices(
        farmatrix, testpivots, trialpivots, testfardata, trialfardata; ntasks=ntasks
    )

    return PetrovGalerkinWNCA{ComplexF64}(
        tree,
        nears,
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
        coupling,
        size(farmatrix),
        ntasks,
    )
end

function Base.size(A::PetrovGalerkinWNCA, dim=nothing)
    if dim === nothing
        return (A.dim[1], A.dim[2])
    elseif dim == 1
        return A.dim[1]
    elseif dim == 2
        return A.dim[2]
    else
        error("dim must be either 1 or 2")
    end
end

function Base.size(A::Adjoint{T}, dim=nothing) where {T<:PetrovGalerkinWNCA}
    if dim === nothing
        return reverse(A.dim[1], A.dim[2])
    elseif dim == 1
        return h2mat.lmap.dim[2]
    elseif dim == 2
        return h2mat.lmap.dim[1]
    else
        error("dim must be either 1 or 2")
    end
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::PetrovGalerkinWNCA{K}, x::AbstractVector
) where {K}
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(K))

    xhat = Vector{Dict{Int,Vector{K}}}(undef, numberofnodes(trialtree(A.tree)))
    yhat = Vector{Dict{Int,Vector{K}}}(undef, numberofnodes(testtree(A.tree)))

    mul!(y, A.nearinteractions, x)

    @tasks for (s, dirnbs) in collect(A.nestedtrialbases)
        @set ntasks = A.ntasks
        res = Vector{K}[]
        for (dir, nb) in dirnbs
            # use the trial-index vector stored in the direction/basis (dir.σ)
            push!(res, nb * x[H2Trees.values(trialtree(A.tree), s)])
        end
        xhat[s] = Dict(keys(A.nestedtrialbases[s]) .=> res)
    end

    for level in reverse(levels(trialtree(A.tree)))
        @tasks for node in collect(H2Trees.LevelIterator(trialtree(A.tree), level))
            @set ntasks = A.ntasks
            if haskey(A.trialtransfermatrices, node)
                res = Vector{K}[]
                for dtmats in values(A.trialtransfermatrices[node])
                    chds = collect(ChildIterator(trialtree(A.tree), node))
                    push!(
                        res,
                        mapreduce(+, enumerate(chds)) do (cidx, chd)
                            dtmats[cidx][2] * xhat[chd][dtmats[cidx][1]]
                        end,
                    )
                end
                xhat[node] = Dict(keys(A.trialtransfermatrices[node]) .=> res)
            end
        end
    end

    @tasks for t in eachindex(A.couplingmatrices)
        @set ntasks = A.ntasks
        if A.couplingmatrices[t] != Dict()
            dirs = Int[]
            res = Vector{K}[]
            for (st, dircmat) in A.couplingmatrices[t]
                if dircmat[1][1] ∈ dirs
                    idx = findfirst(x -> x == dircmat[1][1], dirs)
                    res[idx] += dircmat[2] * xhat[st[2]][dircmat[1][2]]
                else
                    push!(dirs, dircmat[1][1])
                    push!(res, dircmat[2] * xhat[st[2]][dircmat[1][2]])
                end
            end
            yhat[t] = Dict(dirs .=> res)
        end
    end

    for level in levels(testtree(A.tree))
        @tasks for node in collect(H2Trees.LevelIterator(testtree(A.tree), level))
            @set ntasks = A.ntasks
            if haskey(A.testtransfermatrices, node)
                for (dir, dtmats) in A.testtransfermatrices[node]
                    chds = collect(ChildIterator(testtree(A.tree), node))
                    for (cidx, chd) in enumerate(chds)
                        if isassigned(yhat, chd)
                            if haskey(yhat[chd], dtmats[cidx][1])
                                yhat[chd][dtmats[cidx][1]] +=
                                    dtmats[cidx][2] * yhat[node][dir]
                            else
                                yhat[chd][dtmats[cidx][1]] =
                                    dtmats[cidx][2] * yhat[node][dir]
                            end
                        else
                            yhat[chd] = Dict(
                                dtmats[cidx][1] => dtmats[cidx][2] * yhat[node][dir]
                            )
                        end
                    end
                end
            end
        end
    end
    #=
    @tasks for (s, dirnbs) in collect(A.nestedtrialbases)
        @set ntasks = A.ntasks
        res = Vector{K}[]
        for nb in values(dirnbs)
            # use the trial-index vector stored in the direction/basis (dir.σ)
            push!(res, nb * x[H2Trees.values(trialtree(A.tree), s)])
        end
        xhat[s] = Dict(keys(A.nestedtrialbases[s]) .=> res)
    end=#

    for (node, dnbs) in collect(A.nestedtestbases)
        # ntasks = A.ntasks
        for (dir, nb) in dnbs
            # write into the basis' target indices
            y[H2Trees.values(testtree(A.tree), node)] += nb * yhat[node][dir]
        end
    end

    return y
end

## Only symmetric case
@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:PetrovGalerkinWNCA},
    x::AbstractVector,
)
    return mul!(y, At.lmap, x)
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:PetrovGalerkinWNCA{K}},
    x::AbstractVector,
) where {K}
    A = At.lmap
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(K))
    xhat = Vector{Dict{Int,Vector{K}}}(undef, numberofnodes(trialtree(A.tree)))
    yhat = Vector{Dict{Int,Vector{K}}}(undef, numberofnodes(testtree(A.tree)))
    mul!(y, adjoint(A.nearinteractions), x)

    for (s, dirnbs) in collect(A.nestedtrialbases)
        #@set ntasks = A.ntasks
        res = Vector{K}[]
        for (dir, nb) in dirnbs
            # use the trial-index vector stored in the direction/basis (dir.σ)
            push!(res, conj.(nb) * x[H2Trees.values(trialtree(A.tree), s)])
        end
        xhat[s] = Dict(keys(A.nestedtrialbases[s]) .=> res)
    end

    for level in reverse(levels(trialtree(A.tree)))
        for node in collect(H2Trees.LevelIterator(trialtree(A.tree), level))
            if haskey(A.trialtransfermatrices, node)
                res = Vector{K}[]
                for dtmats in values(A.trialtransfermatrices[node])
                    chds = collect(ChildIterator(trialtree(A.tree), node))
                    push!(
                        res,
                        mapreduce(+, enumerate(chds)) do (cidx, chd)
                            conj.(dtmats[cidx][2]) * xhat[chd][dtmats[cidx][1]]
                        end,
                    )
                end
                xhat[node] = Dict(keys(A.trialtransfermatrices[node]) .=> res)
            end
        end
    end

    for t in eachindex(A.couplingmatrices)
        #  @set ntasks = A.ntasks
        if A.couplingmatrices[t] != Dict()
            dirs = Int[]
            res = Vector{K}[]
            for (st, dircmat) in A.couplingmatrices[t]
                if dircmat[1][1] ∈ dirs
                    idx = findfirst(x -> x == dircmat[1][1], dirs)
                    res[idx] += conj.(dircmat[2]) * xhat[st[2]][dircmat[1][2]]
                else
                    push!(dirs, dircmat[1][1])
                    push!(res, conj.(dircmat[2]) * xhat[st[2]][dircmat[1][2]])
                end
            end
            yhat[t] = Dict(dirs .=> res)
        end
    end

    for level in levels(testtree(A.tree))
        for node in collect(H2Trees.LevelIterator(testtree(A.tree), level))
            if haskey(A.testtransfermatrices, node)
                for (dir, dtmats) in A.testtransfermatrices[node]
                    chds = collect(ChildIterator(testtree(A.tree), node))
                    for (cidx, chd) in enumerate(chds)
                        if isassigned(yhat, chd)
                            if haskey(yhat[chd], dtmats[cidx][1])
                                yhat[chd][dtmats[cidx][1]] +=
                                    conj.(dtmats[cidx][2]) * yhat[node][dir]
                            else
                                yhat[chd][dtmats[cidx][1]] =
                                    conj.(dtmats[cidx][2]) * yhat[node][dir]
                            end
                        else
                            yhat[chd] = Dict(
                                dtmats[cidx][1] => conj.(dtmats[cidx][2]) * yhat[node][dir]
                            )
                        end
                    end
                end
            end
        end
    end

    for (s, dirnbs) in collect(A.nestedtrialbases)
        #@set ntasks = A.ntasks
        res = Vector{K}[]
        for nb in values(dirnbs)
            # use the trial-index vector stored in the direction/basis (dir.σ)
            push!(res, conj.(nb) * x[H2Trees.values(trialtree(A.tree), s)])
        end
        xhat[s] = Dict(keys(A.nestedtrialbases[s]) .=> res)
    end

    for (node, dnbs) in collect(A.nestedtestbases)
        for (dir, nb) in dnbs
            # write into the basis' target indices
            y[H2Trees.values(testtree(A.tree), node)] += conj.(nb) * yhat[node][dir]
        end
    end

    return y
end
