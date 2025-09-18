using LinearMaps

struct PetrovGalerkinWNCA{
    T,
    TreeType,
    DirectionTreeType,
    NearInteractionType,
    LFInteractionType,
    NestedBasesDict,
    TransferMatrixType,
    CouplingMatrixType,
} <: LinearMaps.LinearMap{T}
    tree::TreeType
    dtree::DirectionTreeType
    nearinteractions::NearInteractionType
    lowfrequencyinteractions::LFInteractionType
    nestedtestbases::NestedBasesDict
    nestedtrialbases::NestedBasesDict
    testtransfermatrices::TransferMatrixType
    trialtransfermatrices::TransferMatrixType
    couplingmatrices::CouplingMatrixType
    dim::Tuple{Int,Int}
    ismultithreaded::Int

    function PetrovGalerkinWNCA{T}(
        tree,
        dtree,
        nearinteractions,
        lowfrequencyinteractions,
        nestedtestbases,
        nestedtrialbases,
        testtransfermatrices,
        trialtransfermatrices,
        couplingmatrices,
        dim,
        ismultithreaded,
    ) where {T}
        return new{
            T,
            typeof(tree),
            typeof(dtree),
            typeof(nearinteractions),
            typeof(lowfrequencyinteractions),
            typeof(nestedtestbases),
            typeof(testtransfermatrices),
            typeof(couplingmatrices),
        }(
            tree,
            dtree,
            nearinteractions,
            lowfrequencyinteractions,
            nestedtestbases,
            nestedtrialbases,
            testtransfermatrices,
            trialtransfermatrices,
            couplingmatrices,
            dim,
            ismultithreaded,
        )
    end
end

function isnear(k, treea, treeb, nodea, nodeb; ηₗ=1.0, ηₕ=4.0)
    ths = H2Trees.halfsize(treea, nodea) * sqrt(3)
    shs = H2Trees.halfsize(treeb, nodeb) * sqrt(3)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)
    if k / pi * 4 * min(ths, shs) <= 1
        (2 * max(ths, shs) <= ηₗ * max(dist, 0.0)) ? (return false) : (return true)
    else
        (4 * k * max(ths^2, shs^2) <= ηₕ * max(dist, 0.0)) ? (return false) : (return true)
    end
end

function islf(k, tree, level)
    return k / pi * 4 * H2Trees.halfsize(tree.testcluster) / 2^(level - 1) <= 1
end

function islf(k)
    lf(tree, level) = islf(k, tree, level)
    return lf
end
wavenumber(operator) = operator.wavenumber
function PetrovGalerkinWNCA(
    operator,
    testspace,
    trialspace,
    tree;
    farquadstrat=defaultfarquadstrat(operator, testspace, trialspace),
    nearquadstrat=defaultnearquadstrat(operator, testspace, trialspace),
    testcompressor=TopDownCompressor(),
    trialcompressor=TopDownCompressor(),
    lfcompressor=AdaptiveCrossApproximation.ACA(),
    ntasks=Threads.nthreads(),
    isnear=H2Trees.isnear,
    islf=islf(wavenumber(operator)),
    maxrank=40,
)

    # near interactions
    nearmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; quadstrat=nearquadstrat
    )
    farmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; quadstrat=farquadstrat
    )
    dtree = 𝒟tree(H2Trees.halfsize(tree.testcluster), imag(operator.gamma))
    values, nearvalues, fars, dirs, lfvalues, lffarvalues = directionalitneractions(
        tree, dtree, islf, isnear
    )

    blocks = Vector{Matrix{eltype(nearmatrix)}}(undef, length(values))
    println("nears")
    @tasks for i in eachindex(values)
        @set ntasks = ntasks
        blk = zeros(eltype(nearmatrix), length(values[i]), length(nearvalues[i]))
        nearmatrix(blk, values[i], nearvalues[i])
        blocks[i] = blk
    end
    nearinteractions = BlockSparseMatrix(blocks, values, nearvalues, size(nearmatrix))

    println("lfs")

    lfblocks = blockcompressor(
        farmatrix, lfvalues, lffarvalues, lfcompressor; maxrank=maxrank, ntasks=ntasks
    )

    println("compress_testtree")
    nestedtestbases, testtransfermatrices, testpivots = testcompressor(
        farmatrix,
        dtree,
        tree,
        fars,
        dirs,
        reverse(testbuffer(testcompressor, farmatrix; maxrank=maxrank, ntasks=ntasks));
        ntasks=ntasks,
        maxrank=maxrank,
    )
    println("compress_trialtree")
    nestedtrialbases, trialtransfermatrices, trialpivots = trialcompressor(
        farmatrix,
        dtree,
        tree,
        fars,
        dirs,
        trialbuffer(trialcompressor, farmatrix; maxrank=maxrank, ntasks=ntasks);
        ntasks=ntasks,
        maxrank=maxrank,
    )
    @time coupling = assemble_couplingmatrices(
        farmatrix, testpivots, trialpivots, fars, dirs; ntasks=ntasks
    )
    return PetrovGalerkinWNCA{ComplexF64}(
        tree,
        dtree,
        nearinteractions,
        lfblocks,
        nestedtestbases,
        nestedtrialbases,
        testtransfermatrices,
        trialtransfermatrices,
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
    y::AbstractVecOrMat, A::PetrovGalerkinWNCA, x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))

    xhat = Vector{Dict{Int,Vector{eltype(y)}}}(undef, length(A.tree.trialcluster.nodes))
    yhat = Vector{Dict{Int,Vector{eltype(y)}}}(undef, length(A.tree.testcluster.nodes))

    for (idx, moment) in A.nestedtrialbases
        res = Vector{eltype(y)}[]
        for (k, dir) in moment
            push!(res, dir.T * x[dir.σ])
        end
        xhat[idx] = Dict(keys(moment) .=> res)
    end

    for nodes in reverse(A.trialtransfermatrices)
        for (node, data) in nodes
            res = Vector{eltype(y)}[]
            for (dir, transfers) in data
                #childs = transfers.T.children#collect(H2Trees.children(A.tree.trialcluster, node))
                xhatdir = transfers.T[1] * xhat[transfers.children[1]][parent(A.dtree, dir)]
                for i in 2:length(transfers.children)#enumerate(transfers[2:end])
                    xhatdir +=
                        transfers.T[i] * xhat[transfers.children[i]][parent(A.dtree, dir)]
                end
                push!(res, xhatdir)
            end
            xhat[node] = Dict(keys(data) .=> res)
        end
    end

    for lrb in A.couplingmatrices
        if isassigned(yhat, lrb.row_basis)
            if haskey(yhat[lrb.row_basis], lrb.dir)
                yhat[lrb.row_basis][lrb.dir] += lrb.Z * xhat[lrb.col_basis][lrb.dir]
            else
                yhat[lrb.row_basis][lrb.dir] = lrb.Z * xhat[lrb.col_basis][lrb.dir]
            end
        else
            yhat[lrb.row_basis] = Dict(lrb.dir => lrb.Z * xhat[lrb.col_basis][lrb.dir])
        end
    end

    for nodes in A.testtransfermatrices
        for (node, data) in nodes
            childs = collect(H2Trees.children(A.tree.testcluster, node))

            for (dir, transfers) in data
                childdir = parent(A.dtree, dir)
                for (i, child) in enumerate(transfers.children)
                    if isassigned(yhat, child)
                        if haskey(yhat[child], childdir)
                            yhat[child][childdir] += transfers.T[i] * yhat[node][dir]
                        else
                            yhat[child][childdir] = transfers.T[i] * yhat[node][dir]
                        end
                    else
                        yhat[child] = Dict(childdir => transfers.T[i] * yhat[node][dir])
                    end
                end
            end
        end
    end

    for (idx, moment) in A.nestedtestbases
        for (dir, basis) in moment
            y[basis.τ] += basis.T * yhat[idx][dir]
        end
    end

    for lrb in A.lowfrequencyinteractions
        y[lrb.τ] += lrb.M * x[lrb.σ]
    end

    y += A.nearinteractions * x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:PetrovGalerkinWNCA},
    x::AbstractVector,
)
    A = At.lmap
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))

    xhat = Vector{Dict{Int,Vector{eltype(y)}}}(undef, length(A.tree.trialcluster.nodes))
    yhat = Vector{Dict{Int,Vector{eltype(y)}}}(undef, length(A.tree.testcluster.nodes))

    for (idx, moment) in A.nestedtestbases
        res = Vector{eltype(y)}[]
        for (k, dir) in moment
            push!(res, transpose(dir.T) * x[dir.τ])
        end
        xhat[idx] = Dict(keys(moment) .=> res)
    end

    for nodes in reverse(A.testtransfermatrices)
        for (node, data) in nodes
            res = Vector{eltype(y)}[]
            for (dir, transfers) in data
                #childs = collect(H2Trees.children(A.tree.testcluster, node))
                xhatdir =
                    transpose(transfers.T[1]) *
                    xhat[transfers.children[1]][parent(A.dtree, dir)]
                for (i, transfer) in enumerate(transfers.T[2:end])
                    xhatdir +=
                        transpose(transfer) *
                        xhat[transfers.children[i + 1]][parent(A.dtree, dir)]
                end
                push!(res, xhatdir)
            end
            xhat[node] = Dict(keys(data) .=> res)
        end
    end

    for lrb in A.couplingmatrices
        if isassigned(yhat, lrb.col_basis)
            if haskey(yhat[lrb.col_basis], lrb.dir)
                yhat[lrb.col_basis][lrb.dir] +=
                    transpose(lrb.Z) * xhat[lrb.row_basis][lrb.dir]
            else
                yhat[lrb.col_basis][lrb.dir] =
                    transpose(lrb.Z) * xhat[lrb.row_basis][lrb.dir]
            end
        else
            yhat[lrb.col_basis] = Dict(
                lrb.dir => transpose(lrb.Z) * xhat[lrb.row_basis][lrb.dir]
            )
        end
    end

    for nodes in A.trialtransfermatrices
        for (node, data) in nodes
            #childs = collect(H2Trees.children(A.tree.trial_cluster, node))
            for (dir, transfers) in data
                childdir = parent(A.dtree, dir)
                for (i, child) in enumerate(transfers.children)
                    if isassigned(yhat, child)
                        if haskey(yhat[child], childdir)
                            yhat[child][childdir] +=
                                transpose(transfers.T[i]) * yhat[node][dir]
                        else
                            yhat[child][childdir] =
                                transpose(transfers.T[i]) * yhat[node][dir]
                        end
                    else
                        yhat[child] = Dict(
                            childdir => transpose(transfers.T[i]) * yhat[node][dir]
                        )
                    end
                end
            end
        end
    end

    for (idx, moment) in A.nestedtrialbases
        for (dir, basis) in moment
            y[basis.σ] += transpose(basis.T) * yhat[idx][dir]
        end
    end

    for lrb in A.lowfrequencyinteractions
        y[lrb.σ] += transpose(lrb.M) * x[lrb.τ]
    end

    y += transpose(A.nearinteractions) * x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:PetrovGalerkinWNCA},
    x::AbstractVector,
)
    A = At.lmap
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))

    xhat = Vector{Dict{Int,Vector{eltype(y)}}}(undef, length(A.tree.trialcluster.nodes))
    yhat = Vector{Dict{Int,Vector{eltype(y)}}}(undef, length(A.tree.testcluster.nodes))

    for (idx, moment) in A.nestedtestbases
        res = Vector{eltype(y)}[]
        for (k, dir) in moment
            push!(res, adjoint(dir.T) * x[dir.τ])
        end
        xhat[idx] = Dict(keys(moment) .=> res)
    end

    for nodes in reverse(A.testtransfermatrices)
        for (node, data) in nodes
            res = Vector{eltype(y)}[]
            for (dir, transfers) in data
                #childs = collect(H2Trees.children(A.tree.testcluster, node))
                xhatdir =
                    adjoint(transfers.T[1]) *
                    xhat[transfers.children[1]][parent(A.dtree, dir)]
                for (i, transfer) in enumerate(transfers.T[2:end])
                    xhatdir +=
                        adjoint(transfer) *
                        xhat[transfers.children[i + 1]][parent(A.dtree, dir)]
                end
                push!(res, xhatdir)
            end
            xhat[node] = Dict(keys(data) .=> res)
        end
    end

    for lrb in A.couplingmatrices
        if isassigned(yhat, lrb.col_basis)
            if haskey(yhat[lrb.col_basis], lrb.dir)
                yhat[lrb.col_basis][lrb.dir] +=
                    adjoint(lrb.Z) * xhat[lrb.row_basis][lrb.dir]
            else
                yhat[lrb.col_basis][lrb.dir] = adjoint(lrb.Z) * xhat[lrb.row_basis][lrb.dir]
            end
        else
            yhat[lrb.col_basis] = Dict(
                lrb.dir => adjoint(lrb.Z) * xhat[lrb.row_basis][lrb.dir]
            )
        end
    end

    for nodes in A.trialtransfermatrices
        for (node, data) in nodes
            #childs = collect(H2Trees.children(A.tree.trial_cluster, node))
            for (dir, transfers) in data
                childdir = parent(A.dtree, dir)
                for (i, child) in enumerate(transfers.children)
                    if isassigned(yhat, child)
                        if haskey(yhat[child], childdir)
                            yhat[child][childdir] +=
                                adjoint(transfers.T[i]) * yhat[node][dir]
                        else
                            yhat[child][childdir] =
                                adjoint(transfers.T[i]) * yhat[node][dir]
                        end
                    else
                        yhat[child] = Dict(
                            childdir => adjoint(transfers.T[i]) * yhat[node][dir]
                        )
                    end
                end
            end
        end
    end

    for (idx, moment) in A.nestedtrialbases
        for (dir, basis) in moment
            y[basis.σ] += adjoint(basis.T) * yhat[idx][dir]
        end
    end

    for lrb in A.lowfrequencyinteractions
        y[lrb.σ] += adjoint(lrb.M) * x[lrb.τ]
    end

    y += adjoint(A.nearinteractions) * x

    return y
end
