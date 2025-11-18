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
    ntasks::Int

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
        ntasks,
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
            ntasks,
        )
    end
end

struct IsLowFrequencyFunctor{F}
    k::F
end

function islf(k::F) where {F}
    return IsLowFrequencyFunctor{F}(k)
end

function (islf::IsLowFrequencyFunctor{F})(tree::TwoNTree, level::Int) where {F}
    return 2 * sqrt(3) * H2Trees.halfsize(tree) * islf.k / (2.0^(level - 1)) <= 1
end

function (islf::IsLowFrequencyFunctor{F})(hs::F) where {F}
    return 2 * sqrt(3) * hs * islf.k <= 1
end

struct IsNearFunctor{F}
    k::F
    ηₗ::F
    ηₕ::F
    islf::IsLowFrequencyFunctor{F}
end

function isnear(k::F; ηₗ::F=1.0, ηₕ::F=5.0, islf=islf(k)) where {F}
    return IsNearFunctor{F}(k, ηₗ, ηₕ, islf)
end

function (isnear::IsNearFunctor{F})(
    treea::TwoNTree, treeb::TwoNTree, nodea::Int, nodeb::Int
) where {F}
    ths = H2Trees.halfsize(treea, nodea) * sqrt(3)
    shs = H2Trees.halfsize(treeb, nodeb) * sqrt(3)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)
    if isnear.islf(min(ths, shs))
        (2 * max(ths, shs) <= isnear.ηₗ * max(dist, 0.0)) ? (return false) : (return true)
    else
        if (4 * isnear.k * max(ths^2, shs^2) <= isnear.ηₕ * max(dist, 0.0))
            (return false)
        else
            (return true)
        end
    end
end

function maxlevel(tree::TwoNTree, islf::IsLowFrequencyFunctor{F}) where {F}
    level = 0
    while !islf(tree, level)
        level += 1
    end
    return level
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
    ntasks=Threads.nthreads(),
    isnear=isnear(wavenumber(operator)),
    islf=islf(wavenumber(operator)),
    maxrank=40,
)

    # near interactions
    nearmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; quadstrat=nearquadstrat
    )
    values, nearvalues = H2Trees.nearinteractions(
        tree; isnear=isnear, extractselfvalues=false
    )
    println("nearinteractions")
    blocks = Vector{Matrix{eltype(nearmatrix)}}(undef, length(values))
    @time @tasks for i in eachindex(values)
        @set ntasks = ntasks
        blk = zeros(eltype(nearmatrix), length(values[i]), length(nearvalues[i]))
        nearmatrix(blk, values[i], nearvalues[i])
        blocks[i] = blk
    end
    nearinteractions = BlockSparseMatrix(blocks, values, nearvalues, size(nearmatrix))

    farmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; quadstrat=farquadstrat
    )
    dtree = 𝒟tree(H2Trees.halfsize(tree.testcluster), maxlevel(testtree(tree), islf))
    Ft, eₜ, Fs, eₛ = directionalfarinteractions(tree, dtree; isnear=isnear)
    #println(admissiblelevel(Ft, eₜ, tree))
    tolerance!(testcompressor.lrf, admissiblelevel(Ft, eₜ, tree))
    tolerance!(trialcompressor.lrf, admissiblelevel(Fs, eₛ, tree))

    println("compress_testtree")
    @time nestedtestbases, testtransfermatrices, testpivots = testcompressor(
        farmatrix,
        Ft,
        eₜ,
        tree,
        dtree,
        reverse(testbuffer(testcompressor, farmatrix; maxrank=maxrank, ntasks=ntasks));
        ntasks=ntasks,
        maxrank=maxrank,
    )

    println("compress_trialtree")
    @time nestedtrialbases, trialtransfermatrices, trialpivots = trialcompressor(
        farmatrix,
        Fs,
        eₛ,
        tree,
        dtree,
        trialbuffer(trialcompressor, farmatrix; maxrank=maxrank, ntasks=ntasks);
        ntasks=ntasks,
        maxrank=maxrank,
    )

    println("assemble_couplingmatrices2")
    @time coupling = assemble_couplingmatrices(
        farmatrix, testpivots, trialpivots, tree, dtree; ntasks=ntasks, isnear=isnear
    )
    return PetrovGalerkinWNCA{ComplexF64}(
        tree,
        dtree,
        nearinteractions,
        Int[],
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

    type = eltype(y)
    for s in collect(keys(A.nestedtrialbases))
        res = Vector{type}[]
        for (k, dir) in A.nestedtrialbases[s]
            # use the trial-index vector stored in the direction/basis (dir.σ)
            push!(res, dir.T * x[dir.σ])
        end
        xhat[s] = Dict(keys(A.nestedtrialbases[s]) .=> res)
    end

    for nodes in reverse(A.trialtransfermatrices)
        for (node, data) in nodes
            res = Vector{eltype(y)}[]
            for (dir, transfers) in data
                # transfers.children holds the child indices for the transfer; use those
                xhatdir = transfers.T[1] * xhat[transfers.children[1]][parent(A.dtree, dir)]
                for i in 2:length(transfers.children)
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

    for t in collect(keys(A.nestedtestbases))
        for (dir, basis) in A.nestedtestbases[t]
            # write into the basis' target indices
            y[basis.τ] += basis.T * yhat[t][dir]
        end
    end

    @time y += A.nearinteractions * x

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

    y += adjoint(A.nearinteractions) * x

    return y
end
