using Base.Threads
using FastBEAST
import FastBEAST.NminClusterTrees.NminTree
using ClusterTrees
import ClusterTrees.PointerBasedTrees
using ThreadsX

function deriveinformation(
    tree::NminTree{D},
    hffars::Vector{Vector{Tuple{Int,Int}}},
    dtree::PointerBasedTrees.PointerBasedTree{T},
) where {D,T}
    tfars, sfars = NestedCrossApproximation.sortinteractionsWNCA(
        ClusterTrees.BlockTrees.BlockTree(tree, tree), hffars
    )
    tclink = FastBEAST.cluster_link(tree)
    sclink = tclink
    firstlevel = findfirst(!=([]), hffars)
    lastlevel = findlast(!=([]), hffars)
    depth = lastlevel - firstlevel + 1
    tdirfars = Vector{Dict{Int,Vector{Int}}}(undef, length(tfars))
    sdirfars = Vector{Dict{Int,Vector{Int}}}(undef, length(sfars))

    for (i, levelidx) in enumerate(firstlevel:lastlevel)
        for node in tclink[levelidx]
            directions = zeros(Int, length(tfars[node]))
            ct = tree.nodes[node].node.data.ct
            for (faridx, far) in enumerate(tfars[node])
                directions[faridx] = NestedCrossApproximation.find_direction(
                    dtree, tree.nodes[far].node.data.ct - ct, depth - (i - 1)
                )
            end

            Fe = [
                tfars[node][findall(x -> x == dir, directions)] for
                dir in unique(directions)
            ]
            directions = unique!(directions)
            if i > 1
                for dir in keys(tdirfars[ClusterTrees.parent(tree, node)])
                    if !(ClusterTrees.parent(dtree, dir) in directions)
                        push!(directions, ClusterTrees.parent(dtree, dir))
                        push!(Fe, Int[])
                    end
                end
            end

            tdirfars[node] = Dict(directions .=> Fe)
        end
    end
    for (i, levelidx) in enumerate(firstlevel:lastlevel)
        for node in sclink[levelidx]
            directions = zeros(Int, length(sfars[node]))
            ct = tree.nodes[node].node.data.ct
            for (faridx, far) in enumerate(sfars[node])
                directions[faridx] = NestedCrossApproximation.find_direction(
                    dtree, tree.nodes[far].node.data.ct - ct, depth - (i - 1)
                )
            end

            Fe = [
                sfars[node][findall(x -> x == dir, directions)] for
                dir in unique(directions)
            ]
            directions = unique!(directions)
            if i > 1
                for dir in keys(sdirfars[ClusterTrees.parent(tree, node)])
                    if !(ClusterTrees.parent(dtree, dir) in directions)
                        push!(directions, ClusterTrees.parent(dtree, dir))
                        push!(Fe, Int[])
                    end
                end
            end
            sdirfars[node] = Dict(directions .=> Fe)
        end
    end
    return tdirfars, sdirfars
end

function compresstestinteraction(
    cbuffer::Matrix{K},
    rbuffer::Channel{Matrix{K}},
    assembler::Function,
    testidcs::Vector{Int},
    trialidcs::Vector{Int},
    lrf::NestedCrossApproximation.iACA;
    tol=1e-4,
    maxrank=40,
) where {K}
    lm = FastBEAST.LRF.LazyMatrix(assembler, testidcs, trialidcs, K)
    locallrf = NestedCrossApproximation.init(lrf, lm)
    localrbuffer = take!(rbuffer)
    cbuffer[testidcs, 1:maxrank] .= 0
    if maxrank > min(length(testidcs), length(trialidcs))
        maxrank = min(length(testidcs), length(trialidcs))
    end
    rpivots, cpivots = locallrf(
        lm, localrbuffer, view(cbuffer, testidcs, 1:maxrank), maxrank, tol
    )
    npivots = length(rpivots)
    if any(isnan.(cbuffer[testidcs, 1:npivots])) ||
        any(.!isfinite.(cbuffer[testidcs, 1:npivots]))
        println("Fail")
    end
    cbuffer[testidcs, 1:npivots] =
        cbuffer[testidcs, 1:npivots] * localrbuffer[1:npivots, 1:npivots]
    localrbuffer[1:npivots, 1:npivots] .= 0

    put!(rbuffer, localrbuffer)
    return (testidcs[rpivots], rpivots),
    trialidcs[cpivots],
    copy(cbuffer[testidcs, 1:npivots])
end
function compresstestinteraction(
    cbuffer::Matrix{K},
    rbuffer::Channel{Matrix{K}},
    assembler::Function,
    testidcs::Vector{Int},
    trialidcs::Vector{Int},
    lrf::FastBEAST.LRF.ACA;
    tol=1e-4,
    maxrank=40,
) where {K}
    lm = FastBEAST.LRF.LazyMatrix(assembler, testidcs, trialidcs, K)
    locallrf = LRF.init(lrf, lm)
    localrbuffer = take!(rbuffer)
    cbuffer[testidcs, 1:maxrank] .= 0
    if maxrank > min(length(testidcs), length(trialidcs))
        maxrank = min(length(testidcs), length(trialidcs))
    end
    rpivots, cpivots = locallrf(
        lm, localrbuffer, view(cbuffer, testidcs, 1:maxrank), maxrank, tol
    )
    npivots = length(rpivots)
    if any(isnan.(cbuffer[testidcs, 1:npivots])) ||
        any(.!isfinite.(cbuffer[testidcs, 1:npivots]))
        println("Fail")
    end
    cbuffer[testidcs, 1:npivots] =
        cbuffer[testidcs, 1:npivots] * localrbuffer[1:npivots, 1:npivots]
    localrbuffer[1:npivots, 1:length(trialidcs)] .= 0

    put!(rbuffer, localrbuffer)
    return (testidcs[rpivots], rpivots),
    trialidcs[cpivots],
    copy(cbuffer[testidcs, 1:npivots])
end

function compresstrialinteraction(
    cbuffer::Channel{Matrix{K}},
    rbuffer::Matrix{K},
    assembler::Function,
    testidcs::Vector{Int},
    trialidcs::Vector{Int},
    lrf::NestedCrossApproximation.iACA;
    tol=1e-4,
    maxrank=40,
) where {K}
    lm = FastBEAST.LRF.LazyMatrix(assembler, testidcs, trialidcs, K)
    locallrf = NestedCrossApproximation.init(lrf, lm)
    localcbuffer = take!(cbuffer)
    rbuffer[1:maxrank, trialidcs] .= 0
    if maxrank > min(length(testidcs), length(trialidcs))
        maxrank = min(length(testidcs), length(trialidcs))
    end
    rpivots, cpivots = locallrf(
        lm, view(rbuffer, 1:maxrank, trialidcs), localcbuffer, maxrank, tol
    )
    npivots = length(rpivots)

    rbuffer[1:npivots, trialidcs] =
        localcbuffer[1:npivots, 1:npivots] * rbuffer[1:npivots, trialidcs]
    localcbuffer[1:npivots, 1:npivots] .= 0

    put!(cbuffer, localcbuffer)
    return (trialidcs[cpivots], cpivots),
    testidcs[rpivots],
    copy(rbuffer[1:npivots, trialidcs])
end

function compresstrialinteraction(
    cbuffer::Channel{Matrix{K}},
    rbuffer::Matrix{K},
    assembler::Function,
    testidcs::Vector{Int},
    trialidcs::Vector{Int},
    lrf::FastBEAST.LRF.ACA;
    tol=1e-4,
    maxrank=40,
) where {K}
    lm = FastBEAST.LRF.LazyMatrix(assembler, testidcs, trialidcs, K)
    locallrf = LRF.init(lrf, lm)
    localcbuffer = take!(cbuffer)
    rbuffer[1:maxrank, trialidcs] .= 0
    if maxrank > min(length(testidcs), length(trialidcs))
        maxrank = min(length(testidcs), length(trialidcs))
    end
    rpivots, cpivots = locallrf(
        lm, view(rbuffer, 1:maxrank, trialidcs), localcbuffer, maxrank, tol
    )
    npivots = length(rpivots)

    rbuffer[1:npivots, trialidcs] =
        localcbuffer[1:npivots, 1:npivots] * rbuffer[1:npivots, trialidcs]
    localcbuffer[1:length(testidcs), 1:npivots] .= 0

    put!(cbuffer, localcbuffer)
    return (trialidcs[cpivots], cpivots),
    testidcs[rpivots],
    copy(rbuffer[1:npivots, trialidcs])
end

function directionalcompressor(
    assembler::Function,
    tree::NminTree{D},
    hffars::Vector{Vector{Tuple{Int,Int}}},
    k::F;
    tlrf=LRF.ACA(),
    slrf=LRF.ACA(),
    maxrank=40,
    tol=1e-4,
    ηₕ=5.0,
    hs=(data(tree, root(tree)).hs / 2^(findfirst(!=([]), hffars) - 1)) / √3,
    multithreading=true,
) where {F,D}
    dtree = NestedCrossApproximation.directionaltree(hs, k, ηₕ)
    tdirfars, sdirfars = deriveinformation(tree, hffars, dtree)
    tclink = FastBEAST.cluster_link(tree)
    sclink = tclink
    firstlevel = findfirst(!=([]), hffars)
    lastlevel = findlast(!=([]), hffars)
    tpivots = Vector{Dict{Int,Tuple{Vector{Int},Vector{Int}}}}(undef, length(tdirfars))
    tbases = Vector{Dict{Int,Matrix{ComplexF64}}}(undef, length(tdirfars))
    Ftpivots = Vector{Dict{Int,Vector{Int}}}(undef, length(tdirfars))
    spivots = Vector{Dict{Int,Tuple{Vector{Int},Vector{Int}}}}(undef, length(sdirfars))
    sbases = Vector{Dict{Int,Matrix{ComplexF64}}}(undef, length(sdirfars))
    Fspivots = Vector{Dict{Int,Vector{Int}}}(undef, length(sdirfars))

    buffer = Channel{Matrix{ComplexF64}}(Threads.nthreads())
    for _ in 1:Threads.nthreads()
        if tlrf isa iACA
            put!(buffer, zeros(ComplexF64, maxrank, maxrank))
        else
            put!(buffer, zeros(ComplexF64, maxrank, tree.num_elements))
        end
    end
    basebuffer = zeros(ComplexF64, tree.num_elements, maxrank)
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    for (i, levelidx) in enumerate(firstlevel:lastlevel)
        # println("Level: ", levelidx)
        _foreach(tclink[levelidx]) do node
            nodepivots = Tuple{Vector{Int},Vector{Int}}[]
            nodeFpivots = Vector{Int}[]
            nodebases = Matrix{ComplexF64}[]
            for (ℯ, Fℯ) in tdirfars[node]
                Ftidcs = value(tree, Fℯ)
                for childdir in ClusterTrees.children(dtree, ℯ)
                    if i > 1 && haskey(Ftpivots[ClusterTrees.parent(tree, node)], childdir)
                        append!(Ftidcs, Ftpivots[ClusterTrees.parent(tree, node)][childdir])
                    end
                end
                tidcs = value(tree, node)
                pivots, Fpivots, basis = compresstestinteraction(
                    basebuffer,
                    buffer,
                    assembler,
                    tidcs,
                    Ftidcs,
                    tlrf;
                    tol=tol,
                    maxrank=maxrank,
                )
                push!(nodepivots, pivots)
                push!(nodeFpivots, Fpivots)
                push!(nodebases, basis)
            end
            tpivots[node] = Dict(keys(tdirfars[node]) .=> nodepivots)
            Ftpivots[node] = Dict(keys(tdirfars[node]) .=> nodeFpivots)
            tbases[node] = Dict(keys(tdirfars[node]) .=> nodebases)
        end
    end
    buffer = Channel{Matrix{ComplexF64}}(Threads.nthreads())
    for _ in 1:Threads.nthreads()
        if slrf isa iACA
            put!(buffer, zeros(ComplexF64, maxrank, maxrank))
        else
            put!(buffer, zeros(ComplexF64, tree.num_elements, maxrank))
        end
    end
    basebuffer = zeros(ComplexF64, maxrank, tree.num_elements)
    for (i, levelidx) in enumerate(firstlevel:lastlevel)
        _foreach(sclink[levelidx]) do node
            nodepivots = Tuple{Vector{Int},Vector{Int}}[]
            nodeFpivots = Vector{Int}[]
            nodebases = Matrix{ComplexF64}[]
            for (ℯ, Fℯ) in sdirfars[node]
                Fsidcs = value(tree, Fℯ)
                for childdir in ClusterTrees.children(dtree, ℯ)
                    if i > 1 && haskey(Fspivots[ClusterTrees.parent(tree, node)], childdir)
                        append!(Fsidcs, Fspivots[ClusterTrees.parent(tree, node)][childdir])
                    end
                end
                Fsidcs == [] && continue
                sidcs = value(tree, node)
                pivots, Fpivots, basis = compresstrialinteraction(
                    buffer,
                    basebuffer,
                    assembler,
                    Fsidcs,
                    sidcs,
                    slrf;
                    tol=tol,
                    maxrank=maxrank,
                )
                push!(nodepivots, pivots)
                push!(nodeFpivots, Fpivots)
                push!(nodebases, basis)
            end
            spivots[node] = Dict(keys(sdirfars[node]) .=> nodepivots)
            Fspivots[node] = Dict(keys(sdirfars[node]) .=> nodeFpivots)
            sbases[node] = Dict(keys(sdirfars[node]) .=> nodebases)
        end
    end

    return tbases, tpivots, tdirfars, Ftpivots, sbases, spivots, sdirfars, Fspivots, dtree
end

function builddirectionalH2(
    tree, dtree, hffars, tmat, tpivots, smat, spivots; multithreading=false
)
    lk = Threads.SpinLock()
    tbases = Dict{Int,Matrix{ComplexF64}}[]
    tbidcs = Int[]
    ttransfer = Dict{Int,Dict{Int,Vector{Matrix{ComplexF64}}}}[]

    sbases = Dict{Int,Matrix{ComplexF64}}[]
    sbidcs = Int[]
    stransfer = Dict{Int,Dict{Int,Vector{Matrix{ComplexF64}}}}[]

    tclink = FastBEAST.cluster_link(tree)
    sclink = tclink
    firstlevel = findfirst(!=([]), hffars)
    lastlevel = findlast(!=([]), hffars)
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    for levelidx in firstlevel:lastlevel
        leveledttransfer = Dict{Int,Vector{Matrix{ComplexF64}}}[]
        ttidcs = Int[]
        leveledstransfer = Dict{Int,Vector{Matrix{ComplexF64}}}[]
        stidcs = Int[]

        _foreach(tclink[levelidx]) do node
            #println("level: ", levelidx, ", lastlevel: ", lastlevel)
            if !(ClusterTrees.haschildren(tree, node) && levelidx != lastlevel)
                bases = Matrix{ComplexF64}[]
                for (dir, dirmat) in tmat[node]
                    basis = dirmat / dirmat[tpivots[node][dir][2], :]
                    push!(bases, basis)
                end
                lock(lk) do
                    push!(tbases, Dict(keys(tmat[node]) .=> bases))
                    push!(tbidcs, node)
                end
            else
                transfer = Vector{Matrix{ComplexF64}}[]
                for (dir, dirmat) in tmat[node]
                    childtransfer = Matrix{ComplexF64}[]
                    for child in ClusterTrees.children(tree, node)
                        τ = [
                            findfirst(x -> x == cpp, value(tree, node)) for
                            cpp in tpivots[child][ClusterTrees.parent(dtree, dir)][1]
                        ]
                        Θ = dirmat[τ, :] / dirmat[tpivots[node][dir][2], :]
                        push!(childtransfer, Θ)
                    end
                    push!(transfer, childtransfer)
                end
                lock(lk) do
                    push!(leveledttransfer, Dict(keys(tmat[node]) .=> transfer))
                    push!(ttidcs, node)
                end
            end
        end
        push!(ttransfer, Dict(ttidcs .=> leveledttransfer))
        _foreach(sclink[levelidx]) do node
            if !(ClusterTrees.haschildren(tree, node) && levelidx != lastlevel)
                bases = Matrix{ComplexF64}[]
                for (dir, dirmat) in smat[node]
                    basis = dirmat[:, spivots[node][dir][2]] \ dirmat
                    push!(bases, basis)
                end

                lock(lk) do
                    push!(sbases, Dict(keys(smat[node]) .=> bases))
                    push!(sbidcs, node)
                end
            else
                transfer = Vector{Matrix{ComplexF64}}[]
                for (dir, dirmat) in smat[node]
                    childtransfer = Matrix{ComplexF64}[]
                    for child in ClusterTrees.children(tree, node)
                        σ = [
                            findfirst(x -> x == cpp, value(tree, node)) for
                            cpp in spivots[child][ClusterTrees.parent(dtree, dir)][1]
                        ]
                        Θ = dirmat[:, spivots[node][dir][2]] \ dirmat[:, σ]

                        push!(childtransfer, Θ)
                    end
                    push!(transfer, childtransfer)
                end
                lock(lk) do
                    push!(leveledstransfer, Dict(keys(smat[node]) .=> transfer))
                    push!(stidcs, node)
                end
            end
        end
        push!(stransfer, Dict(stidcs .=> leveledstransfer))
    end

    return Dict(tbidcs .=> tbases), ttransfer, Dict(sbidcs .=> sbases), stransfer
end
