
function leveledtestfarfield(tree, levelfars::Vector{Vector{Tuple{Int,Int}}})
    sortedfars = Dict{Int,Vector{Int}}[]
    for (level, fars) in enumerate(levelfars)
        sort!(fars)
        nodes = Int[]
        sfars = Vector{Int}[]
        for far in fars
            if far[1] in nodes
                push!(sfars[end], far[2])
            else
                push!(nodes, far[1])
                if ClusterTrees.parent(tree, far[1]) != 0 &&
                    haskey(sortedfars[level - 1], ClusterTrees.parent(tree, far[1]))
                    push!(sfars, sortedfars[level - 1][ClusterTrees.parent(tree, far[1])])
                    push!(sfars[end], far[2])
                else
                    push!(sfars, [far[2]])
                end
            end
        end
        push!(sortedfars, Dict(nodes .=> sfars))
    end

    return sortedfars
end

function leveledtrialfarfield(tree, levelfars::Vector{Vector{Tuple{Int,Int}}})
    sortedfars = Dict{Int,Vector{Int}}[]
    for (level, fars) in enumerate(levelfars)
        sort!(fars; by=x -> x[2])
        nodes = Int[]
        sfars = Vector{Int}[]
        for far in fars
            if far[2] in nodes
                push!(sfars[end], far[1])
            else
                push!(nodes, far[2])
                if ClusterTrees.parent(tree, far[2]) != 0 &&
                    haskey(sortedfars[level - 1], ClusterTrees.parent(tree, far[2]))
                    push!(sfars, sortedfars[level - 1][ClusterTrees.parent(tree, far[2])])
                    push!(sfars[end], far[1])
                else
                    push!(sfars, [far[1]])
                end
            end
        end
        push!(sortedfars, Dict(nodes .=> sfars))
    end

    return sortedfars
end

function assemble_coupling(
    farassembler, fars::Vector{Tuple{Int,Int}}, τt::Matrix{Int}, σs::Matrix{Int}, k::Int, K
)
    coupling = [zeros(K, k, k) for _ in 1:length(fars)]
    for (i, far) in enumerate(fars)
        @views farassembler(coupling[i], τt[far[1], 1:k], σs[far[2], 1:k])
    end

    return coupling
end

function testmv(
    testvector::Vector{K},
    resultvector::Vector{K},
    rowbuffer::Matrix{K},
    colbuffer::Matrix{K},
    coupling::Vector{Matrix{K}},
    testfarfield::Dict{Int,Vector{Int}},
    τt::Matrix{Int},
    trialfarfield::Dict{Int,Vector{Int}},
    σs::Matrix{Int},
    fars::Vector{Tuple{Int,Int}},
    testtree,
    trialtree,
    k::I,
) where {I,K}
    xhat = Vector{Vector{eltype(rowbuffer)}}(undef, length(trialtree.nodes))
    yhat = Vector{Vector{eltype(rowbuffer)}}(undef, length(testtree.nodes))

    for (s, farfield) in trialfarfield
        xhat[s] =
            pinv(rowbuffer[1:k, σs[s, 1:k]]) *
            rowbuffer[1:k, value(trialtree, s)] *
            testvector[value(trialtree, s)]
    end

    for (ind, (t, s)) in enumerate(fars)
        if isassigned(yhat, t)
            yhat[t] += coupling[ind] * xhat[s]
        else
            yhat[t] = coupling[ind] * xhat[s]
        end
    end

    for (t, farfield) in testfarfield
        resultvector[value(testtree, t)] +=
            colbuffer[value(testtree, t), 1:k] * pinv(colbuffer[τt[t, 1:k], 1:k]) * yhat[t]
    end

    return resultvector
end

function PetrovGalerkinMCA(
    operator,
    testspace,
    trialspace;
    testtree=create_tree(testspace.pos, KMeansTreeOptions(; nmin=50, maxlevel=50)),
    trialtree=create_tree(trialspace.pos, KMeansTreeOptions(; nmin=50, maxlevel=50)),
    nearinteractionquadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
    momentquadstrat=BEAST.DoubleNumQStrat(2, 3),
    multithreading=true,
    maxrank=40,     #Should be moved to the compressor
    tol=1e-4,       #global
    η=1.0,          #global
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

    testvector = rand(scalartype(operator), length(trialspace.pos))
    refproduct = nearinteractions * testvector

    @views farblkassembler = BEAST.blockassembler(
        operator, testspace, trialspace; quadstrat=momentquadstrat
    )
    @views function farassembler(Z, tdata, sdata)
        @views store(v, m, n) = (Z[m, n] += v)
        return farblkassembler(tdata, sdata, store)
    end
    test_farfield = leveledtestfarfield(testtree, fars)
    trial_farfield = leveledtrialfarfield(trialtree, fars)
    τt = zeros(Int, length(testspace.pos), maxrank)
    σt = zeros(Int, length(testspace.pos), maxrank)
    τs = zeros(Int, length(trialspace.pos), maxrank)
    σs = zeros(Int, length(trialspace.pos), maxrank)

    colbuffer = zeros(scalartype(operator), length(testspace.pos), maxrank)
    rowbuffer = zeros(scalartype(operator), maxrank, length(trialspace.pos))

    for level in eachindex(test_farfield)
        println("Level: ", level)
        resultvector = copy(refproduct)
        if length(test_farfield[level]) != 0
            for k in 1:10
                for (t, Ft) in test_farfield[level]
                    τt[t, k] = rand(
                        value(testtree, t)[filter(
                            x -> !(x in τt[t, 1:(k - 1)]), eachindex(value(testtree, t))
                        )],
                    )
                    σt[t, k] = rand(
                        value(trialtree, Ft)[filter(
                            x -> !(x in σt[t, 1:(k - 1)]), eachindex(value(trialtree, Ft))
                        )],
                    )
                    @views farassembler(
                        colbuffer[value(testtree, t), k:k],
                        value(testtree, t),
                        σt[t, k]:σt[t, k],
                    )
                end
                for (s, Fs) in trial_farfield[level]
                    τs[s, k] = rand(
                        value(testtree, Fs)[filter(
                            x -> !(x in τs[s, 1:(k - 1)]), eachindex(value(testtree, Fs))
                        )],
                    )
                    σs[s, k] = rand(
                        value(trialtree, s)[filter(
                            x -> !(x in σs[s, 1:(k - 1)]), eachindex(value(trialtree, s))
                        )],
                    )
                    @views farassembler(
                        rowbuffer[k:k, value(trialtree, s)],
                        τs[s, k]:τs[s, k],
                        value(trialtree, s),
                    )
                end

                coupling = assemble_coupling(
                    farassembler, fars[level], τt, σs, k, scalartype(operator)
                )

                newvector = testmv(
                    testvector,
                    copy(refproduct),
                    rowbuffer,
                    colbuffer,
                    coupling,
                    test_farfield[level],
                    τt,
                    trial_farfield[level],
                    σs,
                    fars[level],
                    testtree,
                    trialtree,
                    k,
                )
                println(norm(resultvector - newvector) / norm(resultvector))
                resultvector = copy(newvector)
            end
        end
    end
end

##
