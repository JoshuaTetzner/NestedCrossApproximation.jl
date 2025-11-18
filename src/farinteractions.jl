function farinteractions(tree; isnear=H2Trees.isnear, ntasks=Threads.nthreads())
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)
    testfarnodes = Vector{Vector{Int}}(undef, length(tree.testcluster.nodes))
    trialfarnodes = Vector{Vector{Int}}(undef, length(tree.trialcluster.nodes))
    for level in levels(testtree(tree))
        @tasks for node in LevelIterator(testtree(tree), level)
            @set ntasks = ntasks
            testfarnodes[node] = collect(iterator(trialtree(tree), testtree(tree), node))
        end
    end
    for level in levels(H2Trees.trialtree(tree))
        @tasks for node in LevelIterator(trialtree(tree), level)
            @set ntasks = ntasks
            trialfarnodes[node] = collect(iterator(trialtree(tree), trialtree(tree), node))
        end
    end
    return testfarnodes, trialfarnodes
end

function directionalfarinteractions(
    tree, dtree; isnear=H2Trees.isnear, ntasks=Threads.nthreads()
)
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)
    testdirs = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    trialdirs = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    testfarnodes = Vector{Vector{Int}}(undef, length(tree.testcluster.nodes))
    trialfarnodes = Vector{Vector{Int}}(undef, length(tree.trialcluster.nodes))

    for level in levels(testtree(tree))
        @tasks for t in LevelIterator(testtree(tree), level)
            @set ntasks = ntasks
            testfarnodes[t] = collect(iterator(trialtree(tree), testtree(tree), t))
            testdirs[t] = map(testfarnodes[t]) do s
                direction(
                    center(trialtree(tree), s) - center(testtree(tree), t),
                    dtree,
                    max(0, dtree.level + 1 - level),
                )
            end
        end
    end

    for level in levels(trialtree(tree))
        @tasks for s in LevelIterator(trialtree(tree), level)
            @set ntasks = ntasks
            trialfarnodes[s] = collect(iterator(testtree(tree), trialtree(tree), s))
            trialdirs[s] = map(trialfarnodes[s]) do t
                direction(
                    center(trialtree(tree), s) - center(testtree(tree), t),
                    dtree,
                    max(0, dtree.level + 1 - level),
                )
            end
        end
    end
    return testfarnodes, testdirs, trialfarnodes, trialdirs
end

function bottomupfars!(tree, dtree, F, e; ntasks=Threads.nthreads())
    for level in levels(tree)
        level == 1 && continue
        !isroot(dtree, level) && continue
        @tasks for t in LevelIterator(tree, level)
            @set ntasks = ntasks
            append!(F[t], F[parent(tree, t)])
            append!(e[t], map(dir -> parent(dtree, dir), e[parent(tree, t)]))
        end
    end
end

function admissiblelevel(Ft, eₜ, tree::BlockTree)
    lflevel = 0
    hflevel = 0
    for level in levels(testtree(tree))
        for t in LevelIterator(testtree(tree), level)
            length(Ft[t]) > 0 && if eₜ[t][1] == 0
                lflevel += 1
                break
            else
                hflevel += 1
                break
            end
        end
    end
    return max(lflevel, hflevel)
end
