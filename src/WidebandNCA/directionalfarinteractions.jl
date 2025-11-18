function directionalfarinteractions(tree, dtree, ishfnear; ntasks=Threads.nthreads())
    iterator = WellSeparatedIterator(; isnear=(tree) -> ishfnear)(tree)
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
                    dtree.level + 1 - level,
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
                    center(testtree(tree), t) - center(trialtree(tree), s),
                    dtree,
                    dtree.level + 1 - level,
                )
            end
        end
    end
    return testfarnodes, testdirs, trialfarnodes, trialdirs
end

function admissiblelevel(Ft, tree)
    lflevel = 0
    hflevel = 0
    for level in H2Trees.levels(tree.testcluster)
        for t in H2Trees.LevelIterator(tree.testcluster, level)
            if length(Ft[t]) > 0
                if eₜ[t][1] == 0
                    lflevel += 1
                    break
                else
                    hflevel += 1
                    break
                end
            end
        end
    end
    return max(lflevel, hflevel)
end
