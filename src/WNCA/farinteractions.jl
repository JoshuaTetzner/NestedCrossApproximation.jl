function directionalitneractions(tree, dtree, islf, isnear; ntasks=Threads.nthreads())
    lk = Threads.SpinLock()
    values, nearvalues = H2Trees.nearinteractions(
        tree; isnear=isnear, extractselfvalues=false
    )
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)
    fars = Vector{Tuple{Int,Int}}[]
    lfvalues = Vector{Int}[]
    lffarvalues = Vector{Vector{Int}}[]
    dirs = Vector{Int}[]

    for level in H2Trees.levels(H2Trees.testtree(tree))
        levelfars = Tuple{Int,Int}[]
        leveldirs = Int[]
        testclusters = collect(H2Trees.LevelIterator(H2Trees.testtree(tree), level))
        @tasks for t in testclusters
            @set ntasks = 1

            lfclusterfars = Vector{Int}[]
            for s in iterator(H2Trees.trialtree(tree), H2Trees.testtree(tree), t)
                dir = 0
                (!islf(tree, level)) && (
                    dir = (direction(
                        H2Trees.center(tree.trialcluster, s) -
                        H2Trees.center(tree.testcluster, t),
                        dtree,
                        dtree.level + 1 - level,
                    ))
                )
                if dir == 0
                    push!(lfclusterfars, H2Trees.values(tree.trialcluster, s))
                else
                    lock(lk) do
                        push!(levelfars, (t, s))
                        push!(leveldirs, dir)
                    end
                end
            end
            if islf(tree, level)
                lock(lk) do
                    push!(lfvalues, H2Trees.values(tree.testcluster, t))
                    push!(lffarvalues, lfclusterfars)
                end
            end
        end
        push!(fars, levelfars)
        push!(dirs, leveldirs)
    end
    return values, nearvalues, fars, dirs, lfvalues, lffarvalues
end

##
function directionalitneractions2(tree, dtree, islf, isnear; ntasks=Threads.nthreads())
    lk = Threads.SpinLock()
    nears = Tuple{Int,Int}[]
    fars = Vector{Tuple{Int,Int}}[]
    lfvalues = Vector{Int}[]
    lffarvalues = Vector{Vector{Int}}[]
    dirs = Vector{Tuple{Int,Int}}[]

    for level in H2Trees.levels(H2Trees.testtree(tree))
        levelfars = Tuple{Int,Int}[]
        leveldirs = Tuple{Int,Int}[]
        testclusters = collect(H2Trees.LevelIterator(H2Trees.testtree(tree), level))
        @tasks for t in testclusters
            @set ntasks = 1

            lfclusterfars = Vector{Int}[]
            for s in H2Trees.LevelIterator(H2Trees.trialtree(tree), level)
                if !isnear(tree.testcluster, tree.trialcluster, t, s) && isnear(
                    tree.testcluster,
                    tree.trialcluster,
                    H2Trees.parent(tree.testcluster, t),
                    H2Trees.parent(tree.trialcluster, s),
                )
                    if islf(tree, level)
                        lock(lk) do
                            push!(lfclusterfars, H2Trees.values(tree.trialcluster, s))
                        end
                    else
                        dir = (0, 0)
                        !islf(tree, level) && (
                            dir = (
                                direction(
                                    H2Trees.center(tree.trialcluster, s) -
                                    H2Trees.center(tree.testcluster, t),
                                    dtree,
                                    dtree.level + 1 - level,
                                ),
                                direction(
                                    H2Trees.center(tree.testcluster, t) -
                                    H2Trees.center(tree.trialcluster, s),
                                    dtree,
                                    dtree.level + 1 - level,
                                ),
                            )
                        )
                        if dir[1] == (0, 0)
                            println("You do not want to be here.")
                            push!(lfclusterfars, H2Trees.values(tree.trialcluster, s))
                        else
                            lock(lk) do
                                push!(levelfars, (t, s))
                                push!(leveldirs, dir)
                            end
                        end
                    end
                elseif isnear(tree.testcluster, tree.trialcluster, t, s) && (
                    iszero(H2Trees.firstchild(tree.testcluster, t)) ||
                    iszero(H2Trees.firstchild(tree.trialcluster, s))
                )
                    lock(lk) do
                        push!(nears, (t, s))
                    end
                end
            end
            if lffarvalues != []
                push!(lffarvalues, lfclusterfars)
                push!(lfvalues, H2Trees.values(tree.trialcluster, s))
            end
        end
        push!(fars, levelfars)
        push!(dirs, leveldirs)
    end
    return nears, fars, dirs, lfvalues, lffarvalues
end
