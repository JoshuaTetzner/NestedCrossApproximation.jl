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
            @set ntasks = ntasks
            lfclusterfars = Vector{Int}[]
            for s in iterator(H2Trees.testtree(tree), H2Trees.trialtree(tree), t)
                dir = 0
                !islf(tree, level) && (
                    dir = direction(
                        H2Trees.center(tree.trialcluster, s) -
                        H2Trees.center(tree.testcluster, t),
                        dtree,
                        dtree.level + 1 - level,
                    )
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